"""Step 0a of the trans background: measure an unchanged pysceptre on a real screen.

Runs against whichever pysceptre is on PYTHONPATH (for Step 0a the 0.5.0
source) and changes no value it computes. Engine functions are wrapped at
runtime to log what each call costs, and width-1 gene fits are reused across
runs from a cache, which is exact because such a fit depends only on its
gene. Every path comes from the command line; a subcommand writes only under
its `--out` directory, plus the gene-fit cache named by `--fit-cache`.

  specificity_step0.py env --out DIR
  specificity_step0.py qc-crosscheck --dataset H5MU --out DIR
  specificity_step0.py gene-fits --dataset H5MU --fit-cache NPZ --n-genes N --n-jobs J --seed S --out DIR
  specificity_step0.py cis-slice --dataset H5MU --tsv CIS.tsv --fit-cache NPZ --chunks LIST --n-jobs J --out DIR
  specificity_step0.py trans-slice --dataset H5MU --tsv CIS.tsv --gtf GTF --fit-cache NPZ
      --chunks LIST --n-genes N --k K --n-jobs J --out DIR
  specificity_step0.py b1-blocks --dataset H5MU --tsv CIS.tsv --fit-cache NPZ --chunks LIST --out DIR
  specificity_step0.py summarize --logs DIR --out DIR

`env` records the software and the node (versions, BLAS, numpy CPU dispatch,
CPU model, transparent huge pages, NUMA layout). `qc-crosscheck` recounts the
cis pairs' nonzero cells with `pairwise_qc.nonzero_counts` and compares them
with the export's `discovery_pairs_with_info`. `gene-fits` fits random cis
genes with the engine's own width-1 call. `cis-slice` reruns the cis pairs of
whole reference target chunks and compares all ten output columns with the
reference result, value for value. `trans-slice` tests the same targets
against random genes on other chromosomes. `b1-blocks` compares B1 null
statistics computed in row blocks with the full product, bit for bit.
`summarize` turns the call logs of one or more runs into cost tables.

Reference chunks are the target chunks of the reference (cis) run, rebuilt
the way the engine builds them: the targets of `grna_target_cells` that have
a pair, in dict order, cut into consecutive slices of the width the memory
budget gives. `--chunks` names them by 0-based index, or `default`: evenly
spaced chunks plus chunks holding stage-3 and stage-2 reference pairs.

Call logs are JSON lines, one file per process (`calls_<host>_<pid>.jsonl`),
written a line at a time because forked pool workers exit without running
exit handlers. Needs the `io` extra (mudata) through `sceptre_io.load_export`,
taken from `--sceptre-io-dir`, else from the `scripts/` directory next to the
imported pysceptre. Run it in a batch job: it loads the whole dataset. See
docs/design.md, "The trans background".
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pysceptre
from pysceptre.glm.irls import one_blas_thread
from pysceptre.pipeline import discovery
from pysceptre.pipeline.api import _resampling_budget, run_discovery_analysis
from pysceptre.pipeline.pairwise_qc import nonzero_counts
from pysceptre.test_statistic import resampling, score_stat

_RESULT_COLUMNS = [
    "response_id",
    "grna_target",
    "p_value",
    "fold_change",
    "se_fold_change",
    "pct_change_es",
    "pct_change_es_ci_low",
    "pct_change_es_ci_high",
    "z_orig",
    "stage",
]
_KEYS = ["response_id", "grna_target"]
_VALUE_COLUMNS = _RESULT_COLUMNS[2:]
_SIDES = {-1: "left", 0: "both", 1: "right"}
_ENV_KEYS = (
    "PYSCEPTRE_BACKEND",
    "PYSCEPTRE_PREFETCH_DEPTH",
    "NUMPY_MADVISE_HUGEPAGE",
    "MALLOC_TRIM_THRESHOLD_",
    "MALLOC_MMAP_THRESHOLD_",
    "MALLOC_MMAP_MAX_",
    "MALLOC_ARENA_MAX",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_CACHE_DIR",
    "PYTHONPATH",
    "SLURM_JOB_ID",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
)
_TARGET_NAME = re.compile(r"^(chr[0-9XYM]+):(\d+)-(\d+)$")
_GENE_ID = re.compile(r'gene_id "([^"]+)"')
_CANONICAL = {f"chr{c}" for c in [*range(1, 23), "X", "Y", "M"]}


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _finite_or_none(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _finite_or_none(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _as_bool(x) -> float:
    """1.0 or 0.0 for a logical however the export encoded it, NaN when missing."""
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    if isinstance(x, (int, np.integer)) and x in (0, 1):
        return float(x)
    if isinstance(x, str) and x.strip().upper() in ("TRUE", "FALSE"):
        return float(x.strip().upper() == "TRUE")
    return float("nan")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------

# Per-process state. A forked child gets a fresh lock and its own file (see
# `_reset_after_fork`), and keeps the inherited file object alive so that
# closing it cannot flush the parent's buffer into the parent's file.
_LOG: dict = {"dir": None, "pid": None, "fh": None, "lock": threading.Lock(), "inv": None}
_INHERITED: list = []
_TLS = threading.local()
_ORIGINAL: dict = {}


def _reset_after_fork() -> None:
    if _LOG["fh"] is not None:
        _INHERITED.append(_LOG["fh"])
    _LOG["pid"] = None
    _LOG["fh"] = None
    _LOG["lock"] = threading.Lock()
    _TLS.depth = 0
    _TLS.phase = None


os.register_at_fork(after_in_child=_reset_after_fork)


def _emit(record: dict) -> None:
    """Append one record to this process's call log, flushed at once."""
    if _LOG["dir"] is None:
        return
    line = json.dumps(record, default=_json_default) + "\n"
    with _LOG["lock"]:
        if _LOG["pid"] != os.getpid():
            _LOG["pid"] = os.getpid()
            path = Path(_LOG["dir"]) / f"calls_{socket.gethostname()}_{os.getpid()}.jsonl"
            _LOG["fh"] = open(path, "a", buffering=1)
        _LOG["fh"].write(line)


def _timed(name: str, fn, extra=None, phase: str | None = None):
    """`fn`, logging each call's wall time, thread CPU, faults and `extra(args, kwargs, result)`.

    Calls through and returns the result unchanged. `phase` labels this call
    and every logged call nested in it on the same thread. `depth` counts the
    logged calls the thread is already inside, so totals can skip nested ones.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        depth = getattr(_TLS, "depth", 0)
        outer = getattr(_TLS, "phase", None)
        here = phase if phase is not None else outer
        _TLS.depth = depth + 1
        _TLS.phase = here
        r0 = resource.getrusage(resource.RUSAGE_THREAD)
        c0 = time.thread_time()
        t_start = time.time()
        w0 = time.perf_counter()
        result = None
        error = None
        try:
            result = fn(*args, **kwargs)
            return result
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            wall = time.perf_counter() - w0
            cpu = time.thread_time() - c0
            r1 = resource.getrusage(resource.RUSAGE_THREAD)
            _TLS.depth = depth
            _TLS.phase = outer
            record = {
                "fn": name,
                "wall_s": wall,
                "thread_cpu_s": cpu,
                "utime_s": r1.ru_utime - r0.ru_utime,
                "stime_s": r1.ru_stime - r0.ru_stime,
                "minflt": r1.ru_minflt - r0.ru_minflt,
                "majflt": r1.ru_majflt - r0.ru_majflt,
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "t_start": t_start,
                "inv": _LOG["inv"],
                "phase": here,
                "depth": depth,
                "extra": {},
            }
            if error is not None:
                record["extra"] = {"error": error}
            elif extra is not None:
                try:
                    record["extra"] = extra(args, kwargs, result)
                except Exception as exc:  # a logging problem must not stop the run
                    record["extra"] = {"extra_error": repr(exc)}
            _emit(record)

    return wrapper


def _arg(args, kwargs, i: int, name: str):
    return kwargs[name] if name in kwargs else args[i]


def _x_batch(args, kwargs, result):
    Y = np.asarray(_arg(args, kwargs, 1, "Y"))
    return {
        "k": int(Y.shape[0]) if Y.ndim == 2 else 1,
        "n_iter_max": int(np.max(result.n_iter)),
        "converged": int(np.sum(result.converged)),
    }


def _x_theta(args, kwargs, result):
    return {"method": int(result[1])}


def _x_sampler(args, kwargs, result):
    return {"B": int(_arg(args, kwargs, 1, "B")), "n_trt": int(_arg(args, kwargs, 3, "n_trt"))}


def _x_test(args, kwargs, result):
    return {"stage": _finite_or_none(result.stage), "n_trt": int(len(kwargs["trt_idxs"]))}


def _x_nulls(args, kwargs, result):
    draws = _arg(args, kwargs, 1, "draws")
    first = draws[0] if isinstance(draws, tuple) else draws
    return {"rows": int(first.shape[0]), "nnz": int(first.nnz)}


def _x_skew(args, kwargs, result):
    return {"used": bool(result.used)}


def _x_draw_matrix(args, kwargs, result):
    return {"rows": int(result.shape[0]), "nnz": int(result.nnz)}


def _x_targets(args, kwargs, result):
    cells = _arg(args, kwargs, 0, "grna_target_cells")
    return {"n_targets": len(cells), "targets": list(cells)}


def _x_gene_job(args, kwargs, result):
    gene_id, targets = args[0]
    return {"gene": gene_id, "n_targets": len(targets)}


def _x_gene_fit_job(args, kwargs, result):
    return {"genes": list(args[0][1])}


# (module, global name, extra, phase). Module globals are looked up at call
# time, so replacing them reaches every caller in that module, forked
# workers included; pickling a wrapped worker function finds the wrapper
# under the original name, which is what the process pool needs.
_WRAPPED = (
    (discovery, "fit_poisson_glm_batch", _x_batch, None),
    (discovery, "estimate_theta", _x_theta, None),
    (discovery, "fit_binomial_glm_batch", _x_batch, "target_prep"),
    (discovery, "crt_index_sampler", _x_sampler, "target_prep"),
    (discovery, "compute_precomputation_pieces", None, None),
    (discovery, "stack_pieces", None, None),
    (discovery, "run_low_level_test_full", _x_test, None),
    (discovery, "fit_all_targets", _x_targets, "target_prep"),
    (discovery, "_gene_job", _x_gene_job, "gene_test"),
    (discovery, "_gene_fit_job", _x_gene_fit_job, "gene_fit"),
    (resampling, "compute_null_statistics_from_draws", _x_nulls, None),
    (resampling, "fit_and_evaluate_skew_normal", _x_skew, None),
    (score_stat, "draws_to_matrix", _x_draw_matrix, None),
)


def install_instrumentation(log_dir: Path) -> None:
    """Wrap the engine functions in `_WRAPPED` so every call is logged under `log_dir`."""
    _LOG["dir"] = str(log_dir)
    for module, name, extra, phase in _WRAPPED:
        key = f"{module.__name__}.{name}"
        if key in _ORIGINAL:
            continue
        _ORIGINAL[key] = getattr(module, name)
        setattr(module, name, _timed(name, _ORIGINAL[key], extra, phase))


@contextlib.contextmanager
def invocation(name: str, n_jobs: int, **extra):
    """Log one top-level call: wall time and process CPU (self and reaped children).

    Records logged inside it carry its id, which is what `summarize` uses to
    relate per-call thread CPU to the invocation's wall time. Yields a dict
    whose entries are added to the record's `extra`.
    """
    inv_id = f"{name}:{os.getpid()}:{time.time_ns()}"
    _LOG["inv"] = inv_id
    s0 = resource.getrusage(resource.RUSAGE_SELF)
    k0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t_start = time.time()
    w0 = time.perf_counter()
    info: dict = {}
    _say(f"start {name} (n_jobs={n_jobs})")
    try:
        yield info
    finally:
        wall = time.perf_counter() - w0
        s1 = resource.getrusage(resource.RUSAGE_SELF)
        k1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        record = {
            "fn": "invocation",
            "name": name,
            "inv": inv_id,
            "wall_s": wall,
            "t_start": t_start,
            "n_jobs": n_jobs,
            "backend": discovery.parallel_backend(),
            "self_utime_s": s1.ru_utime - s0.ru_utime,
            "self_stime_s": s1.ru_stime - s0.ru_stime,
            "children_utime_s": k1.ru_utime - k0.ru_utime,
            "children_stime_s": k1.ru_stime - k0.ru_stime,
            "self_minflt": s1.ru_minflt - s0.ru_minflt,
            "children_minflt": k1.ru_minflt - k0.ru_minflt,
            "maxrss_self_kb": s1.ru_maxrss,
            "maxrss_children_kb": k1.ru_maxrss,
            "pid": os.getpid(),
            "env": {k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
            "extra": {**extra, **info},
        }
        _emit(record)
        _LOG["inv"] = None
        cpu = sum(record[k] for k in record if k.endswith(("_utime_s", "_stime_s")))
        _say(f"end {name}: wall {wall:.1f} s, cpu {cpu:.1f} s")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _package_sha256() -> str:
    """sha256 over the imported pysceptre package's .py files, names included."""
    root = Path(pysceptre.__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(str(f.relative_to(root)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _madvise_hugepage():
    try:
        from numpy._core.multiarray import _get_madvise_hugepage
    except ImportError:  # numpy < 2
        try:
            from numpy.core.multiarray import _get_madvise_hugepage
        except ImportError:
            return None
    return bool(_get_madvise_hugepage())


def _blas_info() -> list[dict]:
    try:
        from threadpoolctl import threadpool_info

        return threadpool_info()
    except Exception as exc:
        return [{"error": repr(exc)}]


def _blas_architecture() -> str:
    return ",".join(
        sorted({str(d.get("architecture")) for d in _blas_info() if "architecture" in d})
    )


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def runtime_info(full: bool = False) -> dict:
    """What a measurement depends on: versions, BLAS, CPU, allocator and job settings.

    `full` adds numpy's enabled CPU features, `numactl -H` and the package hash.
    """
    import numba
    import scipy

    from pysceptre.crt import sampler

    info = {
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "pandas": pd.__version__,
        "pysceptre_file": pysceptre.__file__,
        "pysceptre_version": getattr(pysceptre, "__version__", None),
        "have_numba": bool(getattr(sampler, "_HAVE_NUMBA", False)),
        "glibc": os.confstr("CS_GNU_LIBC_VERSION") if hasattr(os, "confstr") else None,
        "numpy_madvise_hugepage": _madvise_hugepage(),
        "thp_enabled": _read_text("/sys/kernel/mm/transparent_hugepage/enabled"),
        "thp_defrag": _read_text("/sys/kernel/mm/transparent_hugepage/defrag"),
        "blas": _blas_info(),
        "cpus_available": len(os.sched_getaffinity(0)),
        "parallel_backend": discovery.parallel_backend(),
        "env": {k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__)),
    }
    if full:
        try:
            from numpy._core._multiarray_umath import (
                __cpu_baseline__,
                __cpu_dispatch__,
                __cpu_features__,
            )

            info["numpy_cpu_baseline"] = list(__cpu_baseline__)
            info["numpy_cpu_dispatch"] = list(__cpu_dispatch__)
            info["numpy_cpu_enabled"] = [k for k, v in __cpu_features__.items() if v]
        except ImportError as exc:
            info["numpy_cpu_features_error"] = repr(exc)
        try:
            run = subprocess.run(["numactl", "-H"], capture_output=True, text=True, timeout=30)
            info["numactl_H"] = run.stdout.strip() or run.stderr.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            info["numactl_H"] = f"unavailable: {exc!r}"
        info["pysceptre_src_sha256"] = _package_sha256()
    return info


# ---------------------------------------------------------------------------
# Gene-fit cache
# ---------------------------------------------------------------------------

_FIT_FIELDS = (
    ("theta", np.float64),
    ("theta_method", np.int64),
    ("theta_clamped", np.bool_),
    ("glm_converged", np.bool_),
    ("min_eigenvalue", np.float64),
    ("max_eigenvalue", np.float64),
    ("n_covariates", np.int64),
)


class GeneFitCache:
    """Width-1 gene fits on disk, keyed by gene id, valid for one setup only.

    The setup is `fingerprint`: the package source, the CPU model and BLAS
    core, numpy and scipy, the covariate matrix and the fit settings. A cache
    made under another setup is refused rather than reused. Arrays are stored
    with `allow_pickle=False` and read back bit for bit.
    """

    def __init__(self, path: str | Path, fingerprint: dict):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.fits: dict[str, discovery.GenePrecomputation] = {}
        self.last_call: dict = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with np.load(self.path, allow_pickle=False) as z:
            stored = json.loads(str(z["fingerprint"]))
            diff = {
                k: {"cache": stored.get(k), "now": v}
                for k, v in self.fingerprint.items()
                if stored.get(k) != v
            }
            if diff:
                raise SystemExit(f"refusing {self.path}: made under another setup: {diff}")
            ids = z["gene_ids"].tolist()
            coefs = z["coefs"]
            fields = {name: z[name] for name, _ in _FIT_FIELDS}
        for i, gene_id in enumerate(ids):
            self.fits[gene_id] = discovery.GenePrecomputation(
                fitted_coefs=np.array(coefs[i]),
                theta=float(fields["theta"][i]),
                theta_method=int(fields["theta_method"][i]),
                theta_clamped=bool(fields["theta_clamped"][i]),
                glm_converged=bool(fields["glm_converged"][i]),
                min_eigenvalue=float(fields["min_eigenvalue"][i]),
                max_eigenvalue=float(fields["max_eigenvalue"][i]),
                n_covariates=int(fields["n_covariates"][i]),
            )
        _say(f"gene-fit cache {self.path}: {len(self.fits)} genes")

    def save(self) -> None:
        """Write every fit held, atomically."""
        ids = list(self.fits)
        arrays = {
            "gene_ids": np.array(ids, dtype=str),
            "coefs": np.stack(
                [np.asarray(self.fits[g].fitted_coefs, dtype=np.float64) for g in ids]
            ),
            "fingerprint": np.array(json.dumps(self.fingerprint, sort_keys=True)),
        }
        for name, dtype in _FIT_FIELDS:
            arrays[name] = np.array([getattr(self.fits[g], name) for g in ids], dtype=dtype)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, self.path)

    def compare(self, fits: dict) -> dict:
        """How many of `fits` the cache already holds, and how many of those are bit-identical."""
        shared = [g for g in fits if g in self.fits]
        equal = [
            g
            for g in shared
            if np.array_equal(
                np.asarray(fits[g].fitted_coefs).view(np.uint64),
                np.asarray(self.fits[g].fitted_coefs).view(np.uint64),
            )
            and np.float64(fits[g].theta).view(np.uint64)
            == np.float64(self.fits[g].theta).view(np.uint64)
        ]
        return {
            "n_shared": len(shared),
            "n_bit_identical": len(equal),
            "differing": [g for g in shared if g not in set(equal)][:20],
        }

    def wrap(self, original):
        """A drop-in `fit_all_genes` that fits only the genes the cache lacks.

        The missing genes are fit by `original` with the caller's own
        arguments, so with `batch_width=1` each fit is the one the engine
        would have made. Any other batch width bypasses the cache.
        """

        @functools.wraps(original)
        def fit_all_genes(response_matrix, gene_ids, covariate_matrix, **kwargs):
            if kwargs.get("batch_width") != 1:
                self.last_call = {"n_requested": len(gene_ids), "bypassed": True}
                return original(response_matrix, gene_ids, covariate_matrix, **kwargs)
            rows = kwargs.pop("gene_rows", None)
            rows = list(range(len(gene_ids))) if rows is None else list(rows)
            missing = [(g, r) for g, r in zip(gene_ids, rows, strict=True) if g not in self.fits]
            if missing:
                fresh = original(
                    response_matrix,
                    [g for g, _ in missing],
                    covariate_matrix,
                    gene_rows=[r for _, r in missing],
                    **kwargs,
                )
                self.fits.update(fresh)
                self.save()
            self.last_call = {
                "n_requested": len(gene_ids),
                "n_computed": len(missing),
                "n_cached": len(gene_ids) - len(missing),
            }
            return {g: self.fits[g] for g in gene_ids}

        return fit_all_genes


def fit_fingerprint(ex, dataset: str) -> dict:
    """The setup a width-1 gene fit depends on, as stored in and checked against the cache."""
    import scipy

    X = np.ascontiguousarray(ex.covariate_matrix, dtype=np.float64)
    return {
        "pysceptre_src_sha256": _package_sha256(),
        "cpu_model": _cpu_model(),
        "blas_architecture": _blas_architecture(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "dataset": Path(dataset).name,
        "dataset_bytes": Path(dataset).stat().st_size,
        "n_cells": int(X.shape[0]),
        "n_covariates": int(X.shape[1]),
        "covariate_sha256": hashlib.sha256(X.view(np.uint8)).hexdigest(),
        "batch_width": int(discovery._GENE_BATCH_WIDTH),
        "gene_chunk_memory_gb": float(discovery._GENE_CHUNK_MEMORY_GB),
        "theta_bounds": [float(b) for b in discovery._THETA_BOUNDS],
    }


def install_cache(ex, a) -> GeneFitCache:
    """Open the gene-fit cache and put its `fit_all_genes` in the engine's place."""
    cache = GeneFitCache(a.fit_cache, fit_fingerprint(ex, a.dataset))
    if "fit_all_genes" not in _ORIGINAL:
        _ORIGINAL["fit_all_genes"] = discovery.fit_all_genes
        discovery.fit_all_genes = _timed(
            "fit_all_genes",
            cache.wrap(_ORIGINAL["fit_all_genes"]),
            lambda args, kwargs, result: dict(cache.last_call),
        )
    return cache


def fit_genes_like_the_engine(ex, gene_ids: list[str], n_jobs: int, fit_all_genes) -> dict:
    """`fit_all_genes` with exactly the arguments `run_discovery_ntcells_complement` passes."""
    row_of = {g: i for i, g in enumerate(ex.gene_ids)}
    X = ex.covariate_matrix
    xo = discovery.x_outer_flat(X)
    try:
        with discovery._ENGINE_LOCK, one_blas_thread():
            return fit_all_genes(
                ex.response_matrix,
                list(gene_ids),
                X,
                x_outer_flat_shared=xo,
                chunk_memory_gb=discovery._GENE_CHUNK_MEMORY_GB,
                gene_rows=[row_of[g] for g in gene_ids],
                batch_width=discovery._GENE_BATCH_WIDTH,
                n_jobs=n_jobs,
            )
    finally:
        del xo
        gc.collect()


def prefit(ex, cache: GeneFitCache, gene_ids: list[str], n_jobs: int) -> None:
    """Fit, through the cache, the genes of `gene_ids` it lacks, as their own invocation."""
    row_of = {g: i for i, g in enumerate(ex.gene_ids)}
    missing = sorted({g for g in gene_ids if g not in cache.fits}, key=row_of.__getitem__)
    _say(f"gene fits: {len(gene_ids) - len(missing)} cached, {len(missing)} to fit")
    if not missing:
        return
    with invocation("gene_fits", n_jobs, n_genes=len(missing)):
        fit_genes_like_the_engine(ex, missing, n_jobs, discovery.fit_all_genes)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def import_sceptre_io(a):
    """`sceptre_io` from `--sceptre-io-dir`, else from the pysceptre tree, else beside this script."""
    if a.sceptre_io_dir:
        candidates = [Path(a.sceptre_io_dir)]
    else:
        candidates = [
            Path(pysceptre.__file__).resolve().parents[2] / "scripts",
            Path(__file__).resolve().parent,
        ]
    for d in candidates:
        path = d / "sceptre_io.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("sceptre_io", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["sceptre_io"] = module
            spec.loader.exec_module(module)
            _say(f"sceptre_io from {path}")
            return module
    raise SystemExit(f"no sceptre_io.py in {[str(c) for c in candidates]}")


def load_dataset(a):
    """The export, and the metadata the analysis settings are read from."""
    sio = import_sceptre_io(a)
    with invocation("load_export", 1, dataset=str(a.dataset)), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
        ex = sio.load_export(a.dataset)
    _say(ex.describe())
    meta = json.loads(Path(a.metadata).read_text()) if a.metadata else dict(ex.metadata)
    for key in ("side_code", "resampling_approximation"):
        if key in ex.metadata and ex.metadata[key] != meta.get(key):
            _say(
                f"WARNING: {key} is {meta.get(key)!r} in --metadata, {ex.metadata[key]!r} in the export"
            )
    return ex, meta


def read_reference(path: str) -> pd.DataFrame:
    """The reference result, every float read back exactly."""
    ref = pd.read_csv(path, sep="\t", float_precision="round_trip", dtype={k: str for k in _KEYS})
    missing = [c for c in _RESULT_COLUMNS if c not in ref.columns]
    if missing:
        raise SystemExit(f"{path} lacks columns {missing}")
    return ref


class Layout:
    """The reference run's target chunks: `chunks`, `width`, `hash` and `chunk_of`.

    `hash` is the sha256 of the chunks as a compact JSON list of lists of ids,
    the encoding `discovery.chunk_layout_hash` uses where it exists, so the two
    can be compared.
    """

    def __init__(self, chunks: list[list[str]], width: int):
        self.chunks = chunks
        self.width = width
        encoded = json.dumps(
            [[str(t) for t in chunk] for chunk in chunks], ensure_ascii=False, separators=(",", ":")
        )
        self.hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.chunk_of = {t: i for i, chunk in enumerate(chunks) for t in chunk}


def reference_layout(ex, meta: dict, a, ref: pd.DataFrame | None) -> Layout:
    """The reference run's chunks, rebuilt as `run_discovery_ntcells_complement` builds them.

    Refuses a width or chunk count other than `--expect-width` and
    `--expect-chunks`, and a reference result whose first-appearance target
    order differs from the chunk order.
    """
    present = set(ex.pairs["grna_target"])
    ids = [t for t in ex.grna_target_cells if t in present]
    n_cells = ex.covariate_matrix.shape[0]
    side_code = int(meta["side_code"])
    B2, B3 = _resampling_budget(
        str(meta["resampling_approximation"]),
        side_code,
        len(ex.pairs),
        float(meta.get("multiple_testing_alpha", 0.1)),
        "crt",
    )
    B_total = 499 + B2 + B3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        width = discovery._resolve_target_chunk_size(
            n_cells,
            B_total,
            [len(ex.grna_target_cells[t]) for t in ids],
            len(ids),
            a.ref_target_chunk_size,
            a.ref_chunk_memory_gb,
        )
    layout = Layout([ids[i : i + width] for i in range(0, len(ids), width)], width)
    engine_hash = getattr(discovery, "chunk_layout_hash", None)
    if engine_hash is not None and engine_hash(layout.chunks) != layout.hash:
        raise SystemExit("the layout hash differs from discovery.chunk_layout_hash")
    _say(
        f"reference layout: {len(layout.chunks)} chunks of width {width} over {len(ids)} targets, "
        f"{n_cells} cells, B1+B2+B3={B_total}; sha256 {layout.hash}"
        + (f"; engine warned: {caught[0].message}" if caught else "")
    )
    if a.expect_width and width != a.expect_width:
        raise SystemExit(f"reference width is {width}, expected {a.expect_width}")
    if a.expect_chunks and len(layout.chunks) != a.expect_chunks:
        raise SystemExit(f"reference has {len(layout.chunks)} chunks, expected {a.expect_chunks}")
    if ref is not None:
        order = list(pd.unique(ref["grna_target"]))
        if order != ids:
            first = next(
                (i for i, (x, y) in enumerate(zip(order, ids, strict=False)) if x != y), None
            )
            raise SystemExit(
                f"the reference result's target order differs from the chunk order: "
                f"{len(order)} against {len(ids)} targets, first difference at {first}"
            )
        _say("reference result's target order equals the chunk order")
    return layout


def stage_counts(ref: pd.DataFrame, layout: Layout) -> pd.DataFrame:
    """Per reference chunk: pairs, genes, and pairs at stages 2 and 3 in the reference result."""
    chunk = ref["grna_target"].map(layout.chunk_of)
    frame = pd.DataFrame(
        {
            "chunk": chunk.to_numpy(),
            "s2": (ref["stage"] == 2).to_numpy(),
            "s3": (ref["stage"] == 3).to_numpy(),
            "response_id": ref["response_id"].to_numpy(),
        }
    )
    out = frame.groupby("chunk").agg(
        n_pairs=("s2", "size"),
        s2=("s2", "sum"),
        s3=("s3", "sum"),
        n_genes=("response_id", "nunique"),
    )
    return out.reindex(range(len(layout.chunks)), fill_value=0)


def default_chunks(counts: pd.DataFrame, n_total: int = 11, n_spread: int = 9) -> dict[int, str]:
    """Chunk index -> reason, `n_total` chunks at most.

    Evenly spaced chunks, first and last included; then, unless one of those
    already holds a stage-3 reference pair, the chunk with the most stage-3
    pairs (ties: more stage-2 pairs, then the lower index); then the chunks
    with the most stage-2 pairs until `n_total`.
    """
    n = len(counts)
    chosen: dict[int, str] = {}
    for c in np.unique(np.linspace(0, n - 1, min(n_spread, n)).round().astype(int)):
        chosen[int(c)] = "spread"
    ranked = counts.rename_axis("chunk").reset_index()
    if not any(counts.at[c, "s3"] > 0 for c in chosen):
        s3 = ranked.loc[ranked["s3"] > 0].sort_values(
            ["s3", "s2", "chunk"], ascending=[False, False, True], kind="stable"
        )
        if len(s3):
            chosen[int(s3["chunk"].iloc[0])] = "stage 3"
    by_s2 = ranked.sort_values(["s2", "chunk"], ascending=[False, True], kind="stable")
    for c in by_s2["chunk"]:
        if len(chosen) >= min(n_total, n):
            break
        if int(c) not in chosen:
            chosen[int(c)] = "stage 2"
    return dict(sorted(chosen.items()))


def choose_chunks(spec: str, layout: Layout, ref: pd.DataFrame | None) -> list[int]:
    """`--chunks`: comma-separated indices, or `default` (see `default_chunks`); printed with reasons."""
    counts = stage_counts(ref, layout) if ref is not None else None
    if spec == "default":
        if counts is None:
            raise SystemExit("--chunks default needs --tsv")
        reasons = default_chunks(counts)
    else:
        reasons = {int(c): "requested" for c in spec.split(",") if c.strip()}
    bad = [c for c in reasons if not 0 <= c < len(layout.chunks)]
    if bad:
        raise SystemExit(f"chunk indices out of range: {bad}")
    for c, why in reasons.items():
        line = f"chunk {c} ({why}): {len(layout.chunks[c])} targets"
        if counts is not None:
            r = counts.loc[c]
            line += f", {r.n_pairs} pairs, {r.n_genes} genes, stage 2: {r.s2}, stage 3: {r.s3}"
        _say(line)
    return list(reasons)


def random_cis_genes(ex, seed: int) -> list[str]:
    """Every cis gene (a gene in the reference pairs), in a seeded random order."""
    cis = set(ex.pairs["response_id"])
    genes = [g for g in ex.gene_ids if g in cis]
    return [genes[i] for i in np.random.default_rng(seed).permutation(len(genes))]


def gene_chromosomes(path: str) -> dict[str, str]:
    """Ensembl gene id without version -> chromosome, from a GTF's gene records.

    Canonical chromosomes only; GENCODE's `_PAR_Y` copies are dropped so each
    pseudoautosomal gene keeps its chrX record.
    """
    out: dict[str, str] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if "\tgene\t" not in line or line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if len(f) < 9 or f[2] != "gene" or f[0] not in _CANONICAL:
                continue
            gene_id = _GENE_ID.search(f[8]).group(1)
            if gene_id.endswith("_PAR_Y"):
                continue
            out[gene_id.split(".")[0]] = f[0]
    return out


def pair_nonzero_counts(ex, genes, targets, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Treated and control nonzero-cell counts per (gene, target) pair, by `nonzero_counts`.

    Computed over blocks of `block` genes against every target named, so no
    more than a block of the response matrix is copied at a time.
    """
    row_of = {g: i for i, g in enumerate(ex.gene_ids)}
    n_cells = ex.covariate_matrix.shape[0]
    target_list = list(dict.fromkeys(targets))
    col_of = {t: j for j, t in enumerate(target_list)}
    cells = [ex.grna_target_cells[t] for t in target_list]
    g_rows = np.fromiter((row_of[g] for g in genes), dtype=np.int64, count=len(genes))
    t_cols = np.fromiter((col_of[t] for t in targets), dtype=np.int64, count=len(targets))
    trt = np.full(len(genes), -1, dtype=np.int64)
    cntrl = np.full(len(genes), -1, dtype=np.int64)
    unique_rows = np.unique(g_rows)
    for start in range(0, unique_rows.size, block):
        rows = unique_rows[start : start + block]
        t_block, c_block = nonzero_counts(ex.response_matrix[rows], cells, n_cells)
        sel = np.isin(g_rows, rows)
        r = np.searchsorted(rows, g_rows[sel])
        trt[sel] = t_block[r, t_cols[sel]]
        cntrl[sel] = c_block[r, t_cols[sel]]
        _say(f"nonzero counts: genes {start + rows.size} of {unique_rows.size}")
    return trt, cntrl


def analysis_kwargs(meta: dict) -> dict:
    """The reference run's settings: CRT, complement group, its side and approximation."""
    return {
        "side": _SIDES[int(meta["side_code"])],
        "resampling_approximation": str(meta["resampling_approximation"]),
        "multiple_testing_alpha": float(meta.get("multiple_testing_alpha", 0.1)),
        "moi": "high",
        "control_group": "complement",
        "resampling_mechanism": "crt",
    }


def run_engine(ex, meta, pairs, a, *, target_chunk_size: int, name: str = "engine"):
    """`run_discovery_analysis` on `pairs`, as one logged invocation."""
    n_chunks = -(-pairs["grna_target"].nunique() // target_chunk_size)
    with invocation(
        name,
        a.n_jobs,
        n_pairs=len(pairs),
        n_targets=int(pairs["grna_target"].nunique()),
        n_genes=int(pairs["response_id"].nunique()),
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=a.chunk_memory_gb,
        gene_job_backend=discovery.gene_job_backend(n_chunks) or discovery.parallel_backend(),
    ):
        return run_discovery_analysis(
            ex.response_matrix,
            ex.gene_ids,
            ex.covariate_matrix,
            ex.grna_target_cells,
            pairs,
            seed=a.seed_test,
            target_chunk_size=target_chunk_size,
            chunk_memory_gb=a.chunk_memory_gb,
            n_jobs=a.n_jobs,
            **analysis_kwargs(meta),
        )


def compare_to_reference(ours: pd.DataFrame, ref: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Every column of `ours` against `ref`, exactly: equal values with equal sign bits, or both NaN.

    Returns a summary and the mismatching (pair, column) entries.
    """
    ours = ours.reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    same_order = len(ours) == len(ref) and bool(
        (ours[_KEYS].to_numpy() == ref[_KEYS].to_numpy()).all()
    )
    merged = ours[_RESULT_COLUMNS].merge(
        ref[_RESULT_COLUMNS],
        on=_KEYS,
        how="outer",
        suffixes=("_ours", "_ref"),
        indicator=True,
        validate="one_to_one",
    )
    both = merged.loc[merged["_merge"] == "both"].reset_index(drop=True)
    per_column = {}
    rows = []
    for c in _VALUE_COLUMNS:
        x = both[f"{c}_ours"].to_numpy(dtype=float)
        y = both[f"{c}_ref"].to_numpy(dtype=float)
        nan_x, nan_y = np.isnan(x), np.isnan(y)
        equal = (nan_x & nan_y) | ((x == y) & (np.signbit(x) == np.signbit(y)))
        per_column[c] = int((~equal).sum())
        for i in np.flatnonzero(~equal):
            diff = abs(x[i] - y[i])
            rows.append(
                {
                    "response_id": both.at[i, "response_id"],
                    "grna_target": both.at[i, "grna_target"],
                    "column": c,
                    "ours": x[i],
                    "reference": y[i],
                    "abs_diff": diff,
                    "rel_diff": diff / abs(y[i]) if y[i] != 0 else np.nan,
                    "stage_ours": both.at[i, "stage_ours"],
                    "stage_ref": both.at[i, "stage_ref"],
                }
            )
    mismatches = pd.DataFrame(
        rows,
        columns=[
            "response_id",
            "grna_target",
            "column",
            "ours",
            "reference",
            "abs_diff",
            "rel_diff",
            "stage_ours",
            "stage_ref",
        ],
    )
    n_pairs_mismatching = int(mismatches[_KEYS].drop_duplicates().shape[0])
    stages = pd.crosstab(
        both["stage_ours"].astype(float).fillna(-1), both["stage_ref"].astype(float).fillna(-1)
    )
    summary = {
        "n_ours": len(ours),
        "n_reference": len(ref),
        "n_compared": len(both),
        "only_in_ours": int((merged["_merge"] == "left_only").sum()),
        "only_in_reference": int((merged["_merge"] == "right_only").sum()),
        "same_row_order": same_order,
        "mismatches_per_column": per_column,
        "n_pairs_with_any_mismatch": n_pairs_mismatching,
        "stage_crosstab_ours_by_ref": {
            str(i): {str(j): int(stages.at[i, j]) for j in stages.columns} for i in stages.index
        },
    }
    summary["bit_identical"] = (
        n_pairs_mismatching == 0
        and summary["only_in_ours"] == 0
        and summary["only_in_reference"] == 0
        and same_order
    )
    if len(mismatches):
        rel = mismatches.loc[mismatches["column"] == "z_orig", "rel_diff"]
        summary["z_orig_rel_diff_max"] = _finite_or_none(rel.max()) if len(rel) else None
        summary["z_orig_rel_diff_median"] = _finite_or_none(rel.median()) if len(rel) else None
    return summary, mismatches


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _start(a) -> Path:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    install_instrumentation(out)
    a._t0 = time.perf_counter()
    _write_json(out / "step.json", {"status": "running", "argv": sys.argv, "args": _args_dict(a)})
    return out


def _args_dict(a) -> dict:
    return {k: v for k, v in vars(a).items() if not k.startswith("_") and k != "fn"}


def _finish(a, out: Path, results: dict) -> None:
    self_ru = resource.getrusage(resource.RUSAGE_SELF)
    child_ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    _write_json(
        out / "step.json",
        {
            "status": "done",
            "argv": sys.argv,
            "args": _args_dict(a),
            "wall_s": time.perf_counter() - a._t0,
            "maxrss_self_kb": self_ru.ru_maxrss,
            "maxrss_children_kb": child_ru.ru_maxrss,
            "runtime": runtime_info(),
            "results": results,
        },
    )


def cmd_env(a) -> None:
    out = _start(a)
    info = runtime_info(full=True)
    for key in (
        "hostname",
        "cpu_model",
        "numpy",
        "scipy",
        "numba",
        "have_numba",
        "glibc",
        "thp_enabled",
        "thp_defrag",
        "numpy_madvise_hugepage",
        "cpus_available",
        "parallel_backend",
        "pysceptre_file",
        "pysceptre_src_sha256",
    ):
        _say(f"{key}: {info.get(key)}")
    for d in info["blas"]:
        _say(
            f"blas: {d.get('internal_api')} {d.get('version')} {d.get('architecture')} "
            f"threads={d.get('num_threads')} {d.get('filepath')}"
        )
    _say(f"numpy CPU baseline {info.get('numpy_cpu_baseline')}")
    _say(f"numpy CPU features enabled {info.get('numpy_cpu_enabled')}")
    _say(f"numactl -H:\n{info.get('numactl_H')}")
    _write_json(out / "env.json", info)
    _finish(a, out, {"env": "env.json"})


def cmd_qc_crosscheck(a) -> None:
    out = _start(a)
    ex, meta = load_dataset(a)
    ref = read_reference(a.tsv) if a.tsv else None
    reference_layout(ex, meta, a, ref)
    info = ex.discovery_pairs_with_info
    results: dict = {
        "info_columns": None if info is None else {c: str(info[c].dtype) for c in info.columns}
    }
    _say(f"discovery_pairs_with_info columns: {results['info_columns']}")
    table = (info if info is not None else ex.pairs).reset_index(drop=True)
    known = table["response_id"].isin(set(ex.gene_ids)) & table["grna_target"].isin(
        set(ex.grna_target_cells)
    )
    results["n_rows"] = len(table)
    results["n_rows_unknown_gene_or_target"] = int((~known).sum())
    table = table.loc[known].reset_index(drop=True)
    with invocation("nonzero_counts", 1, n_pairs=len(table)):
        trt, cntrl = pair_nonzero_counts(
            ex, table["response_id"].tolist(), table["grna_target"].tolist(), a.gene_block
        )
    thr_trt, thr_cntrl = int(meta["n_nonzero_trt_thresh"]), int(meta["n_nonzero_cntrl_thresh"])
    ours_pass = (trt >= thr_trt) & (cntrl >= thr_cntrl)
    frame = table.loc[:, _KEYS].assign(
        n_nonzero_trt_ours=trt, n_nonzero_cntrl_ours=cntrl, pass_qc_ours=ours_pass
    )
    in_pairs = pd.MultiIndex.from_frame(table[_KEYS]).isin(
        pd.MultiIndex.from_frame(ex.pairs[_KEYS])
    )
    frame["in_cis_pairs"] = in_pairs
    disagreeing = np.zeros(len(frame), dtype=bool)
    for col in ("n_nonzero_trt", "n_nonzero_cntrl", "pass_qc"):
        if info is None or col not in table.columns:
            continue
        if col == "pass_qc":
            theirs = table[col].map(_as_bool).to_numpy(dtype=float)
        else:
            theirs = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        mine = frame[f"{col}_ours"].to_numpy(dtype=float)
        frame[col] = theirs
        agree = mine == theirs
        disagreeing |= ~agree
        results[f"{col}_agree"] = int(agree.sum())
        results[f"{col}_disagree"] = int((~agree).sum())
        results[f"{col}_missing_in_export"] = int(np.isnan(theirs).sum())
        results[f"{col}_agree_on_cis_pairs"] = int(agree[in_pairs].sum())
    results["cis_pairs_found"] = int(in_pairs.sum())
    results["cis_pairs_total"] = len(ex.pairs)
    results["cis_pairs_passing_by_our_counts"] = int(ours_pass[in_pairs].sum())
    results["rows_passing_by_our_counts"] = int(ours_pass.sum())
    results["thresholds"] = [thr_trt, thr_cntrl]
    frame.loc[disagreeing].to_csv(out / "qc_disagreements.tsv", sep="\t", index=False)
    for k, v in results.items():
        if k != "info_columns":
            _say(f"{k}: {v}")
    _write_json(out / "qc_crosscheck.json", results)
    _finish(a, out, results)


def cmd_gene_fits(a) -> None:
    out = _start(a)
    ex, meta = load_dataset(a)
    ref = read_reference(a.tsv) if a.tsv else None
    reference_layout(ex, meta, a, ref)
    genes = random_cis_genes(ex, a.seed)[: a.n_genes]
    cache = GeneFitCache(a.fit_cache, fit_fingerprint(ex, a.dataset))
    with invocation("gene_fits", a.n_jobs, n_genes=len(genes)):
        fits = fit_genes_like_the_engine(ex, genes, a.n_jobs, discovery.fit_all_genes)
    check = cache.compare(fits)
    _say(
        f"against the cache: {check['n_shared']} genes already held, "
        f"{check['n_bit_identical']} bit-identical"
    )
    cache.fits.update({g: fits[g] for g in genes if g not in cache.fits})
    cache.save()
    pd.DataFrame(
        {
            "response_id": genes,
            "theta": [fits[g].theta for g in genes],
            "theta_method": [fits[g].theta_method for g in genes],
            "glm_converged": [fits[g].glm_converged for g in genes],
        }
    ).to_csv(out / "gene_fits.tsv", sep="\t", index=False)
    _finish(a, out, {"n_genes": len(genes), "against_cache": check, "cache_size": len(cache.fits)})


def cmd_cis_slice(a) -> None:
    out = _start(a)
    ex, meta = load_dataset(a)
    ref = read_reference(a.tsv)
    layout = reference_layout(ex, meta, a, ref)
    chunk_ids = choose_chunks(a.chunks, layout, ref)
    targets = {t for c in chunk_ids for t in layout.chunks[c]}
    pairs = ex.pairs.loc[ex.pairs["grna_target"].isin(targets), _KEYS].reset_index(drop=True)
    expected = ref.loc[ref["grna_target"].isin(targets)].reset_index(drop=True)
    _say(f"cis slice: {len(chunk_ids)} chunks, {len(targets)} targets, {len(pairs)} pairs")
    cache = install_cache(ex, a)
    if not a.no_prefit:
        prefit(ex, cache, list(pairs["response_id"].unique()), a.n_jobs)
    result = run_engine(ex, meta, pairs, a, target_chunk_size=layout.width)
    summary, mismatches = compare_to_reference(result, expected)
    result.to_csv(out / "result.tsv.gz", sep="\t", index=False)
    mismatches.to_csv(out / "mismatches.tsv", sep="\t", index=False)
    summary.update(chunks=chunk_ids, layout_hash=layout.hash, width=layout.width)
    _write_json(out / "compare.json", summary)
    _say(
        f"compared {summary['n_compared']} pairs: mismatches per column "
        f"{summary['mismatches_per_column']}, only in ours {summary['only_in_ours']}, only in "
        f"reference {summary['only_in_reference']}, same row order {summary['same_row_order']}"
    )
    _say("BIT-IDENTICAL to the reference" if summary["bit_identical"] else "NOT bit-identical")
    _finish(a, out, summary)
    if a.strict and not summary["bit_identical"]:
        sys.exit(3)


def cmd_trans_slice(a) -> None:
    out = _start(a)
    ex, meta = load_dataset(a)
    ref = read_reference(a.tsv)
    layout = reference_layout(ex, meta, a, ref)
    chunk_ids = choose_chunks(a.chunks, layout, ref)
    targets = [t for c in chunk_ids for t in layout.chunks[c]]
    chrom_of = gene_chromosomes(a.gtf)
    excluded = set(filter(None, a.exclude_gene_chroms.split(",")))
    ordered = random_cis_genes(ex, a.seed)
    unplaced = [g for g in ordered if g not in chrom_of]
    pool = [g for g in ordered if g in chrom_of and chrom_of[g] not in excluded]
    genes = pool[: a.n_genes]
    _say(
        f"{len(ordered)} cis genes, {len(unplaced)} without a canonical position in the GTF, "
        f"{len(pool)} eligible; using {len(genes)}"
    )
    pair_g, pair_t = [], []
    for t in targets:
        m = _TARGET_NAME.match(t)
        if m is None:
            _say(f"WARNING: target {t!r} is not chr:start-end; left out")
            continue
        other = [g for g in genes if chrom_of[g] != m.group(1)]
        pair_g += other
        pair_t += [t] * len(other)
    with invocation("nonzero_counts", 1, n_pairs=len(pair_g)):
        trt, cntrl = pair_nonzero_counts(ex, pair_g, pair_t, max(1, len(genes)))
    thr_trt, thr_cntrl = int(meta["n_nonzero_trt_thresh"]), int(meta["n_nonzero_cntrl_thresh"])
    candidates = pd.DataFrame(
        {
            "response_id": pair_g,
            "grna_target": pair_t,
            "n_nonzero_trt": trt,
            "n_nonzero_cntrl": cntrl,
            "pass_qc": (trt >= thr_trt) & (cntrl >= thr_cntrl),
        }
    )
    candidates.to_csv(out / "pairs.tsv.gz", sep="\t", index=False)
    pairs = candidates.loc[candidates["pass_qc"], _KEYS].reset_index(drop=True)
    _say(
        f"trans slice: {len(targets)} targets x {len(genes)} genes, {len(candidates)} pairs on "
        f"other chromosomes, {len(pairs)} pass QC at {thr_trt}/{thr_cntrl}"
    )
    cache = install_cache(ex, a)
    if not a.no_prefit:
        prefit(ex, cache, list(pairs["response_id"].unique()), a.n_jobs)
    result = run_engine(ex, meta, pairs, a, target_chunk_size=a.k)
    result.to_csv(out / "result.tsv.gz", sep="\t", index=False)
    stages = result["stage"].value_counts(dropna=False).sort_index()
    results = {
        "chunks": chunk_ids,
        "layout_hash": layout.hash,
        "k": a.k,
        "n_genes": len(genes),
        "n_targets": len(targets),
        "n_candidate_pairs": len(candidates),
        "n_pairs_pass_qc": len(pairs),
        "stages": {str(k): int(v) for k, v in stages.items()},
        "n_fold_change_ge_1": int((result["fold_change"] >= 1).sum()),
    }
    _say(f"stages {results['stages']}")
    _finish(a, out, results)


def _bits_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return x.shape == y.shape and bool(np.array_equal(x.view(np.uint64), y.view(np.uint64)))


def cmd_b1_blocks(a) -> None:
    out = _start(a)
    ex, meta = load_dataset(a)
    ref = read_reference(a.tsv)
    layout = reference_layout(ex, meta, a, ref)
    chunk_ids = choose_chunks(a.chunks, layout, ref)[: a.max_chunks]
    sizes = [int(s) for s in a.block_sizes.split(",")]
    B1 = 499
    B2, B3 = _resampling_budget(
        str(meta["resampling_approximation"]),
        int(meta["side_code"]),
        len(ex.pairs),
        float(meta.get("multiple_testing_alpha", 0.1)),
        "crt",
    )
    genes_of = ex.pairs.groupby("grna_target", sort=False)["response_id"].apply(list)
    picked = {
        t: genes_of[t][: a.genes_per_target]
        for c in chunk_ids
        for t in layout.chunks[c][: a.targets_per_chunk]
    }
    cache = install_cache(ex, a)
    prefit(ex, cache, sorted({g for gs in picked.values() for g in gs}), a.n_jobs)
    row_of = {g: i for i, g in enumerate(ex.gene_ids)}
    X = ex.covariate_matrix
    entropy = discovery.resolve_entropy(a.seed_test)
    rows = []
    xo = discovery.x_outer_flat(X)
    for c in chunk_ids:
        ids = layout.chunks[c]
        with invocation("fit_all_targets", 1, chunk=c), discovery._ENGINE_LOCK, one_blas_thread():
            prepared = discovery.fit_all_targets(
                {t: ex.grna_target_cells[t] for t in ids},
                X,
                B1=B1,
                B2=B2,
                B3=B3,
                seed=entropy,
                n_jobs=1,
                x_outer_flat_shared=xo,
            )
        for t in ids[: a.targets_per_chunk]:
            draws = prepared[t].draws.slice(0, B1)
            for g in picked[t]:
                gp = cache.fits[g]
                y = discovery._get_row(ex.response_matrix, row_of[g])
                with one_blas_thread():
                    pieces = discovery.compute_precomputation_pieces(
                        y, X, gp.fitted_coefs, gp.theta
                    )
                    stacked = discovery.stack_pieces(pieces.a, pieces.w, pieces.D)
                    w0 = time.perf_counter()
                    full = score_stat.compute_null_statistics_from_draws(stacked, draws)
                    t_full = time.perf_counter() - w0
                    for size in sizes:
                        w0 = time.perf_counter()
                        blocks = np.concatenate(
                            [
                                score_stat.compute_null_statistics_from_draws(
                                    stacked, draws[i : i + size]
                                )
                                for i in range(0, B1, size)
                            ]
                        )
                        t_blocks = time.perf_counter() - w0
                        raw = np.concatenate(
                            [
                                score_stat.statistics_from_segment_sums(
                                    draws[i : i + size] @ stacked
                                )
                                for i in range(0, B1, size)
                            ]
                        )
                        differ = full.view(np.uint64) != blocks.view(np.uint64)
                        rows.append(
                            {
                                "chunk": c,
                                "grna_target": t,
                                "response_id": g,
                                "block_size": size,
                                "rows": int(draws.shape[0]),
                                "nnz": int(draws.nnz),
                                "bit_identical": _bits_equal(full, blocks),
                                "bit_identical_without_empty_shortcut": _bits_equal(full, raw),
                                "n_rows_differing": int(differ.sum()),
                                "max_abs_diff": _finite_or_none(np.nanmax(np.abs(full - blocks))),
                                "n_nan_full": int(np.isnan(full).sum()),
                                "wall_full_s": t_full,
                                "wall_blocks_s": t_blocks,
                            }
                        )
                    del pieces, stacked
        del prepared
    del xo
    table = pd.DataFrame(rows)
    table.to_csv(out / "b1_blocks.tsv", sep="\t", index=False)
    results = {
        "n_comparisons": len(table),
        "n_bit_identical": int(table["bit_identical"].sum()) if len(table) else 0,
        "n_bit_identical_without_empty_shortcut": int(
            table["bit_identical_without_empty_shortcut"].sum()
        )
        if len(table)
        else 0,
        "block_sizes": sizes,
        "chunks": chunk_ids,
    }
    _say(
        f"B1 in blocks against the full product: {results['n_bit_identical']} of "
        f"{results['n_comparisons']} bit-identical"
    )
    _finish(a, out, results)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "total": float(v.sum()),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.9)),
        "max": float(v.max()),
    }


def _parse_time_v(path: Path) -> dict:
    """The fields of `/usr/bin/time -v` output that a step's cost is read from."""
    wanted = {
        "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed",
        "User time (seconds)": "user_s",
        "System time (seconds)": "system_s",
        "Percent of CPU this job got": "percent_cpu",
        "Maximum resident set size (kbytes)": "maxrss_kb",
        "Minor (reclaiming a frame) page faults": "minor_faults",
        "Major (requiring I/O) page faults": "major_faults",
        "Exit status": "exit_status",
    }
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        key, _, value = line.strip().rpartition(": ")
        if key in wanted:
            out[wanted[key]] = value
    return out


def _read_calls(files: list[Path]) -> tuple[list[dict], int]:
    records, bad = [], 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    return records, bad


def summarize_step(records: list[dict]) -> dict:
    """Per-function totals and means, per-stage pair costs, rebuild costs and efficiency."""
    calls = [r for r in records if r.get("fn") != "invocation"]
    invocations = [r for r in records if r.get("fn") == "invocation"]
    out: dict = {}

    by_fn: dict = {}
    for r in calls:
        by_fn.setdefault((r["fn"], r.get("phase") or "-"), []).append(r)
    out["functions"] = {
        f"{fn} [{phase}]": {
            "calls": len(rs),
            "wall_s": _stats(r["wall_s"] for r in rs),
            "thread_cpu_s": _stats(r["thread_cpu_s"] for r in rs),
            "utime_s": _stats(r.get("utime_s") for r in rs),
            "stime_s": _stats(r.get("stime_s") for r in rs),
            "minflt": _stats(r.get("minflt") for r in rs),
        }
        for (fn, phase), rs in sorted(by_fn.items())
    }

    tests = [r for r in calls if r["fn"] == "run_low_level_test_full"]
    stages: dict = {}
    for r in tests:
        stages.setdefault(str(r["extra"].get("stage")), []).append(r)
    out["pair_cost_by_stage"] = {
        s: {
            "pairs": len(rs),
            "thread_cpu_s": _stats(r["thread_cpu_s"] for r in rs),
            "wall_s": _stats(r["wall_s"] for r in rs),
            "minflt": _stats(r["minflt"] for r in rs),
        }
        for s, rs in sorted(stages.items())
    }
    nulls: dict = {}
    for r in calls:
        if r["fn"] == "compute_null_statistics_from_draws":
            nulls.setdefault(str(r["extra"].get("rows")), []).append(r)
    out["null_statistics_by_rows"] = {
        rows: {"calls": len(rs), "thread_cpu_s": _stats(r["thread_cpu_s"] for r in rs)}
        for rows, rs in sorted(nulls.items())
    }

    pieces = [
        r
        for r in calls
        if r["fn"] == "compute_precomputation_pieces" and r.get("phase") == "gene_test"
    ]
    stacks = [r for r in calls if r["fn"] == "stack_pieces" and r.get("phase") == "gene_test"]
    n_rebuild = len(pieces)
    if n_rebuild:

        def per_rebuild(key):
            return (
                sum(r.get(key, 0) for r in pieces) + sum(r.get(key, 0) for r in stacks)
            ) / n_rebuild

        out["rebuild"] = {
            "rebuilds": n_rebuild,
            "thread_cpu_s": per_rebuild("thread_cpu_s"),
            "utime_s": per_rebuild("utime_s"),
            "stime_s": per_rebuild("stime_s"),
            "wall_s": per_rebuild("wall_s"),
            "minflt": per_rebuild("minflt"),
        }
    gene_jobs = [r for r in calls if r["fn"] == "_gene_job"]
    if gene_jobs:
        out["gene_test_job"] = {
            "jobs": len(gene_jobs),
            "thread_cpu_s": _stats(r["thread_cpu_s"] for r in gene_jobs),
            "pairs_per_job": _stats(r["extra"].get("n_targets") for r in gene_jobs),
        }

    fit_jobs = [r for r in calls if r["fn"] == "_gene_fit_job"]
    if fit_jobs:
        out["gene_fit"] = {
            "genes": len(fit_jobs),
            "thread_cpu_s": _stats(r["thread_cpu_s"] for r in fit_jobs),
            "wall_s": _stats(r["wall_s"] for r in fit_jobs),
            "minflt": _stats(r["minflt"] for r in fit_jobs),
            "per_gene": [
                {
                    "genes": r["extra"].get("genes"),
                    "wall_s": r["wall_s"],
                    "thread_cpu_s": r["thread_cpu_s"],
                    "minflt": r["minflt"],
                }
                for r in fit_jobs
            ],
        }
    target_fits = [r for r in calls if r["fn"] == "fit_binomial_glm_batch"]
    if target_fits:
        k = sum(r["extra"].get("k", 0) for r in target_fits)
        out["target_fit"] = {
            "fits": len(target_fits),
            "targets": k,
            "thread_cpu_s_per_fit": _stats(r["thread_cpu_s"] for r in target_fits),
            "thread_cpu_s_per_target": sum(r["thread_cpu_s"] for r in target_fits) / max(1, k),
        }

    out["invocations"] = []
    for inv in invocations:
        top = [r for r in calls if r.get("inv") == inv["inv"] and r.get("depth") == 0]
        cpu = sum(
            inv.get(k, 0.0)
            for k in ("self_utime_s", "self_stime_s", "children_utime_s", "children_stime_s")
        )
        denom = inv["wall_s"] * max(1, int(inv.get("n_jobs") or 1))
        out["invocations"].append(
            {
                "name": inv["name"],
                "wall_s": inv["wall_s"],
                "n_jobs": inv.get("n_jobs"),
                "backend": inv.get("backend"),
                "process_cpu_s": cpu,
                "efficiency_process_cpu": cpu / denom if denom else None,
                "logged_thread_cpu_s": sum(r["thread_cpu_s"] for r in top),
                "efficiency_logged_thread_cpu": sum(r["thread_cpu_s"] for r in top) / denom
                if denom
                else None,
                "maxrss_gb": max(inv.get("maxrss_self_kb", 0), inv.get("maxrss_children_kb", 0))
                / 1e6,
                "minflt": inv.get("self_minflt", 0) + inv.get("children_minflt", 0),
                "env": inv.get("env"),
                "extra": inv.get("extra"),
            }
        )
    return out


def _print_step(name: str, s: dict) -> None:
    print(f"\n=== {name}")
    step = s.get("step") or {}
    rt = step.get("runtime") or {}
    if rt:
        print(f"  host {rt.get('hostname')}  cpu {rt.get('cpu_model')}  env {rt.get('env')}")
    tv = s.get("time_v") or {}
    if tv:
        print(
            f"  /usr/bin/time: elapsed {tv.get('elapsed')}  user {tv.get('user_s')} s  "
            f"system {tv.get('system_s')} s  maxrss {tv.get('maxrss_kb')} kB  "
            f"minor faults {tv.get('minor_faults')}  exit {tv.get('exit_status')}"
        )
    for inv in s.get("invocations", []):
        e1, e2 = inv["efficiency_process_cpu"], inv["efficiency_logged_thread_cpu"]
        print(
            f"  {inv['name']:<16} wall {inv['wall_s']:9.1f} s  n_jobs {inv['n_jobs']!s:>3}  "
            f"{inv['backend']!s:<6} cpu {inv['process_cpu_s']:10.1f} s  eff {e1 or 0:5.2f} "
            f"(logged {e2 or 0:5.2f})  rss {inv['maxrss_gb']:6.1f} GB"
        )
    if s.get("gene_fit"):
        g = s["gene_fit"]
        print(
            f"  gene fit: {g['genes']} genes, thread CPU per gene mean "
            f"{g['thread_cpu_s'].get('mean', 0):.1f} s median {g['thread_cpu_s'].get('median', 0):.1f} s "
            f"max {g['thread_cpu_s'].get('max', 0):.1f} s, faults {g['minflt'].get('mean', 0):.0f}"
        )
    if s.get("target_fit"):
        t = s["target_fit"]
        print(
            f"  target fit: {t['fits']} fits, {t['targets']} targets, "
            f"{t['thread_cpu_s_per_target']:.2f} s CPU per target"
        )
    if s.get("rebuild"):
        r = s["rebuild"]
        print(
            f"  rebuild: {r['rebuilds']} rebuilds, {r['thread_cpu_s']:.2f} s CPU "
            f"(user {r['utime_s']:.2f}, system {r['stime_s']:.2f}), {r['minflt']:.0f} faults each"
        )
    for stage, v in s.get("pair_cost_by_stage", {}).items():
        print(
            f"  stage {stage}: {v['pairs']} pairs, {v['thread_cpu_s'].get('mean', 0) * 1e3:.1f} ms "
            f"CPU per pair"
        )
    print("  functions (phase): calls, total thread CPU s, mean ms, mean faults")
    for fn, v in s.get("functions", {}).items():
        print(
            f"    {fn:<52} {v['calls']:>8} {v['thread_cpu_s'].get('total', 0):12.1f} "
            f"{v['thread_cpu_s'].get('mean', 0) * 1e3:10.2f} {v['minflt'].get('mean', 0):12.0f}"
        )
    results = step.get("results") or {}
    if results:
        print(f"  results: {json.dumps(results, default=_json_default)[:2000]}")


def cmd_summarize(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.logs).resolve()
    steps: dict[Path, list[Path]] = {}
    for f in sorted(root.rglob("calls_*.jsonl")):
        if out.resolve() in f.resolve().parents:
            continue
        steps.setdefault(f.parent, []).append(f)
    for step_json in root.rglob("step.json"):
        if out.resolve() not in step_json.resolve().parents:
            steps.setdefault(step_json.parent, [])
    summary = {}
    for step_dir in sorted(
        steps, key=lambda d: (d / "step.json").stat().st_mtime if (d / "step.json").exists() else 0
    ):
        records, bad = _read_calls(steps[step_dir])
        s = summarize_step(records)
        s["unreadable_lines"] = bad
        s["time_v"] = _parse_time_v(step_dir / "time_v.txt")
        if (step_dir / "step.json").exists():
            s["step"] = json.loads((step_dir / "step.json").read_text())
        name = str(step_dir.relative_to(root)) if step_dir != root else "."
        summary[name] = s
        _print_step(name, s)
    _write_json(out / "summary.json", summary)
    _say(f"wrote {out / 'summary.json'}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--dataset", required=True, help="the .h5mu dataset")
    data.add_argument("--tsv", help="the reference (cis) result, tab-separated")
    data.add_argument(
        "--metadata", help="metadata.json with side_code and resampling_approximation"
    )
    data.add_argument("--sceptre-io-dir", help="directory holding the sceptre_io.py to load with")
    data.add_argument("--fit-cache", help="gene-fit cache (.npz), shared across runs")
    data.add_argument("--n-jobs", type=int, default=1)
    data.add_argument(
        "--seed-test", type=int, default=0, help="the analysis seed (the reference run's)"
    )
    data.add_argument("--chunk-memory-gb", type=float, default=100.0, help="for the slice runs")
    data.add_argument("--ref-target-chunk-size", type=int, default=200, help="the reference run's")
    data.add_argument("--ref-chunk-memory-gb", type=float, default=1.0, help="the reference run's")
    data.add_argument("--expect-width", type=int, default=6, help="0 to skip the check")
    data.add_argument("--expect-chunks", type=int, default=1746, help="0 to skip the check")

    p = sub.add_parser("env")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_env)

    p = sub.add_parser("qc-crosscheck", parents=[data])
    p.add_argument("--out", required=True)
    p.add_argument("--gene-block", type=int, default=1024)
    p.set_defaults(fn=cmd_qc_crosscheck)

    p = sub.add_parser("gene-fits", parents=[data])
    p.add_argument("--out", required=True)
    p.add_argument("--n-genes", type=int, required=True)
    p.add_argument("--seed", type=int, default=0, help="for the choice of genes")
    p.set_defaults(fn=cmd_gene_fits)

    p = sub.add_parser("cis-slice", parents=[data])
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", default="default")
    p.add_argument("--no-prefit", action="store_true", help="fit missing genes inside the run")
    p.add_argument("--strict", action="store_true", help="exit 3 unless bit-identical")
    p.set_defaults(fn=cmd_cis_slice)

    p = sub.add_parser("trans-slice", parents=[data])
    p.add_argument("--out", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--chunks", default="default")
    p.add_argument("--n-genes", type=int, required=True)
    p.add_argument("--k", type=int, default=6, help="target_chunk_size of the run")
    p.add_argument("--seed", type=int, default=0, help="for the choice of genes")
    p.add_argument("--exclude-gene-chroms", default="chrY")
    p.add_argument("--no-prefit", action="store_true", help="fit missing genes inside the run")
    p.set_defaults(fn=cmd_trans_slice)

    p = sub.add_parser("b1-blocks", parents=[data])
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", default="default")
    p.add_argument("--max-chunks", type=int, default=3)
    p.add_argument("--targets-per-chunk", type=int, default=2)
    p.add_argument("--genes-per-target", type=int, default=3)
    p.add_argument("--block-sizes", default="32")
    p.set_defaults(fn=cmd_b1_blocks)

    p = sub.add_parser("summarize")
    p.add_argument("--logs", required=True, help="a run directory; every calls_*.jsonl under it")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_summarize)

    a = ap.parse_args()
    if a.cmd in ("gene-fits", "cis-slice", "trans-slice", "b1-blocks") and not a.fit_cache:
        ap.error(f"{a.cmd} needs --fit-cache")
    if a.cmd in ("cis-slice", "trans-slice", "b1-blocks") and not a.tsv:
        ap.error(f"{a.cmd} needs --tsv")
    _say(f"{a.cmd} on {socket.gethostname()} ({_cpu_model()}), pysceptre from {pysceptre.__file__}")
    a.fn(a)


if __name__ == "__main__":
    main()
