"""Time the structured IRLS against 0.5.0's outer-product IRLS on one real design.

Fits a random sample of the dataset's tested genes (the Poisson GLM, then the
theta estimate) and its first tested targets (the logistic GLM) both ways, on
one BLAS thread, and records CPU seconds per fit and how far apart the two
routes' results are. The outer route fits the targets as one batch, as a
discovery chunk does; the structured route fits each alone. The machine is
recorded beside the numbers: timings and last bits both depend on it
(docs/design.md, "Bits depend on the machine").

    python scripts/measure_structured_irls.py DATASET OUT.json [--genes 24] [--targets 6] [--seed 0]

DATASET is an .h5mu written by `make_h5mu.py`, or a directory holding one.
Real screen data is never committed, so the path is an argument with no
default. Peak memory is about `n_cells * p^2 * 8` bytes, the outer route's
`x_outer_flat`.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sceptre_io import load_export_compact, load_h5mu  # noqa: E402

import pysceptre  # noqa: E402
from pysceptre.glm import irls  # noqa: E402
from pysceptre.glm.design import structured_design  # noqa: E402
from pysceptre.glm.irls import (  # noqa: E402
    fit_binomial_glm_batch,
    fit_binomial_glm_structured,
    fit_poisson_glm_batch,
    fit_poisson_glm_structured,
    one_blas_thread,
    x_outer_flat,
)
from pysceptre.glm.nb_theta import estimate_theta  # noqa: E402
from pysceptre.pipeline.discovery import _get_row  # noqa: E402

THETA_BOUNDS = (0.01, 1000.0)


def machine() -> dict:
    """CPU model, library versions and the BLAS threadpoolctl reports."""
    model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    return {
        "cpu": model or platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba_version,
        "compiled_sums": bool(irls._HAVE_NUMBA),
        "pysceptre": pysceptre.__version__,
        "blas": [
            {k: v for k, v in lib.items() if k in ("internal_api", "version", "architecture")}
            for lib in threadpool_info()
            if lib.get("user_api") == "blas"
        ],
    }


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.abs(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--genes", type=int, default=24)
    ap.add_argument("--targets", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    path = Path(a.dataset)
    h5 = path if path.is_file() else path / "dataset.h5mu"
    pairs = load_h5mu(h5, backed=True, all_cells=True).pairs
    tested = sorted(set(pairs["response_id"]))
    rng = np.random.default_rng(a.seed)
    picked = [tested[i] for i in sorted(rng.choice(len(tested), a.genes, replace=False))]
    ex = load_export_compact(path, genes=picked)

    X = np.ascontiguousarray(ex.covariate_matrix, dtype=float)
    n, p = X.shape
    design = structured_design(X)
    xo = x_outer_flat(X)
    out = {
        "machine": machine(),
        "n_cells": n,
        "p": p,
        "dense_columns": int(design.dense.size),
        "group_levels": [g.n_levels for g in design.groups],
        "genes": [],
        "targets": [],
    }
    print(
        json.dumps(
            {k: out[k] for k in ("machine", "n_cells", "p", "dense_columns", "group_levels")}
        ),
        flush=True,
    )

    dfr = n - p
    row = {g: i for i, g in enumerate(ex.gene_ids)}
    with one_blas_thread():
        fit_poisson_glm_structured(design, _get_row(ex.response_matrix, 0))  # compile once, untimed
        for g in picked:
            y = _get_row(ex.response_matrix, row[g])
            t0 = time.process_time()
            fs = fit_poisson_glm_structured(design, y)
            t_struct = time.process_time() - t0
            t0 = time.process_time()
            fo = fit_poisson_glm_batch(X, y, X_outer_flat=xo)
            t_outer = time.process_time() - t0
            th_s, m_s = estimate_theta(y, fs.fitted_values, dfr)
            th_o, m_o = estimate_theta(y, fo.fitted_values, dfr)
            rec = {
                "gene": g,
                "mean_count": float(y.mean()),
                "cpu_s_structured": t_struct,
                "cpu_s_outer": t_outer,
                "n_iter": [int(fs.n_iter), int(fo.n_iter)],
                "coef_max_abs_diff": float(np.max(np.abs(fs.coefs - fo.coefs))),
                "mu_max_rel_diff": relative(fs.fitted_values, fo.fitted_values),
                "theta": [float(th_s), float(th_o)],
                "theta_method": [int(m_s), int(m_o)],
                "theta_clamped_rel_diff": abs(
                    float(np.clip(th_s, *THETA_BOUNDS)) - float(np.clip(th_o, *THETA_BOUNDS))
                )
                / float(np.clip(th_o, *THETA_BOUNDS)),
            }
            out["genes"].append(rec)
            print(json.dumps(rec), flush=True)

        named = set(pairs["grna_target"])
        targets = [t for t in ex.grna_target_cells if t in named][: a.targets]
        Y = np.zeros((len(targets), n))
        for k, t in enumerate(targets):
            Y[k, ex.grna_target_cells[t]] = 1.0
        t0 = time.process_time()
        fb = fit_binomial_glm_batch(X, Y, X_outer_flat=xo)
        t_batch = time.process_time() - t0
        for k, t in enumerate(targets):
            t0 = time.process_time()
            fs = fit_binomial_glm_structured(design, Y[k])
            t_struct = time.process_time() - t0
            rec = {
                "target": t,
                "n_treated": int(Y[k].sum()),
                "cpu_s_structured": t_struct,
                "cpu_s_outer_batch_per_target": t_batch / len(targets),
                "n_iter": [int(fs.n_iter), int(fb.n_iter[k])],
                "prob_max_rel_diff": relative(fs.fitted_values, fb.fitted_values[k]),
            }
            out["targets"].append(rec)
            print(json.dumps(rec), flush=True)

    G, T = out["genes"], out["targets"]
    out["summary"] = {
        "gene_cpu_s_median": [
            float(np.median([r["cpu_s_structured"] for r in G])),
            float(np.median([r["cpu_s_outer"] for r in G])),
        ],
        "gene_mu_max_rel_diff": max(r["mu_max_rel_diff"] for r in G),
        "gene_theta_clamped_max_rel_diff": max(r["theta_clamped_rel_diff"] for r in G),
        "gene_n_iter_mismatches": sum(r["n_iter"][0] != r["n_iter"][1] for r in G),
        "gene_theta_method_mismatches": sum(
            r["theta_method"][0] != r["theta_method"][1] for r in G
        ),
        "target_cpu_s_median": [
            float(np.median([r["cpu_s_structured"] for r in T])),
            T[0]["cpu_s_outer_batch_per_target"] if T else float("nan"),
        ],
        "target_prob_max_rel_diff": max((r["prob_max_rel_diff"] for r in T), default=float("nan")),
        "target_n_iter_mismatches": sum(r["n_iter"][0] != r["n_iter"][1] for r in T),
    }
    print(json.dumps(out["summary"], indent=1), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
