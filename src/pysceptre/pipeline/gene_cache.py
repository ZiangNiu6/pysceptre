"""Width-1 gene fits, computed once and reused by later runs.

`fit_gene_cache` fits genes with the call the complement engine makes, one
gene per BLAS call against the shared covariate outer products, so each fit is
the engine's fit, bit for bit in the same numeric environment.
`save_gene_cache` writes the fits with a fingerprint of what they depend on to
an `.npz` that loads without pickle, and `load_gene_cache` reads one or
several such files, refusing any whose fingerprint does not match the data,
the engine's settings or the current numeric environment. See docs/design.md,
"The trans background".
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..glm.design import validate_design_matrix
from ..glm.irls import x_outer_flat
from . import discovery as _discovery
from .discovery import GenePrecomputation, _get_row, _one_analysis_at_a_time
from .fingerprint import _numeric_components, environment_fingerprint, numeric_environment_key

__all__ = ["GeneFitCache", "fit_gene_cache", "load_gene_cache", "save_gene_cache"]

_FORMAT = "pysceptre.gene_fit_cache.v1"
_ARRAYS = (
    "gene_ids",
    "coefs",
    "theta",
    "theta_method",
    "theta_clamped",
    "glm_converged",
    "min_eigenvalue",
    "max_eigenvalue",
    "n_covariates",
    "fingerprint",
)
# Fingerprint entries that every file of one cache must share.
_DATA_KEYS = (
    "format",
    "n_cells",
    "n_covariates",
    "dfr",
    "covariate_sha256",
    "theta_bounds",
    "batch_width",
    "glm_solver",
)


@dataclass
class GeneFitCache:
    """Gene fits and the fingerprint they were made under.

    `fits` maps a gene id to its `GenePrecomputation`, as `fit_all_genes`
    returns it. `fingerprint` holds `format`; `n_cells`, `n_covariates` and
    `dfr` of the fit; `covariate_sha256`, of the covariate matrix as float64
    C-order bytes; `theta_bounds` and `batch_width`, the engine settings the
    fits were made with; `count_digests`, per gene the `nnz`, `sum` and `sha1`
    of its nonzero counts; and `environment`, an `environment_fingerprint()`.
    """

    fits: dict[str, GenePrecomputation]
    fingerprint: dict


def _covariate_sha256(covariate_matrix: np.ndarray) -> str:
    """sha256 hex digest of the matrix as little-endian float64, C order."""
    X = np.ascontiguousarray(covariate_matrix, dtype="<f8")
    return hashlib.sha256(X).hexdigest()


def _row_nonzeros(response_matrix, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions (ascending int64) and values (float64) of one gene row's nonzero entries.

    The same for a dense array, any sparse format or a backed reader holding
    the same counts.
    """
    if sparse.issparse(response_matrix) and response_matrix.format == "csr":
        lo, hi = response_matrix.indptr[row], response_matrix.indptr[row + 1]
        idx = np.asarray(response_matrix.indices[lo:hi], dtype=np.int64)
        val = np.asarray(response_matrix.data[lo:hi], dtype=np.float64)
        # Stored entries are already the nonzeros when sorted, unique and nonzero.
        if np.all(idx[1:] > idx[:-1]) and np.all(val != 0):
            return idx, val
    y = _get_row(response_matrix, row)
    idx = np.flatnonzero(y)
    return idx.astype(np.int64), y[idx]


def _count_digest(response_matrix, row: int) -> dict:
    """`nnz`, `sum` and `sha1` of one gene row's nonzero counts (positions, then values)."""
    idx, val = _row_nonzeros(response_matrix, row)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(idx, dtype="<i8"))
    h.update(np.ascontiguousarray(val, dtype="<f8"))
    return {"nnz": int(idx.size), "sum": float(val.sum()), "sha1": h.hexdigest()}


def _same_digest(a: dict, b: dict) -> bool:
    same_sum = a["sum"] == b["sum"] or (np.isnan(a["sum"]) and np.isnan(b["sum"]))
    return a["nnz"] == b["nnz"] and a["sha1"] == b["sha1"] and bool(same_sum)


def _engine_settings() -> dict:
    """The engine settings a cached fit must have been made with."""
    return {
        "theta_bounds": [float(b) for b in _discovery._THETA_BOUNDS],
        "batch_width": int(_discovery._GENE_BATCH_WIDTH),
    }


@_one_analysis_at_a_time
def fit_gene_cache(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    genes=None,
    *,
    n_jobs: int = 1,
    glm_solver: str = "outer",
) -> GeneFitCache:
    """Fit genes exactly as the complement engine fits them, with a fingerprint to check later.

    The covariate matrix is validated and converted as `run_discovery_analysis`
    does, and the fits are the engine's own `fit_all_genes` call: one gene per
    BLAS call, the shared `x_outer_flat`, the engine's gene memory budget. A
    gene's fit depends only on its counts and the covariates, so fitting any
    subset gives each gene the fit a run over all of them gives it.

    Args:
        response_matrix: (n_genes, n_cells) counts, dense or sparse, as for
            `run_discovery_analysis`.
        gene_ids: the row labels of `response_matrix`, in order.
        covariate_matrix: (n_cells, p) design matrix.
        genes: the genes to fit; None fits every row. Fitted in `gene_ids` order.
        n_jobs: workers for the fits; cannot change a fit.
        glm_solver: the engine's `glm_solver`, recorded in the fingerprint;
            fits of one solver are never loaded for the other.

    Returns:
        A `GeneFitCache` with one fit per gene, keyed by its id.

    Raises:
        ValueError: a covariate matrix `run_discovery_analysis` refuses,
            `gene_ids` not labelling every row, or `genes` naming an id that
            `gene_ids` lacks or holds twice.
    """
    validate_design_matrix(covariate_matrix, n_cells=response_matrix.shape[1])
    covariate_matrix = np.asarray(covariate_matrix, dtype=float)
    gene_ids = list(gene_ids)
    if len(gene_ids) != response_matrix.shape[0]:
        raise ValueError(
            f"gene_ids has {len(gene_ids)} entries for {response_matrix.shape[0]} rows of "
            "response_matrix"
        )
    if genes is None:
        rows = list(range(len(gene_ids)))
    else:
        wanted = set(genes)
        known = set(gene_ids)
        unknown = [g for g in dict.fromkeys(genes) if g not in known]
        if unknown:
            raise ValueError(
                f"genes names {len(unknown)} id(s) not in gene_ids, e.g. {unknown[:5]}"
            )
        rows = [i for i, g in enumerate(gene_ids) if g in wanted]
    ids = [gene_ids[i] for i in rows]
    repeated = [g for g, n in Counter(ids).items() if n > 1]
    if repeated:
        raise ValueError(
            f"{len(repeated)} gene id(s) label more than one row of response_matrix, "
            f"e.g. {repeated[:5]}"
        )

    design = _discovery._design_for(glm_solver, covariate_matrix)
    fits = _discovery.fit_all_genes(
        response_matrix,
        ids,
        covariate_matrix,
        x_outer_flat_shared=None if design is not None else x_outer_flat(covariate_matrix),
        design=design,
        chunk_memory_gb=_discovery._GENE_CHUNK_MEMORY_GB,
        gene_rows=rows,
        batch_width=_discovery._GENE_BATCH_WIDTH,
        n_jobs=n_jobs,
    )
    n_cells, p = covariate_matrix.shape
    fingerprint = {
        "format": _FORMAT,
        "n_cells": int(n_cells),
        "n_covariates": int(p),
        "dfr": int(n_cells - p),
        "covariate_sha256": _covariate_sha256(covariate_matrix),
        **_engine_settings(),
        "glm_solver": glm_solver,
        "count_digests": {
            str(g): _count_digest(response_matrix, r) for g, r in zip(ids, rows, strict=True)
        },
        "environment": environment_fingerprint(),
    }
    return GeneFitCache(fits={g: fits[g] for g in ids}, fingerprint=fingerprint)


def save_gene_cache(cache: GeneFitCache, path) -> None:
    """Write `cache` to `path` as an `.npz` that `np.load(..., allow_pickle=False)` reads.

    Arrays: `gene_ids` (unicode), `coefs` (G, p) float64, `theta`,
    `min_eigenvalue` and `max_eigenvalue` float64, `theta_method` and
    `n_covariates` int64, `theta_clamped` and `glm_converged` bool, and
    `fingerprint`, the fingerprint as a JSON string. Floats are stored as
    their float64 bits. The file is written to `path` exactly, without a
    suffix added.

    Raises:
        ValueError: a gene id that is not a string, or coefficients whose
            length is not the fingerprint's `n_covariates`.
    """
    ids = list(cache.fits)
    not_str = [g for g in ids if not isinstance(g, str)]
    if not_str:
        raise ValueError(f"gene ids must be strings to be saved, got e.g. {not_str[:5]}")
    p = int(cache.fingerprint["n_covariates"])
    fits = [cache.fits[g] for g in ids]
    coefs = np.empty((len(ids), p), dtype=np.float64)
    for i, (g, fit) in enumerate(zip(ids, fits, strict=True)):
        c = np.asarray(fit.fitted_coefs)
        if c.shape != (p,):
            raise ValueError(f"gene {g!r} has coefficients of shape {c.shape}, expected ({p},)")
        coefs[i] = c
    arrays = {
        "gene_ids": np.array(ids, dtype=np.str_) if ids else np.zeros(0, dtype="<U1"),
        "coefs": coefs,
        "theta": np.array([f.theta for f in fits], dtype=np.float64),
        "theta_method": np.array([f.theta_method for f in fits], dtype=np.int64),
        "theta_clamped": np.array([f.theta_clamped for f in fits], dtype=bool),
        "glm_converged": np.array([f.glm_converged for f in fits], dtype=bool),
        "min_eigenvalue": np.array([f.min_eigenvalue for f in fits], dtype=np.float64),
        "max_eigenvalue": np.array([f.max_eigenvalue for f in fits], dtype=np.float64),
        "n_covariates": np.array([f.n_covariates for f in fits], dtype=np.int64),
        "fingerprint": np.array(json.dumps(cache.fingerprint, sort_keys=True)),
    }
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def _read(path) -> tuple[dict[str, GenePrecomputation], dict]:
    """One file's fits and fingerprint, refusing anything `save_gene_cache` did not write."""
    not_cache = f"{path} is not a gene-fit cache written by save_gene_cache"
    try:
        z = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as e:
        raise ValueError(f"{not_cache}: {e}") from e
    if not hasattr(z, "files"):
        raise ValueError(f"{not_cache}: it holds one array, not an .npz archive")
    with z:
        missing = [k for k in _ARRAYS if k not in z.files]
        if missing:
            raise ValueError(f"{not_cache}: it lacks the array(s) {missing}")
        try:
            a = {k: z[k] for k in _ARRAYS}
        except ValueError as e:
            raise ValueError(f"{not_cache}: {e}") from e
    try:
        fp = json.loads(str(a["fingerprint"][()]))
    except (ValueError, TypeError) as e:
        raise ValueError(f"{not_cache}: its fingerprint is not JSON ({e})") from e
    if not isinstance(fp, dict) or fp.get("format") != _FORMAT:
        got = fp.get("format") if isinstance(fp, dict) else None
        raise ValueError(f"{not_cache}: format {got!r}, expected {_FORMAT!r}")
    # Caches written before the structured solver existed hold outer-product fits.
    fp.setdefault("glm_solver", "outer")
    missing = [k for k in (*_DATA_KEYS, "count_digests", "environment") if k not in fp]
    if missing:
        raise ValueError(f"{not_cache}: its fingerprint lacks {missing}")

    ids = [str(g) for g in a["gene_ids"].tolist()]
    n, p = len(ids), int(fp["n_covariates"])
    shapes = {k: a[k].shape for k in _ARRAYS if k != "fingerprint"}
    expected = {k: (n,) for k in shapes} | {"coefs": (n, p)}
    if shapes != expected:
        bad = {k: shapes[k] for k in shapes if shapes[k] != expected[k]}
        raise ValueError(f"{path}: array shapes {bad} do not match {n} genes and {p} covariates")
    repeated = [g for g, c in Counter(ids).items() if c > 1]
    if repeated:
        raise ValueError(f"{path} holds {len(repeated)} gene(s) twice, e.g. {repeated[:5]}")
    if set(fp["count_digests"]) != set(ids):
        raise ValueError(f"{path}: its count digests do not name the same genes as its fits")
    if fp["dfr"] != fp["n_cells"] - p:
        raise ValueError(
            f"{path}: dfr {fp['dfr']} is not n_cells - n_covariates = {fp['n_cells'] - p}"
        )

    fits = {
        g: GenePrecomputation(
            fitted_coefs=a["coefs"][i].copy(),
            theta=float(a["theta"][i]),
            theta_method=int(a["theta_method"][i]),
            theta_clamped=bool(a["theta_clamped"][i]),
            glm_converged=bool(a["glm_converged"][i]),
            min_eigenvalue=float(a["min_eigenvalue"][i]),
            max_eigenvalue=float(a["max_eigenvalue"][i]),
            n_covariates=int(a["n_covariates"][i]),
        )
        for i, g in enumerate(ids)
    }
    return fits, fp


def _environment_differences(saved: dict, current: dict) -> str:
    a, b = _numeric_components(saved), _numeric_components(current)
    parts = [f"{k} {a[k]!r} in the cache, {b[k]!r} here" for k in a if a[k] != b[k]]
    return "; ".join(parts) if parts else "the numeric environment keys differ"


def load_gene_cache(
    paths,
    *,
    response_matrix=None,
    gene_ids=None,
    covariate_matrix=None,
    check_environment: bool = True,
    glm_solver: str | None = None,
) -> GeneFitCache:
    """Read gene fits written by `save_gene_cache`, refusing any that cannot be reused.

    Several files are merged into one cache; they must hold disjoint genes fit
    on the same data with the same settings. Every file must carry the
    engine's current gene batch width (1) and theta bounds. Given the data,
    the covariate matrix must hash to the fingerprint's and every cached
    gene's counts must match its digest. With `check_environment`, every
    file's `numeric_environment_key` must equal this process's. See
    docs/design.md, "The trans background".

    Args:
        paths: one path, or a sequence of paths.
        response_matrix, gene_ids: the data the fits will be used with; checked
            together, so give both or neither.
        covariate_matrix: the covariate matrix the fits will be used with.
        check_environment: refuse fits made in another numeric environment.
        glm_solver: refuse fits made with another `glm_solver`; None accepts
            either, provided all files agree.

    Returns:
        A `GeneFitCache` of every file's fits, in file order. Its fingerprint
        is the files' shared one, with the count digests merged and the
        environment of the first file. Floats are bitwise those saved.

    Raises:
        ValueError: no path; a file `save_gene_cache` did not write; another
            batch width or theta bounds; files fit on different data or
            settings, or sharing a gene; a covariate matrix, cell count, gene
            or counts other than the fits were made with; or another numeric
            environment. The message names what differs.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("no gene-fit cache files given")
    if (response_matrix is None) != (gene_ids is None):
        raise ValueError("response_matrix and gene_ids are checked together: give both or neither")
    loaded = [_read(path) for path in paths]

    settings = _engine_settings()
    for path, (_, fp) in zip(paths, loaded, strict=True):
        if glm_solver is not None and fp["glm_solver"] != glm_solver:
            raise ValueError(
                f"{path} holds fits of glm_solver={fp['glm_solver']!r}, not {glm_solver!r}"
            )
        if fp["batch_width"] != settings["batch_width"]:
            raise ValueError(
                f"{path} holds fits of batch width {fp['batch_width']} (genes per BLAS call), "
                f"not {settings['batch_width']}: a batched fit depends on its batch-mates, so it "
                "is not the engine's fit"
            )
        if fp["theta_bounds"] != settings["theta_bounds"]:
            raise ValueError(
                f"{path} was fit with theta bounds {fp['theta_bounds']}, but the engine clamps "
                f"theta to {settings['theta_bounds']}"
            )
    first_path, (_, first) = paths[0], loaded[0]
    for path, (_, fp) in zip(paths[1:], loaded[1:], strict=True):
        differ = [k for k in _DATA_KEYS if fp[k] != first[k]]
        if differ:
            detail = "; ".join(f"{k} {fp[k]!r} against {first[k]!r}" for k in differ)
            raise ValueError(
                f"{path} and {first_path} were fit on different data or settings: {detail}"
            )

    fits: dict[str, GenePrecomputation] = {}
    digests: dict[str, dict] = {}
    source: dict[str, object] = {}
    for path, (file_fits, fp) in zip(paths, loaded, strict=True):
        shared = [g for g in file_fits if g in source]
        if shared:
            raise ValueError(
                f"{len(shared)} gene(s) appear in more than one file, e.g. {shared[0]!r} in "
                f"{source[shared[0]]} and {path}"
            )
        for g, fit in file_fits.items():
            fits[g] = fit
            digests[g] = fp["count_digests"][g]
            source[g] = path

    if check_environment:
        current = environment_fingerprint()
        key = numeric_environment_key(current)
        for path, (_, fp) in zip(paths, loaded, strict=True):
            if numeric_environment_key(fp["environment"]) != key:
                raise ValueError(
                    f"{path} was fit in another numeric environment "
                    f"({_environment_differences(fp['environment'], current)}), so its fits "
                    "need not equal this process's bit for bit; refit them here, or pass "
                    "check_environment=False"
                )

    n_cells, p = first["n_cells"], first["n_covariates"]
    if covariate_matrix is not None:
        X = np.asarray(covariate_matrix, dtype=float)
        if X.shape != (n_cells, p):
            raise ValueError(
                f"covariate_matrix has shape {X.shape}; the gene fits were made with "
                f"({n_cells}, {p})"
            )
        digest = _covariate_sha256(X)
        if digest != first["covariate_sha256"]:
            raise ValueError(
                f"covariate_matrix differs from the one the gene fits were made with: sha256 "
                f"{digest} of its float64 bytes, expected {first['covariate_sha256']}"
            )
    if response_matrix is not None:
        if response_matrix.shape[1] != n_cells:
            raise ValueError(
                f"response_matrix has {response_matrix.shape[1]} cells; the gene fits were "
                f"made with {n_cells}"
            )
        row_of = {g: i for i, g in enumerate(gene_ids)}
        absent = [g for g in fits if g not in row_of]
        if absent:
            raise ValueError(f"{len(absent)} cached gene(s) are not in gene_ids, e.g. {absent[:5]}")
        changed = [
            g
            for g in fits
            if not _same_digest(_count_digest(response_matrix, row_of[g]), digests[g])
        ]
        if changed:
            raise ValueError(
                f"the counts of {len(changed)} cached gene(s) differ from the counts they "
                f"were fit on, e.g. {changed[:5]}"
            )

    fingerprint = {
        **{k: first[k] for k in _DATA_KEYS},
        "count_digests": digests,
        "environment": first["environment"],
    }
    return GeneFitCache(fits=fits, fingerprint=fingerprint)
