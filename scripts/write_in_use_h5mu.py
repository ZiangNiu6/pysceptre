"""Write a dataset with only its cells in use, using MuData's own subsetting.

A dataset exported with `--all-cells` keeps the cells QC removed, with an
`in_use` column in `obs`; pysceptre's backed reader refuses such a file,
because its counts are numbered over every cell. This writes the standard
file instead: every modality (the gRNA assignments of every unit kind and the
raw gRNA counts included) subset to the cells in use by AnnData's views, the
global `obs` subset by the same mask, the global `var` copied as it is, and
`uns` copied with the cell count, the nonzero count and the `all_cells` flag
updated and the NT-cell pool `all_nt_idxs` renumbered among the cells in use,
in its own order. The result reads as `load_export` reads the source.

These are refused rather than copied, before any modality is read into
memory: a cell of the NT-cell pool that is not in use, a `uns` key other than
those `sceptre_io.write_h5mu` writes, a modality's own `uns`, and a top-level
`obsm`, `obsp`, `varm` or `varp` entry that MuData did not make.

Genes, counts and the cells' order are unchanged. Extra keyword arguments go
to `h5py.Group.create_dataset` (compression, shuffle), so the file stays a
plain `.h5mu`. The input is never modified.

    python scripts/write_in_use_h5mu.py SRC.h5mu DST.h5mu [--gzip LEVEL | --lzf] [--no-shuffle]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# The tables `sceptre_io.write_h5mu` stores in `uns`. They are keyed by pair or
# by gRNA, never by cell, so they are copied as they are.
_FRAMES = (
    "grna_target_data_frame",
    "discovery_pairs_with_info",
    "negative_control_pairs",
    "calibration_result",
    "positive_control_pairs",
    "power_result",
    "discovery_result",
)
_UNS_WITHOUT_CELLS = frozenset(
    {"pysceptre", "pairs", *_FRAMES, *(f"{key}_bool_columns" for key in _FRAMES)}
)


def _unhandled(md) -> list[str]:
    """What `md` holds that subsetting its cells would leave in the old numbering, or drop."""
    out = [f"uns[{k!r}]" for k in md.uns if k not in _UNS_WITHOUT_CELLS | {"all_nt_idxs"}]
    out += [f"mod[{name!r}].uns[{k!r}]" for name, adata in md.mod.items() for k in adata.uns]
    # MuData keeps one entry per modality in obsm and varm, and rebuilds them itself.
    for attr in ("obsm", "varm"):
        out += [f"{attr}[{k!r}]" for k in getattr(md, attr) if k not in md.mod]
    for attr in ("obsp", "varp"):
        out += [f"{attr}[{k!r}]" for k in getattr(md, attr)]
    return out


def _pool_in_use(pool, mask: np.ndarray, src: Path) -> np.ndarray:
    """The NT-cell pool `pool` as positions among the cells in `mask`, in the pool's order."""
    pool = np.asarray(pool, dtype=np.int64)
    if pool.size and (pool.min() < 0 or pool.max() >= mask.size):
        raise ValueError(f"{src}: all_nt_idxs has cell indices outside [0, {mask.size})")
    out = pool[~mask[pool]]
    if out.size:
        raise ValueError(
            f"{src}: {out.size} cell(s) of the NT-cell pool all_nt_idxs are not in use, "
            f"e.g. {out[:5].tolist()}"
        )
    return (np.cumsum(mask) - 1)[pool].astype(np.int64)


def write_in_use(src: str | Path, dst: str | Path, **dataset_kwargs) -> Path:
    """Write `src` with only its cells in use to `dst`; return `dst`.

    Raises:
        FileExistsError: `dst` exists.
        ValueError: `src` already holds only cells in use, its modalities'
            cells are not `obs`'s cells in `obs`'s order, a cell of its NT-cell
            pool is not in use, or it holds something this cannot renumber
            (see the module docstring).
    """
    import h5py
    import mudata

    try:
        from anndata.io import write_elem
    except ImportError:  # anndata before 0.11
        from anndata.experimental import write_elem

    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    md = mudata.read_h5mu(src, backed="r")
    mask = md.obs["in_use"].to_numpy(dtype=bool) if "in_use" in md.obs else None
    if mask is None or mask.all():
        raise ValueError(f"{src} already holds only cells in use")
    unhandled = _unhandled(md)
    if unhandled:
        raise ValueError(
            f"{src} holds {', '.join(unhandled)}, which this script cannot renumber to the "
            "cells in use"
        )
    pool = md.uns.get("all_nt_idxs")
    pool = None if pool is None else _pool_in_use(pool, mask, src)
    for name, adata in md.mod.items():
        if not (adata.obs_names == md.obs_names).all():
            raise ValueError(f"{src}: modality {name!r} does not list obs's cells in obs's order")
    mods = {name: adata[mask].to_memory() for name, adata in md.mod.items()}
    new = mudata.MuData(mods)
    obs = md.obs.loc[mask].copy()
    if not (obs.index == new.obs_names).all():
        raise ValueError(f"{src}: the subset modalities' cells differ from the subset obs")
    new.obs = obs
    if not new.var_names.equals(md.var_names):
        raise ValueError(f"{src}: the subset modalities' features differ from var's")
    new.var = md.var.copy()
    uns = dict(md.uns)
    meta = dict(uns["pysceptre"])
    meta.update(
        n_cells=int(mask.sum()),
        n_nonzero=int(mods["rna"].X.nnz),
        all_cells=False,
    )
    uns["pysceptre"] = meta
    if pool is not None:
        uns["all_nt_idxs"] = pool
    tmp = dst.with_name(f"{dst.name}.tmp")
    # Filters apply to arrays only, and `uns` holds scalars: write it after, unfiltered,
    # with AnnData's own element writer.
    new.write(tmp, **dataset_kwargs)
    with h5py.File(tmp, "a") as f:
        if "uns" in f:
            del f["uns"]
        write_elem(f, "uns", uns)
    tmp.replace(dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    comp = ap.add_mutually_exclusive_group()
    comp.add_argument("--gzip", type=int, metavar="LEVEL", help="gzip compression at LEVEL")
    comp.add_argument("--lzf", action="store_true", help="lzf compression")
    ap.add_argument("--no-shuffle", action="store_true", help="no byte-shuffle filter")
    a = ap.parse_args()
    kwargs = {}
    if a.gzip is not None:
        kwargs = dict(compression="gzip", compression_opts=a.gzip)
    elif a.lzf:
        kwargs = dict(compression="lzf")
    if kwargs and not a.no_shuffle:
        kwargs["shuffle"] = True
    t0 = time.time()
    out = write_in_use(a.src, a.dst, **kwargs)
    print(f"wrote {out} in {time.time() - t0:.0f}s with {kwargs or 'no compression'}")


if __name__ == "__main__":
    main()
