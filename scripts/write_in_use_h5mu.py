"""Write a dataset with only its cells in use, using MuData's own subsetting.

A dataset exported with `--all-cells` keeps the cells QC removed, with an
`in_use` column in `obs`; pysceptre's backed reader refuses such a file,
because its counts are numbered over every cell. This writes the standard
file instead: every modality subset to the cells in use by AnnData's views,
`obs` subset by the same mask, `uns` copied with the cell count, the nonzero
count and the `all_cells` flag updated. Genes, counts and the cells' order
are unchanged. Extra keyword arguments go to `h5py.Group.create_dataset`
(compression, shuffle), so the file stays a plain `.h5mu`. The input is
never modified.

    python scripts/write_in_use_h5mu.py SRC.h5mu DST.h5mu [--gzip LEVEL | --lzf] [--no-shuffle]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def write_in_use(src: str | Path, dst: str | Path, **dataset_kwargs) -> Path:
    """Write `src` with only its cells in use to `dst`; return `dst`.

    Raises:
        FileExistsError: `dst` exists.
        ValueError: `src` already holds only cells in use, or its modalities'
            cells are not `obs`'s cells in `obs`'s order.
    """
    import h5py
    import mudata
    from anndata.io import write_elem

    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    md = mudata.read_h5mu(src, backed="r")
    mask = md.obs["in_use"].to_numpy(dtype=bool) if "in_use" in md.obs else None
    if mask is None or mask.all():
        raise ValueError(f"{src} already holds only cells in use")
    mods = {}
    for name, adata in md.mod.items():
        if not (adata.obs_names == md.obs_names).all():
            raise ValueError(f"{src}: modality {name!r} does not list obs's cells in obs's order")
        mods[name] = adata[mask].to_memory()
    new = mudata.MuData(mods)
    obs = md.obs.loc[mask].copy()
    if not (obs.index == new.obs_names).all():
        raise ValueError(f"{src}: the subset modalities' cells differ from the subset obs")
    new.obs = obs
    uns = dict(md.uns)
    meta = dict(uns["pysceptre"])
    meta.update(
        n_cells=int(mask.sum()),
        n_nonzero=int(mods["rna"].X.nnz),
        all_cells=False,
    )
    uns["pysceptre"] = meta
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
