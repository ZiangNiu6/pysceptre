"""Build the inputs of a trans background run: pairs, positions, chunk layout, shards.

The background of `run_specificity_check` is every element of a screen's cis
analysis against every cis gene on another chromosome. This script enumerates
those pairs from a dataset written by `make_h5mu.py`, counts their treated and
control nonzero cells with `pysceptre.pipeline.pairwise_qc.nonzero_counts` (the
QC the calibration and power checks use), records element and gene positions,
and fixes the reference run's target fit chunks, which every shard must reuse.

  build_trans_pairs.py pairs --dataset dataset.h5mu --gtf genes.gtf.gz --out DIR
  build_trans_pairs.py shards --out DIR --n-shards S --seed N

`pairs` writes, under DIR: `elements.parquet` (target code, name, chromosome,
start, end, centre, fit chunk, treated cells), `genes.parquet` (gene code, id,
chromosome, TSS, row), `measured_genes.parquet` (every dataset gene the
annotation places), `trans_pairs.parquet` (gene code, target code, nonzero
counts, pass_qc) and `layout.json`. Element positions come from target names of
the form `chr:start-end`; gene positions from the GTF's gene records, keyed by
Ensembl id without version, TSS on the gene's strand. `shards` splits the
reference chunks, in a seeded random order, into S shards of about equal
passing-pair counts and writes `shard_manifest.json`.

Needs the `io` extra (pyarrow, mudata). Run it in a batch job: it loads the
whole dataset.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from sceptre_io import load_export  # noqa: E402

from pysceptre.pipeline.discovery import chunk_layout_hash  # noqa: E402
from pysceptre.pipeline.pairwise_qc import nonzero_counts  # noqa: E402
from pysceptre.pipeline.trans import trans_fit_chunks  # noqa: E402

_TARGET_NAME = re.compile(r"^(chr[0-9XYM]+):(\d+)-(\d+)$")
_CANONICAL = {f"chr{c}" for c in [*range(1, 23), "X", "Y", "M"]}


def gene_positions_from_gtf(path: Path) -> pd.DataFrame:
    """`response_id` (Ensembl id without version), `chrom`, `tss` from the GTF's gene records.

    Canonical chromosomes only; GENCODE's `_PAR_Y` copies are dropped so each
    pseudoautosomal gene keeps its chrX record.
    """
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if len(f) < 9 or f[2] != "gene" or f[0] not in _CANONICAL:
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
            if gid.endswith("_PAR_Y"):
                continue
            tss = int(f[3]) if f[6] == "+" else int(f[4])
            rows.append((gid.split(".")[0], f[0], tss))
    out = pd.DataFrame(rows, columns=["response_id", "chrom", "tss"])
    dup = out["response_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"the annotation places {out.loc[dup, 'response_id'].nunique()} gene id(s) twice, "
            f"e.g. {out.loc[dup, 'response_id'].head(3).tolist()}"
        )
    return out


def element_positions(targets: list[str]) -> pd.DataFrame:
    """`grna_target`, `chrom`, `start`, `end`, `centre` parsed from `chr:start-end` names."""
    parsed = [_TARGET_NAME.match(t) for t in targets]
    bad = [t for t, m in zip(targets, parsed, strict=True) if m is None]
    if bad:
        raise ValueError(f"{len(bad)} target name(s) are not chr:start-end, e.g. {bad[:5]}")
    start = np.array([int(m.group(2)) for m in parsed])
    end = np.array([int(m.group(3)) for m in parsed])
    return pd.DataFrame(
        {
            "grna_target": targets,
            "chrom": [m.group(1) for m in parsed],
            "start": start,
            "end": end,
            "centre": (start + end) / 2.0,
        }
    )


def cmd_pairs(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ex = load_export(a.dataset)
    n_cells = ex.covariate_matrix.shape[0]
    m = ex.metadata
    print(
        f"loaded {len(ex.gene_ids)} genes x {n_cells} cells in {time.time() - t0:.0f}s", flush=True
    )

    reference = ex.pairs[["response_id", "grna_target"]].reset_index(drop=True)
    expected = (
        tuple(int(x) for x in a.expected_fit_chunks.split(",")) if a.expected_fit_chunks else None
    )
    chunks = trans_fit_chunks(
        ex.grna_target_cells,
        reference,
        n_cells,
        target_chunk_size=a.target_chunk_size,
        chunk_memory_gb=a.chunk_memory_gb,
        expected_fit_chunks=expected,
    )
    layout_hash = chunk_layout_hash(chunks)
    targets = [t for chunk in chunks for t in chunk]
    chunk_of = np.repeat(np.arange(len(chunks)), [len(c) for c in chunks])
    print(f"reference layout: {len(chunks)} chunks, width {len(chunks[0])}, hash {layout_hash}")

    elements = element_positions(targets)
    elements.insert(0, "target_code", np.arange(len(targets), dtype=np.int32))
    elements["fit_chunk"] = chunk_of.astype(np.int32)
    elements["n_trt_cells"] = [len(ex.grna_target_cells[t]) for t in targets]

    annotation = gene_positions_from_gtf(Path(a.gtf))
    row_of = {g: i for i, g in enumerate(ex.gene_ids)}
    cis_genes = [g for g in ex.gene_ids if g in set(reference["response_id"])]
    measured = annotation.loc[annotation["response_id"].isin(row_of)].reset_index(drop=True)
    unplaced = sorted(set(cis_genes) - set(measured["response_id"]))
    if unplaced:
        raise ValueError(
            f"{len(unplaced)} cis gene(s) have no position in {a.gtf}, e.g. {unplaced[:5]}"
        )
    genes = (
        pd.DataFrame({"response_id": cis_genes})
        .merge(measured, on="response_id", how="left", validate="1:1")
        .assign(gene_row=lambda d: d["response_id"].map(row_of).astype(np.int64))
    )
    genes.insert(0, "gene_code", np.arange(len(genes), dtype=np.int32))

    # Element x gene on another chromosome, gene chromosome not excluded.
    excluded = set(a.exclude_gene_chroms.split(",")) if a.exclude_gene_chroms else set()
    g_chrom = genes["chrom"].to_numpy()
    e_chrom = elements["chrom"].to_numpy()
    gene_code_parts, target_code_parts = [], []
    for c in pd.unique(e_chrom):
        es = np.flatnonzero(e_chrom == c).astype(np.int32)
        gs = np.flatnonzero((g_chrom != c) & ~np.isin(g_chrom, list(excluded))).astype(np.int32)
        gene_code_parts.append(np.repeat(gs, es.size))
        target_code_parts.append(np.tile(es, gs.size))
    gene_code = np.concatenate(gene_code_parts)
    target_code = np.concatenate(target_code_parts)
    order = np.lexsort((target_code, gene_code))
    gene_code, target_code = gene_code[order], target_code[order]
    print(f"trans pairs before QC: {gene_code.size:,}", flush=True)

    t0 = time.time()
    trt, cntrl = nonzero_counts(
        ex.response_matrix, [ex.grna_target_cells[t] for t in targets], n_cells
    )
    print(f"nonzero counts in {time.time() - t0:.0f}s", flush=True)
    rows = genes["gene_row"].to_numpy()
    n_trt = trt[rows[gene_code], target_code].astype(np.int32)
    n_cntrl = cntrl[rows[gene_code], target_code].astype(np.int32)
    thr_trt, thr_cntrl = int(m["n_nonzero_trt_thresh"]), int(m["n_nonzero_cntrl_thresh"])
    pass_qc = (n_trt >= thr_trt) & (n_cntrl >= thr_cntrl)

    # The cis side was QC'd by R; the counts must agree on its pairs.
    crosscheck = {}
    info = ex.discovery_pairs_with_info
    if info is not None and {"n_nonzero_trt", "n_nonzero_cntrl"} <= set(info.columns):
        t_code = {t: i for i, t in enumerate(targets)}
        known = info["grna_target"].isin(t_code) & info["response_id"].isin(row_of)
        sub = info.loc[known]
        r = sub["response_id"].map(row_of).to_numpy()
        c = sub["grna_target"].map(t_code).to_numpy()
        crosscheck = {
            "pairs_compared": int(len(sub)),
            "n_nonzero_trt_equal": int((trt[r, c] == sub["n_nonzero_trt"].to_numpy()).sum()),
            "n_nonzero_cntrl_equal": int((cntrl[r, c] == sub["n_nonzero_cntrl"].to_numpy()).sum()),
        }
        print(f"QC cross-check against the export's discovery pairs: {crosscheck}", flush=True)

    table = pa.table(
        {
            "gene_code": gene_code.astype(np.int32),
            "target_code": target_code.astype(np.int32),
            "n_nonzero_trt": n_trt,
            "n_nonzero_cntrl": n_cntrl,
            "pass_qc": pass_qc,
        }
    )
    pq.write_table(table, out / "trans_pairs.parquet", row_group_size=8_000_000)
    elements.to_parquet(out / "elements.parquet", index=False)
    genes.to_parquet(out / "genes.parquet", index=False)
    measured.to_parquet(out / "measured_genes.parquet", index=False)
    per_chunk = np.bincount(chunk_of[target_code[pass_qc]], minlength=len(chunks))
    layout = {
        "dataset": str(Path(a.dataset).resolve()),
        "gtf": str(Path(a.gtf).resolve()),
        "n_cells": int(n_cells),
        "n_reference_pairs": int(len(reference)),
        "target_chunk_size": a.target_chunk_size,
        "chunk_memory_gb": a.chunk_memory_gb,
        "fit_chunk_width": len(chunks[0]),
        "n_fit_chunks": len(chunks),
        "layout_hash": layout_hash,
        "n_elements": len(targets),
        "n_genes": len(genes),
        "exclude_gene_chroms": sorted(excluded),
        "qc_thresholds": [thr_trt, thr_cntrl],
        "n_trans_pairs": int(gene_code.size),
        "n_trans_pairs_pass_qc": int(pass_qc.sum()),
        "passing_pairs_per_chunk": per_chunk.tolist(),
        "qc_crosscheck": crosscheck,
    }
    (out / "layout.json").write_text(json.dumps(layout, indent=1))
    print(
        f"wrote {out}: {layout['n_trans_pairs_pass_qc']:,} of {layout['n_trans_pairs']:,} "
        "trans pairs pass QC"
    )


def cmd_shards(a: argparse.Namespace) -> None:
    out = Path(a.out)
    layout = json.loads((out / "layout.json").read_text())
    per_chunk = np.asarray(layout["passing_pairs_per_chunk"], dtype=np.int64)
    order = np.random.default_rng(a.seed).permutation(per_chunk.size)
    # Consecutive runs of the random order, cut where the cumulative passing
    # pairs cross each shard's share: random chunk sets of about equal work.
    cum = np.cumsum(per_chunk[order])
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, a.n_shards) / a.n_shards)
    shards = [sorted(int(i) for i in s) for s in np.split(order, cuts)]
    manifest = {
        "layout_hash": layout["layout_hash"],
        "n_fit_chunks": layout["n_fit_chunks"],
        "seed": a.seed,
        "shards": shards,
        "passing_pairs_per_shard": [int(per_chunk[s].sum()) for s in shards],
    }
    path = Path(a.manifest) if a.manifest else out / "shard_manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    print(f"{a.n_shards} shards, passing pairs per shard: {manifest['passing_pairs_per_shard']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-gene-chroms", default="chrY")
    p.add_argument("--target-chunk-size", type=int, default=200, help="the reference run's value")
    p.add_argument("--chunk-memory-gb", type=float, default=1.0, help="the reference run's value")
    p.add_argument("--expected-fit-chunks", help="width,count the reference layout must have")
    p.set_defaults(fn=cmd_pairs)
    s = sub.add_parser("shards")
    s.add_argument("--out", required=True)
    s.add_argument("--n-shards", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--manifest", help="where to write it; default OUT/shard_manifest.json")
    s.set_defaults(fn=cmd_shards)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
