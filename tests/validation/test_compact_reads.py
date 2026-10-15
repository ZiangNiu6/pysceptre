"""A compact read must give every row, and every result, of the eager read.

`load_export_compact` keeps only some genes and the cells in use, with 32-bit
cell indices, in blocks of consecutive genes. That is storage only: a row must
read back value for value, and an analysis must give the same numbers bit for
bit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

pytest.importorskip("mudata", reason="the io extra is not installed")

import sceptre_io  # noqa: E402
from sceptre_io import SceptreExport, load_export, load_export_compact, write_h5mu  # noqa: E402

from pysceptre.pipeline.api import run_discovery_analysis  # noqa: E402
from pysceptre.pipeline.discovery import _get_row  # noqa: E402
from pysceptre.pipeline.pairwise_qc import nonzero_counts  # noqa: E402

N_GENES, N_CELLS = 40, 3_000


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """An all-cells export: integer counts, some cells out of use, units in all-cells space."""
    rng = np.random.default_rng(1)
    X = sparse.random(N_GENES, N_CELLS, density=0.3, format="csr", random_state=1)
    X.data = np.ceil(X.data * 30).astype(np.float64)
    in_use = rng.random(N_CELLS) > 0.1
    targets = {f"t{i}": np.sort(rng.choice(N_CELLS, 150, replace=False)) for i in range(6)}
    export = SceptreExport(
        response_matrix=X,
        gene_ids=[f"g{i}" for i in range(N_GENES)],
        covariate_matrix=np.column_stack([np.ones(N_CELLS), rng.normal(size=(N_CELLS, 2))]),
        grna_target_cells=targets,
        pairs=pd.DataFrame(
            [(f"g{g}", t) for g in range(0, N_GENES, 3) for t in targets],
            columns=["response_id", "grna_target"],
        ),
        metadata={
            "n_cells": N_CELLS,
            "n_genes": N_GENES,
            "n_targets": len(targets),
            "n_pairs": 0,
            "n_covariates": 3,
            "n_nonzero": X.nnz,
            "covariate_names": ["intercept", "a", "b"],
            "side_code": 0,
            "run_permutations": False,
            "B1": 499,
            "B2": 4999,
            "B3": 0,
            "sceptre_version": "0.10.3",
            "all_cells": True,
        },
        in_use=in_use,
    )
    path = write_h5mu(export, tmp_path_factory.mktemp("compact") / "dataset.h5mu")
    return path


def _same_rows(eager, compact, genes):
    row_e = {g: i for i, g in enumerate(eager.gene_ids)}
    for j, g in enumerate(compact.gene_ids):
        assert g in genes
        a = _get_row(eager.response_matrix, row_e[g])
        b = _get_row(compact.response_matrix, j)
        assert a.dtype == b.dtype and np.array_equal(a, b), g


@pytest.mark.parametrize("genes", [None, ["g3", "g4", "g5", "g17", "g39", "g0"]])
def test_rows_equal_the_eager_read(dataset, genes):
    eager = load_export(dataset)
    compact = load_export_compact(dataset, genes=genes)
    want = eager.gene_ids if genes is None else [g for g in eager.gene_ids if g in set(genes)]
    assert compact.gene_ids == want
    _same_rows(eager, compact, set(want))
    assert compact.response_matrix.shape == (len(want), eager.response_matrix.shape[1])
    assert np.array_equal(compact.covariate_matrix, eager.covariate_matrix)
    assert compact.grna_target_cells.keys() == eager.grna_target_cells.keys()
    for t, cells in eager.grna_target_cells.items():
        assert np.array_equal(compact.grna_target_cells[t], cells)


def test_many_blocks_and_ranges_across_them(dataset, monkeypatch):
    monkeypatch.setattr(sceptre_io, "_MAX_BLOCK_ENTRIES", 2_000)
    eager = load_export(dataset)
    compact = load_export_compact(dataset, read_entries=1_500)
    m = compact.response_matrix
    assert len(m._blocks) > 3
    assert all(b.indices.dtype == np.int32 for b in m._blocks)
    _same_rows(eager, compact, set(eager.gene_ids))
    a = eager.response_matrix[5:31].toarray()
    b = m.rows(5, 31).toarray()
    assert np.array_equal(a, b)


def test_qc_counts_and_discovery_are_identical(dataset):
    eager = load_export(dataset)
    genes = sorted(set(eager.pairs["response_id"]), key=eager.gene_ids.index)
    compact = load_export_compact(dataset, genes=genes)
    cells = list(eager.grna_target_cells.values())
    n = eager.covariate_matrix.shape[0]
    row_e = [eager.gene_ids.index(g) for g in genes]
    trt_e, ctl_e = nonzero_counts(eager.response_matrix, cells, n)
    trt_c, ctl_c = nonzero_counts(compact.response_matrix, cells, n)
    assert np.array_equal(trt_e[row_e], trt_c) and np.array_equal(ctl_e[row_e], ctl_c)
    runs = [
        run_discovery_analysis(
            ex.response_matrix,
            ex.gene_ids,
            ex.covariate_matrix,
            ex.grna_target_cells,
            eager.pairs,
            seed=5,
        )
        for ex in (eager, compact)
    ]
    pd.testing.assert_frame_equal(runs[0], runs[1], check_exact=True)


def test_unknown_gene_is_refused(dataset):
    with pytest.raises(KeyError, match="not in"):
        load_export_compact(dataset, genes=["g1", "nope"])
