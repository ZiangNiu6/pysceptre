"""`reference_fit_chunks` is the layout the complement engine fits its targets in.

A target's batched logistic fit depends on its chunk-mates in the last bits,
so a run that reproduces another run's target fits needs that run's chunks.
The engine builds them with `reference_fit_chunks`; these tests record the
target lists the engine actually hands `fit_all_targets` and compare them with
the function's, on the CRT (default, budget-reduced and explicit widths, pairs
in another order) and on the permutation path. See docs/design.md, "The trans
background".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline import discovery
from pysceptre.pipeline.api import run_discovery_analysis
from pysceptre.pipeline.discovery import (
    chunk_layout_hash,
    reference_fit_chunks,
    target_bytes_per_item,
)

N_CELLS = 2_000
N_TARGETS = 10
SKEW_NORMAL_B = 499 + 4999
PERMUTATION_B = 499 + 4999 + 24999
DEFAULT_WIDTH = 200
DEFAULT_BUDGET_GB = 1.0


@pytest.fixture(scope="module")
def screen():
    """Three genes and ten targets of 30 to 59 cells, fixed seed."""
    rng = np.random.default_rng(20261014)
    cov = np.column_stack([np.ones(N_CELLS), rng.normal(size=N_CELLS)])
    mu = np.exp(cov @ np.array([1.2, 0.2]))
    resp = rng.negative_binomial(6.0, 6.0 / (6.0 + mu), size=(3, N_CELLS)).astype(float)
    genes = [f"gene_{i}" for i in range(3)]
    cells = {
        f"target_{t}": np.sort(rng.choice(N_CELLS, int(rng.integers(30, 60)), replace=False))
        for t in range(N_TARGETS)
    }
    return resp, genes, cov, cells


def _all_pairs(genes, targets) -> pd.DataFrame:
    return pd.DataFrame(
        [(g, t) for g in genes for t in targets], columns=["response_id", "grna_target"]
    )


def _engine_chunks(monkeypatch, **kwargs):
    """The target lists `run_discovery_analysis` passes `fit_all_targets`, in call order.

    `n_jobs=1`, so chunks are prepared one after another, in layout order.
    """
    seen: list[list[str]] = []
    original = discovery.fit_all_targets

    def spy(grna_target_cells, *args, **kw):
        seen.append(list(grna_target_cells))
        return original(grna_target_cells, *args, **kw)

    with monkeypatch.context() as m:
        m.setattr(discovery, "fit_all_targets", spy)
        result = run_discovery_analysis(**kwargs, seed=11, n_jobs=1)
    return seen, result


def _reduction_warnings(record) -> list[str]:
    return [str(w.message) for w in record if "reducing target_chunk_size" in str(w.message)]


def test_default_layout_is_the_engines(screen, monkeypatch):
    resp, genes, cov, cells = screen
    pairs = _all_pairs(genes, cells)
    calls = []
    original = discovery.reference_fit_chunks

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(discovery, "reference_fit_chunks", counting)
    seen, _ = _engine_chunks(
        monkeypatch,
        response_matrix=resp,
        gene_ids=genes,
        covariate_matrix=cov,
        grna_target_cells=cells,
        pairs=pairs,
    )
    assert len(calls) == 1, "the engine must take its layout from reference_fit_chunks"
    expected = original(
        cells,
        pairs,
        N_CELLS,
        B_total=SKEW_NORMAL_B,
        target_chunk_size=DEFAULT_WIDTH,
        chunk_memory_gb=DEFAULT_BUDGET_GB,
    )
    assert seen == expected
    assert expected == [list(cells)]


def test_a_budget_reduced_width_is_the_engines_and_warns_the_same(screen, monkeypatch):
    resp, genes, cov, cells = screen
    pairs = _all_pairs(genes, cells)
    per_target = target_bytes_per_item(N_CELLS, SKEW_NORMAL_B, [len(c) for c in cells.values()])
    budget = 3.5 * per_target / 1e9  # room for three targets

    with warnings.catch_warnings(record=True) as engine_record:
        warnings.simplefilter("always")
        seen, _ = _engine_chunks(
            monkeypatch,
            response_matrix=resp,
            gene_ids=genes,
            covariate_matrix=cov,
            grna_target_cells=cells,
            pairs=pairs,
            chunk_memory_gb=budget,
        )
    with warnings.catch_warnings(record=True) as direct_record:
        warnings.simplefilter("always")
        expected = reference_fit_chunks(
            cells,
            pairs,
            N_CELLS,
            B_total=SKEW_NORMAL_B,
            target_chunk_size=DEFAULT_WIDTH,
            chunk_memory_gb=budget,
        )

    assert seen == expected
    assert [len(c) for c in expected] == [3, 3, 3, 1]
    engine_messages = _reduction_warnings(engine_record)
    assert len(engine_messages) == 1
    assert engine_messages == _reduction_warnings(direct_record)
    assert engine_messages[0].startswith(f"reducing target_chunk_size from {DEFAULT_WIDTH} to 3")


def test_an_explicit_width_is_the_engines(screen, monkeypatch):
    resp, genes, cov, cells = screen
    pairs = _all_pairs(genes, cells)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        seen, _ = _engine_chunks(
            monkeypatch,
            response_matrix=resp,
            gene_ids=genes,
            covariate_matrix=cov,
            grna_target_cells=cells,
            pairs=pairs,
            target_chunk_size=4,
        )
    expected = reference_fit_chunks(
        cells,
        pairs,
        N_CELLS,
        B_total=SKEW_NORMAL_B,
        target_chunk_size=4,
        chunk_memory_gb=DEFAULT_BUDGET_GB,
    )
    assert seen == expected
    assert [len(c) for c in expected] == [4, 4, 2]
    assert not _reduction_warnings(record)


@pytest.mark.parametrize("categorical", [False, True])
def test_targets_follow_grna_target_cells_order_not_the_pairs(screen, monkeypatch, categorical):
    """Pairs list targets shuffled and skip two; the chunks keep the dict's order."""
    resp, genes, cov, cells = screen
    # A dict order that is neither sorted nor the pairs' order.
    order = [f"target_{t}" for t in (7, 2, 9, 0, 5, 3, 8, 1, 6, 4)]
    reordered = {t: cells[t] for t in order}
    named = [t for t in order if t not in ("target_5", "target_1")]
    pairs = _all_pairs(genes[:2], named).sample(frac=1.0, random_state=3).reset_index(drop=True)
    if categorical:
        # Unused categories must not count as named targets.
        pairs["grna_target"] = pd.Categorical(pairs["grna_target"], categories=sorted(order))
        pairs["response_id"] = pd.Categorical(pairs["response_id"], categories=genes)

    seen, result = _engine_chunks(
        monkeypatch,
        response_matrix=resp,
        gene_ids=genes,
        covariate_matrix=cov,
        grna_target_cells=reordered,
        pairs=pairs,
        target_chunk_size=3,
    )
    expected = reference_fit_chunks(
        reordered,
        pairs,
        N_CELLS,
        B_total=SKEW_NORMAL_B,
        target_chunk_size=3,
        chunk_memory_gb=DEFAULT_BUDGET_GB,
    )
    assert seen == expected
    assert [t for chunk in expected for t in chunk] == [
        t for t in order if t not in ("target_5", "target_1")
    ]
    assert len(result) == len(pairs)


@pytest.mark.parametrize("width", [DEFAULT_WIDTH, 4])
def test_the_permutation_layout_is_the_engines(screen, monkeypatch, width):
    """One chunk at the default width, whatever the budget; an explicit width is honoured."""
    resp, genes, cov, cells = screen
    pairs = _all_pairs(genes, cells)
    tiny_budget = 1e-7
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        seen, _ = _engine_chunks(
            monkeypatch,
            response_matrix=resp,
            gene_ids=genes,
            covariate_matrix=cov,
            grna_target_cells=cells,
            pairs=pairs,
            resampling_mechanism="permutations",
            target_chunk_size=width,
            chunk_memory_gb=tiny_budget,
        )
    expected = reference_fit_chunks(
        cells,
        pairs,
        N_CELLS,
        B_total=PERMUTATION_B,
        target_chunk_size=width,
        chunk_memory_gb=tiny_budget,
        resampling_mechanism="permutations",
    )
    assert seen == expected
    assert [len(c) for c in expected] == ([N_TARGETS] if width == DEFAULT_WIDTH else [4, 4, 2])
    assert not _reduction_warnings(record)


def test_no_named_target_gives_no_chunk(screen):
    _, _, _, cells = screen
    pairs = pd.DataFrame({"response_id": ["gene_0"], "grna_target": ["not_a_target"]})
    for mechanism in ("crt", "permutations"):
        assert (
            reference_fit_chunks(
                cells,
                pairs,
                N_CELLS,
                B_total=SKEW_NORMAL_B,
                target_chunk_size=DEFAULT_WIDTH,
                chunk_memory_gb=DEFAULT_BUDGET_GB,
                resampling_mechanism=mechanism,
            )
            == []
        )


def test_an_unknown_mechanism_is_refused(screen):
    _, genes, _, cells = screen
    with pytest.raises(ValueError, match="resampling_mechanism"):
        reference_fit_chunks(
            cells,
            _all_pairs(genes, cells),
            N_CELLS,
            B_total=SKEW_NORMAL_B,
            target_chunk_size=DEFAULT_WIDTH,
            chunk_memory_gb=DEFAULT_BUDGET_GB,
            resampling_mechanism="crt_exact",
        )


# --- chunk_layout_hash ----------------------------------------------------


def test_layout_hash_is_stable():
    """Pinned: a layout hash recorded by one release must be recomputable by the next."""
    assert (
        chunk_layout_hash([["t1", "t2"], ["t3"]])
        == "587ee3bc7d91015ba37f0863a1d6d0255a679680c55d3850fc0ed2fc29b7d4ad"
    )
    assert chunk_layout_hash([("t1", "t2"), ("t3",)]) == chunk_layout_hash([["t1", "t2"], ["t3"]])
    assert chunk_layout_hash([[np.str_("t1"), "t2"], ["t3"]]) == chunk_layout_hash(
        [["t1", "t2"], ["t3"]]
    )


def test_layout_hash_is_order_and_boundary_sensitive():
    base = chunk_layout_hash([["t1", "t2"], ["t3"]])
    others = [
        [["t2", "t1"], ["t3"]],  # order within a chunk
        [["t3"], ["t1", "t2"]],  # order of chunks
        [["t1"], ["t2", "t3"]],  # where a boundary falls
        [["t1", "t2", "t3"]],  # no boundary
        [["t1", "t2"], ["t3"], []],  # an empty chunk
        [["t1,t2"], ["t3"]],  # a separator inside an id
        [['t1"', "t2"], ["t3"]],  # a quote inside an id
    ]
    digests = [chunk_layout_hash(layout) for layout in others]
    assert base not in digests
    assert len(set(digests)) == len(digests)
