"""The trans background's screening primitives reproduce sceptre's test exactly.

`empirical_p_from_counts` is `compute_empirical_p_value` from two counts, and
with counts over a prefix of the nulls it bounds the final p-value from below.
`FlatDraws` builds the CSR matrices `draws_to_matrix` builds, and a stage's
null statistics evaluated in row blocks are, bit for bit, the rows of one
matmul over the stage. `run_low_level_test_screened` with both screens off is
`run_low_level_test_full` field for field; with them on, every pair it tests
in full is too, and every pair it skips is one no call can see. All on a small
simulated NB screen with the engine's own gene fits, target fits and CRT draws.
See docs/design.md, "The trans background".
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pysceptre.crt.sampler import (
    EXACT_SAMPLER_ABOVE_SHARE,
    crt_index_sampler_exact,
    crt_index_sampler_fast,
)
from pysceptre.pipeline.discovery import fit_all_genes, fit_all_targets
from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.empirical_p import (
    compute_empirical_p_value,
    empirical_p_from_counts,
)
from pysceptre.test_statistic.resampling import (
    P_THRESH,
    run_low_level_test_full,
    run_low_level_test_screened,
)
from pysceptre.test_statistic.score_stat import (
    FlatDraws,
    ListDraws,
    compute_null_statistics_block,
    compute_null_statistics_from_draws,
    draws_to_matrix,
    stack_pieces,
    statistics_from_segment_sums,
)

N_CELLS, N_GENES, SEED = 12_000, 16, 20261014
B1, B2 = 499, 4999
# Targets on both sides of the exact sampler's threshold (24 cells here).
TARGET_SIZES = (10, 14, 16, 18, 22, 24, 26, 30, 35, 40, 45, 50, 55, 60)
# (gene, target, factor on the target's cells): down and up effects.
PLANTED = (
    (0, 0, 0.25),
    (1, 7, 0.3),
    (2, 9, 0.35),
    (3, 12, 0.4),
    (4, 3, 3.0),
    (5, 10, 2.5),
    (6, 11, 0.5),
    (7, 5, 1.8),
)
SIDES = (-1, 0, 1)
BLOCK_ROWS = (1, 7, 32)


def _bits(x) -> int:
    return int(np.float64(x).view(np.int64))


def _differences(r, full) -> list[str]:
    """The fields in which a screened result differs from the full one, bit for bit."""
    out = [
        name
        for name in ("p_value", "z_orig", "fold_change", "se_fold_change", "stage")
        if _bits(getattr(r, name)) != _bits(getattr(full, name))
    ]
    if (r.sn_params is None) != (full.sn_params is None) or (
        r.sn_params is not None
        and [_bits(v) for v in r.sn_params] != [_bits(v) for v in full.sn_params]
    ):
        out.append("sn_params")
    return out


def _n_null_full(full, fit_parametric_curve: bool, b2: int, b3: int) -> int:
    """Null statistics `run_low_level_test_full` computed, from the stage it ended at."""
    if not np.isfinite(full.stage):
        return 0
    n = B1
    if full.stage >= 2 and fit_parametric_curve:
        n += b2
    if full.stage == 3 and b3 > 0:
        n += b3
    return n


def _smallest_unescalated_p(side: int) -> float:
    """The smallest first-stage p-value above `P_THRESH`, i.e. of a pair that does not escalate."""
    grid = [empirical_p_from_counts(c, c, B1, side) for c in range(B1 + 1)]
    return min(p for p in grid if p > P_THRESH)


@pytest.fixture(scope="module")
def case():
    """A screen of 12,000 cells and five covariates (intercept, depth, two batch
    one-hots, a continuous covariate), 16 NB genes and 14 targets, with the
    engine's width-1 gene fits, its batched target fit and its CRT draws."""
    rng = np.random.default_rng(SEED)
    depth = rng.normal(0.0, 0.3, N_CELLS)
    batch = rng.integers(0, 3, N_CELLS)
    X = np.column_stack(
        [
            np.ones(N_CELLS),
            depth,
            (batch == 1).astype(float),
            (batch == 2).astype(float),
            rng.normal(size=N_CELLS),
        ]
    )
    target_ids = [f"t{j:02d}" for j in range(len(TARGET_SIZES))]
    cells = {
        t: np.sort(rng.choice(N_CELLS, size, replace=False))
        for t, size in zip(target_ids, TARGET_SIZES, strict=True)
    }
    base = rng.uniform(1.0, 6.0, N_GENES)
    theta = rng.uniform(2.0, 20.0, N_GENES)
    mu = base[:, None] * np.exp(0.8 * depth + 0.2 * (batch == 1) - 0.3 * (batch == 2))[None, :]
    for g, t, factor in PLANTED:
        mu[g, cells[target_ids[t]]] *= factor
    counts = rng.negative_binomial(theta[:, None], theta[:, None] / (theta[:, None] + mu)).astype(
        float
    )
    gene_ids = [f"g{i:02d}" for i in range(N_GENES)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = fit_all_genes(counts, gene_ids, X, batch_width=1)
        targets = fit_all_targets(cells, X, B1=B1, B2=B2, B3=0, seed=SEED)
    genes = []
    for i, g in enumerate(gene_ids):
        pieces = compute_precomputation_pieces(counts[i], X, fits[g].fitted_coefs, fits[g].theta)
        genes.append(
            dict(
                y=counts[i],
                mu=pieces.mu,
                a=pieces.a,
                w=pieces.w,
                D=pieces.D,
                stacked=stack_pieces(pieces.a, pieces.w, pieces.D),
            )
        )
    draws = [targets[t].draws._idxs for t in target_ids]
    return dict(
        X=X,
        genes=genes,
        trt=[targets[t].trt_idxs for t in target_ids],
        draws=draws,
        flat={},
        full={},
        pairs=[(g, t) for g in range(N_GENES) for t in range(len(target_ids))],
    )


def _flat(case, t: int, block_rows: int | None) -> FlatDraws:
    """Target `t`'s draws as `FlatDraws`, with blocks of `block_rows` prebuilt (None: none)."""
    key = (t, block_rows)
    if key not in case["flat"]:
        flat = FlatDraws(case["draws"][t], N_CELLS)
        if block_rows is not None:
            flat.prebuild_blocks(B1, block_rows)
        case["flat"][key] = flat
    return case["flat"][key]


def _inputs(case, g: int, t: int) -> dict:
    gene = case["genes"][g]
    return dict(
        y=gene["y"],
        mu=gene["mu"],
        a=gene["a"],
        w=gene["w"],
        D=gene["D"],
        trt_idxs=case["trt"][t],
        stacked=gene["stacked"],
    )


def _full_results(case, side: int, fit_parametric_curve=True, b2=B2, b3=0) -> list:
    """`run_low_level_test_full` on every pair, from `ListDraws` as the 0.5.0 engine holds them."""
    key = (side, fit_parametric_curve, b2, b3)
    if key not in case["full"]:
        list_draws = [ListDraws(d, N_CELLS) for d in case["draws"]]
        case["full"][key] = [
            run_low_level_test_full(
                **_inputs(case, g, t),
                synthetic_idxs=list_draws[t],
                B1=B1,
                B2=b2,
                B3=b3,
                fit_parametric_curve=fit_parametric_curve,
                side_code=side,
            )
            for g, t in case["pairs"]
        ]
    return case["full"][key]


# --- empirical_p_from_counts -------------------------------------------------------------


def _counts(nulls: np.ndarray, z: float) -> tuple[int, int]:
    return int(np.count_nonzero(z >= nulls)), int(np.count_nonzero(z <= nulls))


@pytest.mark.parametrize("side", SIDES)
def test_counts_formula_equals_empirical_p_value(side):
    rng = np.random.default_rng(1)
    grid = np.round(rng.normal(size=12), 2)
    for B in (0, 1, 2, 31, 32, 499, 4999):
        for rep in range(20):
            nulls = rng.choice(grid, B)
            if B and rep % 2:
                nulls[rng.random(B) < 0.1] = np.nan
            if B > 2 and rep % 5 == 0:
                nulls[:2] = (np.inf, -np.inf)
            for z in (*rng.choice(grid, 4), rng.normal(), -10.0, 10.0):
                want = compute_empirical_p_value(nulls, z, side)
                got = empirical_p_from_counts(*_counts(nulls, z), B, side)
                assert got == want and _bits(got) == _bits(want), (B, z, side)


@pytest.mark.parametrize("side", SIDES)
def test_bound_at_every_prefix_is_at_most_the_final_p(side):
    rng = np.random.default_rng(2)
    for rep in range(40):
        nulls = rng.normal(size=B1)
        nulls[rng.random(B1) < 0.02] = np.nan
        z = (rng.normal() * 3) if rep % 4 else float(nulls[~np.isnan(nulls)][0])
        final = compute_empirical_p_value(nulls, z, side)
        left = np.concatenate([[0], np.cumsum(z >= nulls)])
        right = np.concatenate([[0], np.cumsum(z <= nulls)])
        bounds = np.array(
            [
                empirical_p_from_counts(int(cl), int(cr), B1, side)
                for cl, cr in zip(left, right, strict=True)
            ]
        )
        assert (bounds <= final).all()
        assert (np.diff(bounds) >= 0).all()
        assert bounds[-1] == final


# --- FlatDraws ---------------------------------------------------------------------------


def _assert_same_matrix(got, want, stacked):
    assert got.shape == want.shape
    assert np.array_equal(got.indptr, want.indptr)
    assert np.array_equal(got.indices, want.indices)
    assert np.array_equal(got.data, want.data)
    assert np.array_equal(got.toarray(), want.toarray())
    if got.shape[0]:
        assert np.array_equal((got @ stacked).view(np.int64), (want @ stacked).view(np.int64))


def _draw_lists():
    """Draw lists of every shape `FlatDraws` must hold."""
    rng = np.random.default_rng(3)
    n = 2_000
    p_small = np.full(n, 30 / n)
    p_dup = np.zeros(n)
    p_dup[:200] = 0.05  # a share far above the fast sampler's use, so it repeats cells
    p_exact = np.full(n, 0.02)
    fast = crt_index_sampler_fast(p_small, B1 + B2, rng)
    dup = crt_index_sampler_fast(p_dup, B1 + B2, rng)
    assert any(np.unique(d).size < d.size for d in dup)
    exact = crt_index_sampler_exact(p_exact, B1 + B2, rng)
    hand = [d.copy() for d in fast[:600]]
    hand[3] = np.empty(0, dtype=np.int64)
    hand[100] = np.empty(0, dtype=np.int64)
    hand[5] = np.concatenate([hand[5], hand[5][:2], [n - 1, n - 1]])
    return n, {
        "fast": fast,
        "duplicates": dup,
        "exact": exact,
        "empty_and_repeated": hand,
        "all_empty": [np.empty(0, dtype=np.int64) for _ in range(40)],
        "none": [],
    }


def test_flat_draws_build_the_matrices_draws_to_matrix_builds():
    n, lists = _draw_lists()
    stacked = np.random.default_rng(4).normal(size=(n, 7)) * np.exp(
        np.random.default_rng(5).normal(size=(n, 7))
    )
    for idxs in lists.values():
        B = len(idxs)
        flat = FlatDraws(idxs, n)
        assert flat.n_draws == B and flat._cells.dtype == np.int32
        assert flat._offsets.dtype == np.int64
        ranges = [(0, B1), (B1, B1 + B2), (0, B), (3, 4), (0, 1), (B - 1, B), (98, 103)]
        for lo, hi in ranges:
            if 0 <= lo < hi <= B:
                _assert_same_matrix(flat.slice(lo, hi), draws_to_matrix(idxs[lo:hi], n), stacked)
        # Out-of-range and empty requests clamp as every StagedDraws does.
        ref = ListDraws(idxs, n)
        for lo, hi in [(-3, 10), (B - 2, B + 10), (5, 5), (7, 2), (B + 1, B + 3)]:
            _assert_same_matrix(flat.slice(lo, hi), ref.slice(lo, hi), stacked)
        # Not memoized: a fresh matrix each call, nothing kept.
        if B:
            assert flat.slice(0, B) is not flat.slice(0, B)
        assert not flat._cache and not flat._blocks
        for block_rows in (1, 7, 32, 499):
            flat.prebuild_blocks(B1, block_rows)
            stop = min(B1, B)
            assert len(flat._blocks) == -(-stop // block_rows)
            for lo in range(0, stop, block_rows):
                hi = min(lo + block_rows, stop)
                got = flat.slice(lo, hi)
                assert got is flat.slice(lo, hi) is flat._blocks[(lo, hi)]
                _assert_same_matrix(got, draws_to_matrix(idxs[lo:hi], n), stacked)
                assert not got.data.flags.writeable and not got.indices.flags.writeable
            # Anything else is still built on the fly, and still equal.
            if B > 1:
                _assert_same_matrix(flat.slice(1, 2), draws_to_matrix(idxs[1:2], n), stacked)
            assert not flat._cache


def test_flat_draws_cells_are_int64_when_int32_cannot_hold_them():
    flat = FlatDraws([np.array([2**31 + 5, 3]), np.array([], dtype=np.int64)], 2**31 + 10)
    assert flat._cells.dtype == np.int64
    assert np.array_equal(flat.slice(0, 2).indices, [2**31 + 5, 3])


@pytest.mark.parametrize("bad", [[np.array([0, 1]), np.array([-1, 2])], [np.array([4, 5])]])
def test_flat_draws_refuse_out_of_range_cells_as_draws_to_matrix_does(bad):
    with pytest.raises(ValueError) as want:
        draws_to_matrix(bad, 5)
    with pytest.raises(ValueError) as got:
        FlatDraws(bad, 5)
    assert str(got.value) == str(want.value)
    assert "out of range" in str(got.value)


# --- block null statistics ---------------------------------------------------------------


def _assert_bitwise(got: np.ndarray, want: np.ndarray):
    assert got.shape == want.shape
    assert np.array_equal(got, want, equal_nan=True)
    finite = ~np.isnan(want)
    assert np.array_equal(got[finite].view(np.int64), want[finite].view(np.int64))
    # NaN payloads too: an empty resample's 0/0 is the same NaN in a block.
    assert np.array_equal(got.view(np.int64), want.view(np.int64))


def _block_nulls(stacked, draws, block_rows: int) -> np.ndarray:
    return np.concatenate(
        [
            compute_null_statistics_block(stacked, draws.slice(lo, min(lo + block_rows, B1)))
            for lo in range(0, B1, block_rows)
        ]
    )


@pytest.mark.parametrize("block_rows", (1, 7, 32, 499))
def test_block_nulls_are_the_rows_of_the_full_product(case, block_rows):
    idxs = [d.copy() for d in case["draws"][2]]
    idxs[3] = np.empty(0, dtype=np.int64)
    idxs[100] = np.empty(0, dtype=np.int64)
    idxs[5] = np.concatenate([idxs[5], idxs[5][:2]])
    sources = {"edited": idxs}
    sources.update({f"t{t}": case["draws"][t] for t in (0, 6, 13)})
    for g in (0, 4, 9):
        stacked = case["genes"][g]["stacked"]
        for name, src in sources.items():
            whole = draws_to_matrix(src[:B1], N_CELLS)
            want = statistics_from_segment_sums(whole @ stacked)
            _assert_bitwise(compute_null_statistics_from_draws(stacked, whole), want)
            if name == "edited":
                assert np.isnan(want[[3, 100]]).all()
            prebuilt = FlatDraws(src, N_CELLS)
            prebuilt.prebuild_blocks(B1, block_rows)
            for draws in (prebuilt, FlatDraws(src, N_CELLS), ListDraws(src, N_CELLS)):
                _assert_bitwise(_block_nulls(stacked, draws, block_rows), want)


def test_block_nulls_of_resamples_with_no_cells():
    rng = np.random.default_rng(6)
    stacked = rng.normal(size=(50, 6))
    empty = FlatDraws([np.empty(0, dtype=np.int64)] * 40, 50)
    want = statistics_from_segment_sums(draws_to_matrix(empty._index_arrays(0, 40), 50) @ stacked)
    assert np.isnan(want).all()
    for block_rows in (1, 7, 40):
        got = np.concatenate(
            [
                compute_null_statistics_block(stacked, empty.slice(lo, min(lo + block_rows, 40)))
                for lo in range(0, 40, block_rows)
            ]
        )
        _assert_bitwise(got, want)
    assert compute_null_statistics_block(stacked, empty.slice(5, 5)).shape == (0,)


# --- run_low_level_test_screened ---------------------------------------------------------


@pytest.mark.parametrize("block_rows", BLOCK_ROWS)
@pytest.mark.parametrize("side", SIDES)
def test_screens_off_is_the_full_test(case, side, block_rows):
    full = _full_results(case, side)
    bad = []
    for (g, t), want in zip(case["pairs"], full, strict=True):
        r = run_low_level_test_screened(
            **_inputs(case, g, t),
            synthetic_idxs=_flat(case, t, block_rows),
            side_code=side,
            screen_fold_change=False,
            p_exact_below=None,
            block_rows=block_rows,
        )
        if _differences(r, want):
            bad.append((g, t, _differences(r, want)))
        assert r.screen == "none" and np.isnan(r.p_value_floor)
        assert r.n_null_evaluated == _n_null_full(want, True, B2, 0)
    assert not bad
    assert any(f.stage == 2 for f in full) and any(f.stage == 1 for f in full)


@pytest.mark.parametrize(
    ("fit_parametric_curve", "b2", "b3"),
    [(False, B2, 0), (True, 2000, 2999), (False, 2000, 2999)],
)
def test_screens_off_is_the_full_test_on_every_budget(case, fit_parametric_curve, b2, b3):
    full = _full_results(case, 0, fit_parametric_curve, b2, b3)
    for (g, t), want in zip(case["pairs"], full, strict=True):
        for screens in ((False, None), (True, 0.05)):
            r = run_low_level_test_screened(
                **_inputs(case, g, t),
                synthetic_idxs=_flat(case, t, 7),
                B2=b2,
                B3=b3,
                fit_parametric_curve=fit_parametric_curve,
                side_code=0,
                screen_fold_change=screens[0],
                p_exact_below=screens[1],
                block_rows=7,
            )
            if r.screen == "none":
                assert not _differences(r, want), (g, t, screens)
                assert r.n_null_evaluated == _n_null_full(want, fit_parametric_curve, b2, b3)
    # Without the skew-normal fit an escalated pair goes straight to stage 3.
    assert any(f.stage == (2 if fit_parametric_curve else 3) for f in full)


def _check_screened(case, full, results, side, block_rows, p_exact_below):
    """Every invariant the screens promise, over all pairs; returns the screen counts."""
    floor_min = max(p_exact_below, _smallest_unescalated_p(side))
    kinds = {"none": 0, "fold_change": 0, "early_stop": 0}
    for (g, t), want, r in zip(case["pairs"], full, results, strict=True):
        kinds[r.screen] += 1
        for name in ("z_orig", "fold_change", "se_fold_change"):
            assert _bits(getattr(r, name)) == _bits(getattr(want, name)), (g, t, name)
        if want.stage >= 2 and want.fold_change < 1:
            assert r.screen == "none", (g, t)
        if r.screen == "none":
            assert not _differences(r, want), (g, t)
            assert np.isnan(r.p_value_floor)
        elif r.screen == "fold_change":
            assert want.fold_change >= 1.0
            assert np.isnan(r.p_value) and np.isnan(r.stage) and np.isnan(r.p_value_floor)
            assert r.n_null_evaluated == 0
        else:
            assert r.screen == "early_stop"
            assert want.stage == 1 and r.stage == 1 and np.isnan(r.p_value)
            assert want.p_value >= r.p_value_floor >= floor_min, (g, t)
            n = r.n_null_evaluated
            assert 0 < n < B1 and n % block_rows == 0
            # The floor is the bound at the block where it stopped, and the
            # block before it did not stop yet.
            nulls = compute_null_statistics_from_draws(
                case["genes"][g]["stacked"], draws_to_matrix(case["draws"][t][:B1], N_CELLS)
            )
            z = want.z_orig
            assert r.p_value_floor == empirical_p_from_counts(*_counts(nulls[:n], z), B1, side)
            if n > block_rows:
                prev = empirical_p_from_counts(*_counts(nulls[: n - block_rows], z), B1, side)
                assert not (prev > P_THRESH and prev >= p_exact_below)
    return kinds


@pytest.mark.parametrize("block_rows", BLOCK_ROWS)
@pytest.mark.parametrize("side", SIDES)
def test_screens_on_are_exact(case, side, block_rows):
    full = _full_results(case, side)
    results = [
        run_low_level_test_screened(
            **_inputs(case, g, t),
            synthetic_idxs=_flat(case, t, block_rows),
            side_code=side,
            block_rows=block_rows,
        )
        for g, t in case["pairs"]
    ]
    kinds = _check_screened(case, full, results, side, block_rows, 0.05)
    assert kinds["early_stop"] and kinds["fold_change"]
    # Right-sided, a pair with fold change below 1 stops early almost at once,
    # so fully tested pairs are expected only on the other two sides.
    assert kinds["none"] or side == 1
    escalated_down = [f.stage >= 2 and f.fold_change < 1 for f in full]
    escalated_up = [f.stage >= 2 and f.fold_change >= 1 for f in full]
    if side <= 0:
        assert any(escalated_down)
    if side >= 0:
        # Escalated up-regulated pairs are exactly what the fold-change screen skips.
        assert any(escalated_up)
        assert all(
            r.screen == "fold_change" for r, up in zip(results, escalated_up, strict=True) if up
        )


@pytest.mark.parametrize("p_exact_below", (0.01, 0.3))
def test_screens_on_are_exact_at_other_stop_levels(case, p_exact_below):
    full = _full_results(case, 0)
    results = [
        run_low_level_test_screened(
            **_inputs(case, g, t),
            synthetic_idxs=_flat(case, t, 32),
            side_code=0,
            p_exact_below=p_exact_below,
        )
        for g, t in case["pairs"]
    ]
    kinds = _check_screened(case, full, results, 0, 32, p_exact_below)
    assert kinds["early_stop"]


def test_every_draw_source_gives_the_same_screened_result(case):
    """Prebuilt blocks, blocks of another size, no blocks, `ListDraws` and a plain list."""
    for g, t in case["pairs"][::3]:
        results = [
            run_low_level_test_screened(
                **_inputs(case, g, t), synthetic_idxs=src, side_code=0, block_rows=32
            )
            for src in (
                _flat(case, t, 32),
                _flat(case, t, 7),
                _flat(case, t, None),
                ListDraws(case["draws"][t], N_CELLS),
                case["draws"][t],
            )
        ]
        first = results[0]
        for r in results[1:]:
            assert not _differences(r, first)
            assert (r.screen, r.n_null_evaluated) == (first.screen, first.n_null_evaluated)
            assert _bits(r.p_value_floor) == _bits(first.p_value_floor)


def test_untestable_pairs_get_the_full_tests_result(case):
    g, t = 4, 3  # a planted up effect, so the fold-change screen would apply
    inputs = _inputs(case, g, t)
    draws = _flat(case, t, 32)
    assert run_low_level_test_screened(**inputs, synthetic_idxs=draws).screen == "fold_change"

    empty = dict(inputs, trt_idxs=np.empty(0, dtype=np.int64))
    want = run_low_level_test_full(**empty, synthetic_idxs=draws)
    got = run_low_level_test_screened(**empty, synthetic_idxs=draws)
    assert not _differences(got, want)
    assert (got.screen, got.n_null_evaluated) == ("none", 0) and np.isnan(got.p_value_floor)

    # A non-finite statistic is reported before the fold-change screen is applied.
    a = inputs["a"].copy()
    a[inputs["trt_idxs"][0]] = np.nan
    broken = dict(inputs, a=a, stacked=stack_pieces(a, inputs["w"], inputs["D"]))
    want = run_low_level_test_full(**broken, synthetic_idxs=draws)
    got = run_low_level_test_screened(**broken, synthetic_idxs=draws)
    assert np.isnan(want.z_orig) and want.fold_change >= 1
    assert not _differences(got, want)
    assert (got.screen, got.n_null_evaluated) == ("none", 0)


def test_block_rows_must_be_positive(case):
    with pytest.raises(ValueError, match="block_rows"):
        run_low_level_test_screened(
            **_inputs(case, 0, 0), synthetic_idxs=_flat(case, 0, 32), block_rows=0
        )
    with pytest.raises(ValueError, match="block_rows"):
        FlatDraws(case["draws"][0], N_CELLS).prebuild_blocks(B1, 0)


def test_both_samplers_are_exercised(case):
    shares = np.array(TARGET_SIZES) / N_CELLS
    assert (shares > EXACT_SAMPLER_ABOVE_SHARE).any() and (
        shares <= EXACT_SAMPLER_ABOVE_SHARE
    ).any()
    assert all(len(d) == B1 + B2 for d in case["draws"])
