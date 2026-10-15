"""Port of `run_low_level_test_full_v4` (low_level_full_test.cpp): the staged
B1 -> B2 -> B3 escalation that decides, per pair, whether the empirical
p-value from the first B1=499 draws is already trustworthy, or whether a
skew-normal tail extrapolation (or a further empirical batch) is needed.

Verified against the pinned upstream commit's source directly; this is a
straight port of ~35 lines of control flow, not a reconstruction.

`run_low_level_test_screened` is the same test for the trans background, with
two exact screens that skip resampling no call of the specificity check can
see. It sits beside `run_low_level_test_full`, which it leaves unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .empirical_p import compute_empirical_p_value, empirical_p_from_counts
from .fold_change import estimate_log_fold_change
from .score_stat import (
    as_staged_draws,
    compute_null_statistics_block,
    compute_null_statistics_from_draws,
    compute_observed_full_statistic,
    stack_pieces,
)
from .skew_normal import fit_and_evaluate_skew_normal

P_THRESH = 0.02


@dataclass
class PairResult:
    p_value: float
    z_orig: float
    fold_change: float
    se_fold_change: float
    stage: int | float  # NaN when nothing was tested
    sn_params: (
        tuple[float, float, float] | None
    )  # (xi, omega, alpha), None if SN was never fit/used
    resampling_dist: np.ndarray | None = None


def run_low_level_test_full(
    y: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    D: np.ndarray,
    trt_idxs: np.ndarray,
    synthetic_idxs,
    *,
    stacked: np.ndarray | None = None,
    B1: int = 499,
    B2: int = 4999,
    B3: int = 0,
    fit_parametric_curve: bool = True,
    side_code: int = 0,
    p_thresh: float = P_THRESH,
    return_resampling_dist: bool = False,
    null_statistics_fn: Callable[[int, int], np.ndarray | None] | None = None,
    trt_weights: np.ndarray | None = None,
) -> PairResult:
    """trt_idxs: 0-based observed treated-cell indices. With none, or with a
        statistic that is not finite, the p-value is NaN rather than a test.

    synthetic_idxs: the B1+B2+B3 draws, consumed in three consecutive slices
        [0:B1], [B1:B1+B2], [B1+B2:B1+B2+B3]. Normally a `StagedDraws`, which
        materializes each slice only when that stage is reached and memoizes
        it, so a target's draws are built once per stage rather than once per
        pair -- and the stages a run never reaches are never built at all.
        A CSR matrix or a list of ragged index arrays is also accepted and
        wrapped, since those are the natural things to write by hand in a
        test.
    stacked: `stack_pieces(a, w, D)`, built once per gene and reused across
        that gene's targets. Computed here when omitted.
    null_statistics_fn: an optional faster route to a stage's null
        statistics, taking `(lo, hi)`. Permutations pass
        `PermutationPrefixSums.statistics`, which reads a column out of one
        per-gene prefix-sum array instead of running a per-target matmul --
        possible only because every target shares the permutation rows.
        Returning `None` for a stage means "no cheaper route here", and the
        draw matrix is used for it, so a provider can decline the stages
        whose arrays would be too large.
    """
    draws = as_staged_draws(synthetic_idxs, a.shape[0])
    if stacked is None:
        stacked = stack_pieces(a, w, D)

    def nulls(lo: int, hi: int) -> np.ndarray:
        if null_statistics_fn is not None:
            got = null_statistics_fn(lo, hi)
            if got is not None:
                return got
        return compute_null_statistics_from_draws(stacked, draws.slice(lo, hi))

    if len(trt_idxs) == 0:
        # Nothing treated, nothing to test; sceptre's pairwise QC never lets such a pair in.
        return PairResult(np.nan, np.nan, np.nan, np.nan, np.nan, None)
    fc, se = estimate_log_fold_change(y, mu, trt_idxs, trt_weights)
    z_orig = compute_observed_full_statistic(a, w, D, trt_idxs, trt_weights)
    if not np.isfinite(z_orig):
        # A degenerate statistic exceeds no null value, which would read as p = 1/(B+1).
        return PairResult(np.nan, z_orig, fc, se, np.nan, None)

    sn_params: tuple[float, float, float] | None = None
    stage = 1
    null_statistics = nulls(0, B1)
    p = compute_empirical_p_value(null_statistics, z_orig, side_code)

    if p <= p_thresh:
        sn_fit_used = False
        if fit_parametric_curve:
            null_statistics = nulls(B1, B1 + B2)
            sn_result = fit_and_evaluate_skew_normal(z_orig, null_statistics, side_code)
            p = sn_result.p
            sn_fit_used = sn_result.used
            if sn_fit_used:
                sn_params = (sn_result.xi, sn_result.omega, sn_result.alpha)
            stage = 2

        if not fit_parametric_curve or not sn_fit_used:
            if B3 > 0:
                null_statistics = nulls(B1 + B2, B1 + B2 + B3)
            p = compute_empirical_p_value(null_statistics, z_orig, side_code)
            stage = 3

    return PairResult(
        p_value=p,
        z_orig=z_orig,
        fold_change=fc,
        se_fold_change=se,
        stage=stage,
        sn_params=sn_params,
        resampling_dist=null_statistics if return_resampling_dist else None,
    )


@dataclass
class ScreenedPairResult:
    """A pair's result from `run_low_level_test_screened`.

    `screen` says what was skipped: "none" for a pair tested as
    `run_low_level_test_full` tests it, whose other fields then equal that
    function's; "fold_change" for a pair not resampled because its fold
    change is at least 1; "early_stop" for a pair whose first stage stopped
    once its p-value could no longer fall below the stop level. A screened
    pair has `p_value` NaN, and keeps `z_orig`, `fold_change` and
    `se_fold_change`. `p_value_floor` is, for an early stop, a lower bound on
    the p-value the full test gives the pair, and NaN otherwise.
    `n_null_evaluated` counts the null statistics computed, over all stages.
    """

    p_value: float
    z_orig: float
    fold_change: float
    se_fold_change: float
    stage: int | float  # NaN when nothing was resampled
    screen: str  # "none", "fold_change" or "early_stop"
    p_value_floor: float
    n_null_evaluated: int
    sn_params: tuple[float, float, float] | None = None


def run_low_level_test_screened(
    y: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    D: np.ndarray,
    trt_idxs: np.ndarray,
    synthetic_idxs,
    *,
    stacked: np.ndarray | None = None,
    B1: int = 499,
    B2: int = 4999,
    B3: int = 0,
    fit_parametric_curve: bool = True,
    side_code: int = 0,
    p_thresh: float = P_THRESH,
    screen_fold_change: bool = True,
    p_exact_below: float | None = 0.05,
    block_rows: int = 32,
) -> ScreenedPairResult:
    """`run_low_level_test_full` with two exact screens, for the trans background.

    Takes the arguments `run_low_level_test_full` takes, without dose weights
    or a null-statistics provider. `z_orig`, `fold_change` and
    `se_fold_change` are computed exactly as there, and a pair with no
    treated cells or a non-finite statistic gets that function's result.

    screen_fold_change: a pair with `fold_change >= 1` is not resampled.
    p_exact_below: the first stage's `B1` null statistics are computed in row
        blocks of `block_rows`, and after each block but the last the
        p-value those counts imply with the full `B1` denominator
        (`empirical_p_from_counts`) is a lower bound on the stage's p-value.
        Once that bound exceeds `p_thresh`, so the pair cannot escalate, and
        reaches `p_exact_below`, the test stops. None disables the stop.
    block_rows: rows per block. A `FlatDraws` whose `prebuild_blocks` was
        called with the same `B1` and `block_rows` serves the blocks without
        rebuilding them.

    A pair that neither screen stops is tested as `run_low_level_test_full`
    tests it: the blocks' statistics are, bit for bit, the first stage's, and
    stages 2 and 3 are that function's own calls. With `screen_fold_change`
    False and `p_exact_below` None every field equals that function's. See
    docs/design.md, "The trans background".
    """
    block_rows = int(block_rows)
    if block_rows < 1:
        raise ValueError(f"block_rows must be positive, got {block_rows!r}")
    draws = as_staged_draws(synthetic_idxs, a.shape[0])
    if stacked is None:
        stacked = stack_pieces(a, w, D)

    def nulls(lo: int, hi: int) -> np.ndarray:
        return compute_null_statistics_from_draws(stacked, draws.slice(lo, hi))

    if len(trt_idxs) == 0:
        return ScreenedPairResult(np.nan, np.nan, np.nan, np.nan, np.nan, "none", np.nan, 0)
    fc, se = estimate_log_fold_change(y, mu, trt_idxs)
    z_orig = compute_observed_full_statistic(a, w, D, trt_idxs)
    if not np.isfinite(z_orig):
        return ScreenedPairResult(np.nan, z_orig, fc, se, np.nan, "none", np.nan, 0)
    if screen_fold_change and fc >= 1.0:
        return ScreenedPairResult(np.nan, z_orig, fc, se, np.nan, "fold_change", np.nan, 0)

    blocks: list[np.ndarray] = []
    c_left = c_right = n_null = 0
    for lo in range(0, B1, block_rows):
        hi = min(lo + block_rows, B1)
        block = compute_null_statistics_block(stacked, draws.slice(lo, hi))
        blocks.append(block)
        n_null += block.size
        c_left += int(np.count_nonzero(z_orig >= block))
        c_right += int(np.count_nonzero(z_orig <= block))
        if p_exact_below is not None and hi < B1:
            floor = empirical_p_from_counts(c_left, c_right, B1, side_code)
            if floor > p_thresh and floor >= p_exact_below:
                return ScreenedPairResult(np.nan, z_orig, fc, se, 1, "early_stop", floor, n_null)

    # From here on, `run_low_level_test_full` verbatim.
    sn_params: tuple[float, float, float] | None = None
    stage = 1
    null_statistics = np.concatenate(blocks) if blocks else np.empty(0)
    p = compute_empirical_p_value(null_statistics, z_orig, side_code)

    if p <= p_thresh:
        sn_fit_used = False
        if fit_parametric_curve:
            null_statistics = nulls(B1, B1 + B2)
            n_null += null_statistics.size
            sn_result = fit_and_evaluate_skew_normal(z_orig, null_statistics, side_code)
            p = sn_result.p
            sn_fit_used = sn_result.used
            if sn_fit_used:
                sn_params = (sn_result.xi, sn_result.omega, sn_result.alpha)
            stage = 2

        if not fit_parametric_curve or not sn_fit_used:
            if B3 > 0:
                null_statistics = nulls(B1 + B2, B1 + B2 + B3)
                n_null += null_statistics.size
            p = compute_empirical_p_value(null_statistics, z_orig, side_code)
            stage = 3

    return ScreenedPairResult(
        p_value=p,
        z_orig=z_orig,
        fold_change=fc,
        se_fold_change=se,
        stage=stage,
        screen="none",
        p_value_floor=np.nan,
        n_null_evaluated=n_null,
        sn_params=sn_params,
    )
