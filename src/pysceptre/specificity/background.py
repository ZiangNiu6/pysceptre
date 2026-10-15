"""The matched background rate and the broad-effect flag (parts 1 and 2 of the check).

A test between an element and a gene on another chromosome cannot be direct
regulation, so how often those tests are called at the cis cutoff, with a
negative effect, is how often a test is called without it. Every function here
takes result tables and returns tables; none needs the count matrix. See
`docs/design.md`, "Specificity check".
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from scipy import sparse, stats

__all__ = [
    "DEFAULT_DISTANCE_BINS",
    "above_background_by_distance",
    "background_pairs",
    "broad_effect_elements",
    "cis_links",
    "nearest_tss",
    "per_gene_background",
]

DEFAULT_DISTANCE_BINS: tuple[float, ...] = (0.0, 1e3, 1e4, 5e4, 1e5, np.inf)
_RESULT_COLUMNS = ("grna_target", "response_id", "p_value", "log_2_fold_change")
_ID_COLUMNS = ("grna_target", "response_id")
# A screened trans result's own columns, carried through when present.
_SCREEN_COLUMNS = ("screen", "p_value_floor")
# `screen` values of a pair that was tested but not resampled: no p-value.
_SCREENED = ("fold_change", "early_stop")
# Draws times pairs held at once by the bootstrap; bounds memory, not the answer.
_BOOTSTRAP_BLOCK = 1 << 22
# An element x gene matrix denser than this is multiplied as a dense array, with the
# same bits; see docs/design.md, "The background in one pass".
_DENSE_ABOVE = 0.5


def _require(frame: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise KeyError(f"{name} is missing column(s): {absent}")


def tested_pairs(result: pd.DataFrame, name: str) -> pd.DataFrame:
    """The rows of a discovery result that were tested, in their order.

    A row was tested when its p-value is finite, or when a screened result
    marks it `screen` "fold_change" or "early_stop": tested, not
    resampled, so its p-value is NaN. With a `pass_qc` column, that must be
    true as well.

    Returns:
        `grna_target` and `response_id` as object dtype, so categorical
        input gives the same frame as object input; `p_value`;
        `log_2_fold_change`, which is `np.log2(fold_change)` when the result
        has only `fold_change`, with a fold change at or below 0 (the dose
        test can report one) as `-inf` and NaN kept NaN; then `screen` and
        `p_value_floor`, each only when the result has it.

    Raises:
        KeyError: a column is missing, or both `log_2_fold_change` and
            `fold_change` are.
    """
    return _tested(result, name)


def _tested(
    result: pd.DataFrame, name: str, targets: Collection[str] | None = None
) -> pd.DataFrame:
    """`tested_pairs`, restricted to the rows of `targets` when given."""
    absent = [c for c in _RESULT_COLUMNS[:3] if c not in result.columns]
    derive = "log_2_fold_change" not in result.columns
    hint = ""
    if derive and "fold_change" not in result.columns:
        absent.append("log_2_fold_change")
        hint = " (fold_change can stand in for log_2_fold_change)"
    if absent:
        raise KeyError(f"{name} is missing column(s): {absent}{hint}")
    keep = np.isfinite(result["p_value"].to_numpy(dtype=float))
    if "screen" in result.columns:
        keep |= result["screen"].isin(_SCREENED).to_numpy()
    if "pass_qc" in result.columns:
        keep &= result["pass_qc"].astype(bool).to_numpy()
    if targets is not None:
        keep &= result["grna_target"].isin(targets).to_numpy()

    # Column by column, so nothing but the kept rows is copied.
    def kept(column: str) -> pd.Series:
        values = result[column][keep]
        values.index = pd.RangeIndex(len(values))
        return values

    out = {c: kept(c) for c in _RESULT_COLUMNS[:3]}
    for c in _ID_COLUMNS:
        if out[c].dtype != object:
            out[c] = out[c].astype(object)
    if derive:
        fc = result["fold_change"].to_numpy(dtype=float)[keep]
        # At or below 0 is -inf, so the direction of a dose-test fold change is kept.
        with np.errstate(divide="ignore"):
            lfc = np.log2(np.where(fc <= 0, 0.0, fc))
        out["log_2_fold_change"] = pd.Series(lfc)
    else:
        out["log_2_fold_change"] = kept("log_2_fold_change")
    for c in _SCREEN_COLUMNS:
        if c in result.columns:
            out[c] = kept(c)
    return pd.DataFrame(out, copy=False)


def called_down(result: pd.DataFrame, cutoff: float) -> np.ndarray:
    """Called at `cutoff` with a negative effect: a link on the cis side, a background call on the other."""
    return (result["p_value"].to_numpy() <= cutoff) & (result["log_2_fold_change"].to_numpy() < 0)


def _refuse_unsafe_screens(
    pairs: pd.DataFrame, name: str, cutoff: float, *, nominal: float | None = None
) -> None:
    """Raise when a screened row of `pairs` could be a call the caller counts.

    A screened row has no p-value, so it counts as not called. That is exact
    when its effect is not negative, or when its `p_value_floor` (a lower
    bound on its p-value) is above `cutoff` and, with `nominal`, at least
    `nominal`. See docs/design.md, "The trans background".
    """
    if "screen" not in pairs.columns or pairs.empty:
        return
    down = pairs["log_2_fold_change"].to_numpy(dtype=float) < 0
    fc = down & (pairs["screen"] == "fold_change").to_numpy()
    if fc.any():
        raise ValueError(
            f"{name} has {int(fc.sum())} row(s) with screen 'fold_change' and a negative "
            "log_2_fold_change, including (grna_target, response_id) "
            f"{_first_pairs(pairs, fc)}. That screen skips only pairs with a fold change of "
            "at least 1, so these rows are inconsistent."
        )
    early = down & (pairs["screen"] == "early_stop").to_numpy()
    if not early.any():
        return
    if "p_value_floor" in pairs.columns:
        floor = pairs["p_value_floor"].to_numpy(dtype=float)
    else:
        floor = np.full(len(pairs), np.nan)
    safe = floor > cutoff
    need = f"above the cutoff {cutoff!r}"
    if nominal is not None:
        safe &= floor >= nominal
        need += f" and at least the nominal level {nominal!r}"
    bad = early & ~safe
    if bad.any():
        raise ValueError(
            f"{name} has {int(bad.sum())} row(s) with screen 'early_stop' and a negative "
            f"log_2_fold_change whose p_value_floor is missing or not {need}, including "
            f"(grna_target, response_id) {_first_pairs(pairs, bad)}. Their p-values could be "
            "calls at that level: test those pairs without the early stop, or call below "
            "their floor."
        )


def _first_pairs(pairs: pd.DataFrame, rows: np.ndarray, n: int = 5) -> list[tuple]:
    first = np.flatnonzero(rows)[:n]
    return list(
        zip(
            pairs["grna_target"].to_numpy()[first],
            pairs["response_id"].to_numpy()[first],
            strict=True,
        )
    )


def element_table(element_positions: pd.DataFrame) -> pd.DataFrame:
    """`element_positions` indexed by `grna_target`, validated."""
    _require(element_positions, "element_positions", ("grna_target", "chrom", "centre"))
    table = element_positions.loc[:, ["grna_target", "chrom", "centre"]]
    dup = table["grna_target"].duplicated()
    if dup.any():
        raise ValueError(
            f"element_positions lists {int(dup.sum())} target(s) twice, including: "
            f"{table.loc[dup, 'grna_target'].tolist()[:5]}"
        )
    return table.set_index("grna_target")


def gene_table(gene_positions: pd.DataFrame, genes: Collection[str]) -> pd.DataFrame:
    """One position per gene in `genes`, indexed by `response_id`.

    `gene_positions` may be a whole annotation. Only the genes asked for have
    to be present and unique there: a symbol on two chromosomes makes its
    distances ambiguous, so it is refused rather than resolved.
    """
    _require(gene_positions, "gene_positions", ("response_id", "chrom", "tss"))
    wanted = set(genes)
    table = gene_positions.loc[
        gene_positions["response_id"].isin(wanted), ["response_id", "chrom", "tss"]
    ]
    dup = table["response_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"gene_positions has more than one position for {table.loc[dup, 'response_id'].nunique()} "
            f"gene(s), including: {sorted(set(table.loc[dup, 'response_id']))[:5]}"
        )
    missing = sorted(wanted - set(table["response_id"]))
    if missing:
        raise KeyError(f"gene_positions has no entry for {len(missing)} gene(s): {missing[:5]}")
    return table.set_index("response_id")


def cis_links(
    cis_result: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    cutoff: float,
    *,
    control_targets: Collection[str] = (),
) -> pd.DataFrame:
    """The tested element-gene pairs, with their distance and whether each is a link.

    Args:
        cis_result: a discovery result (`grna_target`, `response_id`,
            `p_value`, `log_2_fold_change` or `fold_change`, optionally
            `pass_qc`) on pairs with the element and the gene on the same
            chromosome. May be a screened result (`screen`,
            `p_value_floor`): its screened rows are tests that are not links.
        element_positions: `grna_target`, `chrom`, `centre`.
        gene_positions: `response_id`, `chrom`, `tss`.
        cutoff: the nominal p-value threshold the screen was called at.
        control_targets: targets that are not elements, such as TSS
            positive controls. Their pairs are left out.

    Returns:
        `grna_target`, `response_id`, `p_value`, `log_2_fold_change`, then
        `screen` and `p_value_floor` when `cis_result` has them, `distance`
        (element centre to TSS) and `link` (called with a negative effect),
        one row per tested pair.

    Raises:
        KeyError: a target or gene has no position.
        ValueError: a pair has its element and gene on different chromosomes,
            or a screened row of an element with a negative effect could be a
            link at `cutoff`: screened on its fold change, or stopped early
            with a `p_value_floor` that is missing or not above `cutoff`.
    """
    return _cis_links(
        tested_pairs(cis_result, "cis_result"),
        element_positions,
        gene_positions,
        cutoff,
        control_targets=control_targets,
    )


def _cis_links(
    pairs: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    cutoff: float,
    *,
    control_targets: Collection[str] = (),
) -> pd.DataFrame:
    """`cis_links` from `tested_pairs(cis_result)`."""
    pairs = pairs.loc[~pairs["grna_target"].isin(set(control_targets))].reset_index(drop=True)
    _refuse_unsafe_screens(pairs, "cis_result", cutoff)
    elements = element_table(element_positions)
    missing = sorted(set(pairs["grna_target"]) - set(elements.index))
    if missing:
        raise KeyError(
            f"element_positions has no entry for {len(missing)} target(s): {missing[:5]}"
        )
    genes = gene_table(gene_positions, pairs["response_id"].unique())

    e_chrom = elements["chrom"].reindex(pairs["grna_target"]).to_numpy()
    g_chrom = genes["chrom"].reindex(pairs["response_id"]).to_numpy()
    apart = e_chrom != g_chrom
    if apart.any():
        first = np.flatnonzero(apart)[:10]
        listed = list(
            zip(
                pairs["grna_target"].to_numpy()[first],
                pairs["response_id"].to_numpy()[first],
                e_chrom[first],
                g_chrom[first],
                strict=True,
            )
        )
        raise ValueError(
            f"{int(apart.sum())} cis_result pair(s), of "
            f"{pairs.loc[apart, 'grna_target'].nunique()} target(s), put the element and the "
            "gene on different chromosomes, so they have no distance. Pass those as "
            "trans_result, or drop them. The first (grna_target, response_id, element chrom, "
            f"gene chrom): {listed}"
        )
    centre = elements["centre"].reindex(pairs["grna_target"]).to_numpy(dtype=float)
    tss = genes["tss"].reindex(pairs["response_id"]).to_numpy(dtype=float)
    return pairs.assign(distance=np.abs(centre - tss), link=called_down(pairs, cutoff))


def background_pairs(
    trans_result: pd.DataFrame,
    links: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    cutoff: float,
    *,
    exclude_gene_chroms: Collection[str] = ("chrY",),
) -> pd.DataFrame:
    """The tests the background is measured on, and which of them are called.

    The cis elements against the cis genes, gene on another chromosome, gene
    chromosome not in `exclude_gene_chroms`, tested. Restricting to the cis
    sets is what lets the two sides be matched gene for gene. `trans_result`
    may be a screened trans result (`run_trans_analysis`): its screened rows
    are tests that are not calls.

    Returns:
        `grna_target`, `response_id`, `p_value`, `log_2_fold_change`, then
        `screen` and `p_value_floor` when `trans_result` has them, and
        `background` (called at the cis `cutoff` with a negative effect).

    Raises:
        ValueError: a screened row with a negative effect could be a call at
            `cutoff`: screened on its fold change, or stopped early with a
            `p_value_floor` that is missing or not above `cutoff`.
    """
    return _background(
        tested_pairs(trans_result, "trans_result"),
        links,
        element_positions,
        gene_positions,
        cutoff,
        exclude_gene_chroms=exclude_gene_chroms,
    )


def _background(
    pairs: pd.DataFrame,
    links: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    cutoff: float,
    *,
    exclude_gene_chroms: Collection[str] = ("chrY",),
) -> pd.DataFrame:
    """`background_pairs` from `tested_pairs(trans_result)`.

    Consumes `pairs`: when every row is kept, the result is `pairs` itself
    with the `background` column added, so a caller must not use it again.
    """
    elements = element_table(element_positions)
    targets = pd.Index(pd.unique(links["grna_target"]))
    genes = pd.Index(pd.unique(links["response_id"]))
    # Positions in the link sets, -1 outside them; one hash lookup per row and column.
    ti = targets.get_indexer(pairs["grna_target"])
    gi = genes.get_indexer(pairs["response_id"])
    keep = (ti >= 0) & (gi >= 0)
    present = genes[np.bincount(gi[keep], minlength=len(genes)) > 0]
    positions = gene_table(gene_positions, present)
    if keep.any():
        e_chrom = elements["chrom"].reindex(targets).to_numpy()
        g_chrom = positions["chrom"].reindex(genes).to_numpy()
        # Chromosomes as shared codes; a missing one never equals anything, as with `!=`.
        codes = pd.factorize(np.concatenate([e_chrom, g_chrom]))[0]
        e_code, g_code = codes[: len(targets)], codes[len(targets) :]
        g_code[g_code < 0] = -2
        # Via pandas: a gene with no position is pd.NA in a nullable dtype, which np.isin
        # cannot compare.
        excluded = (
            positions["chrom"]
            .isin(list(exclude_gene_chroms))
            .reindex(genes, fill_value=False)
            .to_numpy(dtype=bool)
        )
        keep &= (e_code[ti] != g_code[gi]) & ~excluded[gi]
    if keep.all():
        out = pairs
    else:
        out = pairs.take(np.flatnonzero(keep))
        out.index = pd.RangeIndex(len(out))
    _refuse_unsafe_screens(out, "trans_result", cutoff)
    out["background"] = called_down(out, cutoff)
    return out


def per_gene_background(
    background: pd.DataFrame, *, genes: Collection[str] | None = None
) -> pd.DataFrame:
    """`response_id`, `trans_pairs`, `background_calls`, `background_rate`.

    With `genes`, one row per gene in that order; a gene with no background
    test has zero pairs and calls and a NaN rate.
    """
    out = background.groupby("response_id")["background"].agg(
        trans_pairs="size", background_calls="sum", background_rate="mean"
    )
    if genes is not None:
        out = out.reindex(list(genes)).rename_axis("response_id")
        out["trans_pairs"] = out["trans_pairs"].fillna(0).astype(int)
        out["background_calls"] = out["background_calls"].fillna(0).astype(int)
    return out.reset_index()


def _bin_labels(edges: Sequence[float]) -> list[str]:
    def kb(x: float) -> str:
        return f"{x / 1000:g}"

    return [
        f"> {kb(a)} kb" if np.isinf(b) else f"{kb(a)} to {kb(b)} kb"
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def above_background_by_distance(
    links: pd.DataFrame,
    background: pd.DataFrame,
    *,
    distance_bins: Sequence[float] = DEFAULT_DISTANCE_BINS,
    n_bootstrap: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per distance bin: cis rate, matched background and links above background.

    The background of a bin is the mean, over its tested pairs, of each
    pair's gene's background rate, so a gene counts as often as it is tested
    there. Links above background is `(cis rate - background) / cis rate`.
    Intervals are a bootstrap over **elements**: an element's cis and
    background tests are drawn together, and every gene's rate is recomputed
    in each draw.

    Args:
        links: `cis_links` output.
        background: `background_pairs` output for the same elements.
        distance_bins: bin edges in bp, increasing. Bins are closed on the
            right, and the first also includes its left edge.
        n_bootstrap: bootstrap draws.
        seed: an int, or a Generator to continue drawing from.

    Returns:
        One row per bin: `distance_bin`, `pairs`, `links`, then `cis_rate`,
        `background` and `above`, each with `_lo` and `_hi` (the 2.5 and 97.5
        percentiles of the draws).

    Raises:
        ValueError: a distance falls outside the bins, or the bins do not
            increase.
    """
    edges = np.asarray(distance_bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"distance_bins must be at least two increasing edges, got {distance_bins!r}"
        )
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap!r}")
    rng = np.random.default_rng(seed)
    n_bins = edges.size - 1

    elems = np.sort(links["grna_target"].unique())
    genes = np.sort(links["response_id"].unique())
    e_index, g_index = pd.Index(elems), pd.Index(genes)
    n_e, n_g = elems.size, genes.size

    # Removing an element from `links` must remove it from the background too; see
    # docs/design.md, "Broad-effect elements".
    te = e_index.get_indexer(background["grna_target"])
    tg = g_index.get_indexer(background["response_id"])
    inside = (te >= 0) & (tg >= 0)
    te, tg = te[inside], tg[inside]
    called = background["background"].to_numpy(dtype=float)[inside]
    tested = _incidence(te, tg, np.ones(te.size), (n_e, n_g))
    # Zero terms add nothing to these integer sums, so only the calls are stored.
    nz = called != 0
    calls = _incidence(te[nz], tg[nz], called[nz], (n_e, n_g))
    del te, tg, called, nz

    ce = e_index.get_indexer(links["grna_target"])
    cg = g_index.get_indexer(links["response_id"])
    cb = pd.cut(links["distance"], edges, include_lowest=True, labels=False).to_numpy()
    if np.isnan(cb).any():
        raise ValueError(
            f"{int(np.isnan(cb).sum())} pair(s) have a distance outside distance_bins "
            f"[{edges[0]}, {edges[-1]}]"
        )
    cb = cb.astype(int)
    n_pairs = len(links)
    bin_of_pair = sparse.csr_matrix(
        (np.ones(n_pairs), (np.arange(n_pairs), cb)), shape=(n_pairs, n_bins)
    )
    pairs_eb = np.zeros((n_e, n_bins))
    links_eb = np.zeros((n_e, n_bins))
    np.add.at(pairs_eb, (ce, cb), 1.0)
    np.add.at(links_eb, (ce, cb), links["link"].to_numpy(dtype=float))

    def estimate(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rate = np.nan_to_num(_weighted_sums(calls, w) / _weighted_sums(tested, w))
        n = w @ pairs_eb
        cis = (w @ links_eb) / n
        bg_sum = np.empty((w.shape[0], n_bins))
        step = max(1, _BOOTSTRAP_BLOCK // max(n_pairs, 1))
        for s in range(0, w.shape[0], step):
            block = w[s : s + step, ce] * rate[s : s + step, cg]
            bg_sum[s : s + step] = (bin_of_pair.T @ block.T).T
        back = bg_sum / n
        return cis, back, (cis - back) / cis

    with np.errstate(divide="ignore", invalid="ignore"):
        point = estimate(np.ones((1, n_e)))
        w = rng.multinomial(n_e, np.full(n_e, 1.0 / n_e), n_bootstrap).astype(float)
        draws = estimate(w)

    out = pd.DataFrame(
        {
            "distance_bin": _bin_labels(edges),
            "pairs": pairs_eb.sum(0).astype(int),
            "links": links_eb.sum(0).astype(int),
        }
    )
    with warnings.catch_warnings():
        # A bin no draw can estimate (no pairs) is NaN, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, p, d in zip(("cis_rate", "background", "above"), point, draws, strict=True):
            out[name] = p[0]
            out[f"{name}_lo"], out[f"{name}_hi"] = np.nanpercentile(d, [2.5, 97.5], axis=0)
    return out


def _incidence(
    rows: np.ndarray, cols: np.ndarray, values: np.ndarray, shape: tuple[int, int]
) -> sparse.csc_matrix | np.ndarray:
    """`values` summed into an elements x genes matrix, dense when more than half is set."""
    m = sparse.csc_matrix((values, (rows, cols)), shape=shape)
    if m.nnz > _DENSE_ABOVE * shape[0] * shape[1]:
        return m.toarray()
    return m


def _weighted_sums(m: sparse.csc_matrix | np.ndarray, w: np.ndarray) -> np.ndarray:
    """`w @ m`, draws x genes.

    Every product and partial sum is an integer below 2**53 when `m` counts
    tests or calls, so the dense and the sparse path give the same bits.
    """
    if isinstance(m, np.ndarray):
        return w @ m
    return (m.T @ w.T).T


def nearest_tss(
    chrom: pd.Series, position: pd.Series, gene_positions: pd.DataFrame
) -> pd.DataFrame:
    """The nearest TSS in `gene_positions` on the same chromosome, for each position.

    Returns `nearest_gene` and `tss_distance`, indexed like `chrom`; NaN on a
    chromosome with no gene. A tie goes to the upstream TSS.
    """
    _require(gene_positions, "gene_positions", ("response_id", "chrom", "tss"))
    out = pd.DataFrame(
        {"nearest_gene": pd.Series(np.nan, index=chrom.index, dtype=object), "tss_distance": np.nan}
    )
    for c, idx in chrom.groupby(chrom).groups.items():
        g = gene_positions.loc[gene_positions["chrom"] == c].sort_values("tss", kind="mergesort")
        if g.empty:
            continue
        t = g["tss"].to_numpy(dtype=float)
        p = position.loc[idx].to_numpy(dtype=float)
        if t.size == 1:
            j = np.zeros(p.size, dtype=int)
        else:
            i = np.clip(np.searchsorted(t, p), 1, t.size - 1)
            j = np.where(np.abs(t[i - 1] - p) <= np.abs(t[i] - p), i - 1, i)
        out.loc[idx, "nearest_gene"] = g["response_id"].to_numpy()[j]
        out.loc[idx, "tss_distance"] = np.abs(t[j] - p)
    return out


def broad_effect_elements(
    background: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    *,
    p_threshold: float = 1e-3,
) -> pd.DataFrame:
    """Elements with more background calls than the screen's background rate allows.

    Each element's background calls are tested against the pooled rate over
    all background tests (one-sided binomial). Such an element makes its own
    background, which a per-gene rate cannot see.

    Returns:
        `grna_target`, `trans_pairs`, `trans_calls`, `p_excess`,
        `broad_effect` (`p_excess < p_threshold`), `nearest_gene` and
        `tss_distance` (the nearest TSS in `gene_positions`), one row per
        element with background tests.
    """
    if not 0.0 < p_threshold < 1.0:
        raise ValueError(f"p_threshold must lie in (0, 1), got {p_threshold!r}")
    per = (
        background.groupby("grna_target", sort=True)["background"]
        .agg(trans_calls="sum", trans_pairs="size")
        .reset_index()
    )
    pooled = float(background["background"].mean())
    per["p_excess"] = stats.binom.sf(per["trans_calls"] - 1, per["trans_pairs"], pooled)
    per["broad_effect"] = per["p_excess"] < p_threshold
    elements = element_table(element_positions)
    near = nearest_tss(
        elements["chrom"].reindex(per["grna_target"]).reset_index(drop=True),
        elements["centre"].reindex(per["grna_target"]).reset_index(drop=True),
        gene_positions,
    )
    per = per.join(near)
    return per[
        [
            "grna_target",
            "trans_pairs",
            "trans_calls",
            "p_excess",
            "broad_effect",
            "nearest_gene",
            "tss_distance",
        ]
    ]
