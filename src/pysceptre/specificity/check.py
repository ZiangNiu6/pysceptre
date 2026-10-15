"""`run_specificity_check`: how many discovered links are more than background."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .background import (
    DEFAULT_DISTANCE_BINS,
    _background,
    _cis_links,
    above_background_by_distance,
    broad_effect_elements,
    per_gene_background,
    tested_pairs,
)
from .lookup import _gene_lookup, detour_check, tss_targets

__all__ = ["SpecificityResult", "run_specificity_check"]


@dataclass(frozen=True)
class SpecificityResult:
    """Everything `run_specificity_check` computes, as tables.

    Attributes:
        by_distance: per distance bin, the cis rate, the matched background
            and links above background, with bootstrap intervals.
        by_distance_without_broad: the same with the broad-effect elements
            removed from both the cis and the background tests.
        broad_effect: per element, its background calls and the binomial
            test against the screen's background rate.
        per_gene: per gene, its background tests, calls and rate.
        links: the tested element-gene pairs, with distance and link call.
        background: the tests the background is measured on.
        tss_targets: targets within the TSS window of a measured gene.
        gene_lookup: what each TSS knockdown lowers.
        far_links: the detour check of every far link.
        background_rate: the pooled background rate.
    """

    by_distance: pd.DataFrame
    by_distance_without_broad: pd.DataFrame
    broad_effect: pd.DataFrame
    per_gene: pd.DataFrame
    links: pd.DataFrame
    background: pd.DataFrame
    tss_targets: pd.DataFrame
    gene_lookup: pd.DataFrame
    far_links: pd.DataFrame
    background_rate: float


def run_specificity_check(
    cis_result: pd.DataFrame,
    trans_result: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    cutoff: float,
    *,
    control_targets: Collection[str] = (),
    measured_genes: Collection[str] | None = None,
    distance_bins: Sequence[float] = DEFAULT_DISTANCE_BINS,
    far_distance: float = 1e5,
    tss_window: float = 1_000.0,
    broad_effect_p: float = 1e-3,
    nominal: float = 0.05,
    exclude_gene_chroms: Collection[str] = ("chrY",),
    n_bootstrap: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> SpecificityResult:
    """Which discovered links are more than background.

    The calibration check asks whether the pipeline invents effects and the
    power check whether it recovers effects it should. This asks how many of
    the links it did discover are more than the rate at which a test is
    called without regulation, measured on tests between elements and genes
    on other chromosomes. It is not sceptre's and has no R counterpart; see
    `docs/design.md`, "Specificity check", for its scope and evidence.

    Args:
        cis_result: the screen's discovery result on element-gene pairs on
            the same chromosome: `grna_target`, `response_id`, `p_value`,
            `log_2_fold_change` (or `fold_change`, from which it is
            derived), and `pass_qc` if it has one. Ids may be categorical.
            A screened result is accepted as for `trans_result`.
        trans_result: a discovery result, same columns and the same test,
            for the screen's elements against its cis genes on other
            chromosomes. Pairs outside those sets are ignored. A screened
            result from `run_trans_analysis` is accepted: a row with
            `screen` "fold_change" or "early_stop" was tested and not
            resampled, so it counts as a test that is not a call.
        element_positions: `grna_target`, `chrom`, `centre`, for every target
            in `cis_result`. Take these from the screen's own coordinate
            table, never from parsing target names.
        gene_positions: `response_id`, `chrom`, `tss`. May be a whole
            annotation; the genes the check uses must each appear once.
        cutoff: the nominal p-value threshold the cis analysis was called at,
            e.g. `bh_nominal_cutoff(p_values, alpha)`.
        control_targets: targets that are not elements (TSS positive
            controls). Left out of the link and background counts, and still
            used as TSS knockdowns.
        measured_genes: the genes the screen measured, for finding TSS
            targets. Defaults to every gene in either result.
        distance_bins: bin edges in bp for the link counts.
        far_distance: links beyond this are the ones the detour check reads.
        tss_window: a target within this many bp of a measured TSS knocks
            that gene down.
        broad_effect_p: binomial threshold for the broad-effect flag.
        nominal: the p-value for the nominal detour level.
        exclude_gene_chroms: gene chromosomes left out of the background.
        n_bootstrap: bootstrap draws over elements.
        seed: an int, or a Generator to continue drawing from. The draws with
            all elements come first, then the draws without the broad-effect
            elements, whether or not any element is flagged.

    Returns:
        A `SpecificityResult`.

    Raises:
        KeyError: a column, a target position or a gene position is missing.
        ValueError: `cutoff` is out of range, a cis pair spans two
            chromosomes, a gene has two positions, or a screened row could
            be a call at `cutoff` (or, for a TSS target, at `nominal`).

    Warns:
        UserWarning: some cis genes have no background test. Their
            background rate counts as 0 in the matched background.
    """
    if not 0.0 < cutoff < 1.0:
        raise ValueError(f"cutoff must lie in (0, 1), got {cutoff!r}")
    rng = np.random.default_rng(seed)

    cis = tested_pairs(cis_result, "cis_result")
    links = _cis_links(
        cis, element_positions, gene_positions, cutoff, control_targets=control_targets
    )
    # The one pass over the whole trans result; see docs/design.md,
    # "The background in one pass".
    trans = tested_pairs(trans_result, "trans_result")
    if measured_genes is None:
        measured_genes = set(pd.unique(cis["response_id"])) | set(pd.unique(trans["response_id"]))
    tss = tss_targets(
        cis["grna_target"].unique(),
        element_positions,
        gene_positions,
        measured_genes,
        window=tss_window,
    )
    trans_tss = trans.loc[trans["grna_target"].isin(set(tss["grna_target"]))]
    background = _background(
        trans,
        links,
        element_positions,
        gene_positions,
        cutoff,
        exclude_gene_chroms=exclude_gene_chroms,
    )
    del trans  # `_background` may have taken it over
    lookup = _gene_lookup(
        tss, (("cis_result", cis), ("trans_result", trans_tss)), cutoff, nominal=nominal
    )
    del trans_tss

    per_gene = per_gene_background(background, genes=np.sort(links["response_id"].unique()))
    alone = per_gene.loc[per_gene["trans_pairs"] == 0, "response_id"]
    if len(alone):
        theirs = links["response_id"].isin(set(alone))
        warnings.warn(
            f"{len(alone)} cis gene(s) have no background test, so their background rate "
            f"counts as 0: {int(theirs.sum())} cis pair(s) and "
            f"{int(links.loc[theirs, 'link'].sum())} link(s), genes including "
            f"{alone.tolist()[:5]}. Drop them from cis_result to leave them out of both sides.",
            UserWarning,
            stacklevel=2,
        )

    by_distance = above_background_by_distance(
        links, background, distance_bins=distance_bins, n_bootstrap=n_bootstrap, seed=rng
    )
    broad = broad_effect_elements(
        background, element_positions, gene_positions, p_threshold=broad_effect_p
    )
    flagged = set(broad.loc[broad["broad_effect"], "grna_target"])
    # The flagged elements' background goes with their links; no filtered copy needed.
    by_distance_without = above_background_by_distance(
        links.loc[~links["grna_target"].isin(flagged)],
        background,
        distance_bins=distance_bins,
        n_bootstrap=n_bootstrap,
        seed=rng,
    )
    far = detour_check(links, tss, lookup, far_distance=far_distance)

    return SpecificityResult(
        by_distance=by_distance,
        by_distance_without_broad=by_distance_without,
        broad_effect=broad,
        per_gene=per_gene,
        links=links,
        background=background,
        tss_targets=tss,
        gene_lookup=lookup,
        far_links=far,
        background_rate=float(background["background"].mean()),
    )
