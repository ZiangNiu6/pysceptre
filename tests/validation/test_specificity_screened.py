"""The specificity check on a screened trans result, and at the scale of one.

A screened trans result (`run_trans_analysis`) has `p_value` NaN on the pairs it
tested without resampling, `screen` saying why and `p_value_floor` bounding the
p-value it skipped. Every table the check returns must be the same as from the
unscreened result, bit for bit. The rest pins what the check gained to read
such a result at ~100M rows: one pass over it, categorical ids, `fold_change`
in place of `log_2_fold_change`, the warning for genes without background
tests, the cross-chromosome message and the dense bootstrap products.
Synthetic, fixed seed; no R.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pysceptre import run_specificity_check
from pysceptre.specificity import (
    above_background_by_distance,
    background,
    background_pairs,
    cis_links,
    gene_lookup,
)

CUTOFF = 0.01
FLOOR = 0.052
N_BOOT = 200
TABLES = (
    "by_distance",
    "by_distance_without_broad",
    "broad_effect",
    "per_gene",
    "links",
    "tss_targets",
    "gene_lookup",
    "far_links",
)
LOUD = "E7"


def _screen(seed: int = 0, n_elements: int = 120, n_genes: int = 48, n_chroms: int = 4):
    """Elements and genes over `n_chroms` chromosomes; cis within one, trans across.

    Trans p-values are `U**2`, so about one background test in twenty is called.
    E0 to E5 sit 200 bp from a TSS on their chromosome, so they knock a gene
    down, and `LOUD` is called against every gene it is tested against.
    """
    rng = np.random.default_rng(seed)
    chroms = [f"chr{k + 1}" for k in range(n_chroms)]
    genes = pd.DataFrame(
        {
            "response_id": [f"G{k}" for k in range(n_genes)],
            "chrom": [chroms[k % n_chroms] for k in range(n_genes)],
            "tss": rng.integers(1_000_000, 2_000_000, n_genes).astype(float),
        }
    )
    elements = pd.DataFrame(
        {
            "grna_target": [f"E{k}" for k in range(n_elements)],
            "chrom": [chroms[k % n_chroms] for k in range(n_elements)],
            "centre": rng.integers(800_000, 2_200_000, n_elements).astype(float),
        }
    )
    elements.loc[:5, "centre"] = genes.loc[:5, "tss"].to_numpy() + 200.0

    cis = elements.merge(genes, on="chrom")[["grna_target", "response_id"]].copy()
    hit = rng.random(len(cis)) < 0.2
    cis["p_value"] = np.where(
        hit, rng.uniform(0, CUTOFF / 2, len(cis)), rng.uniform(CUTOFF, 1, len(cis))
    )
    cis["fold_change"] = 2.0 ** np.where(hit, -0.5, rng.normal(0, 0.2, len(cis)))
    cis["pass_qc"] = rng.random(len(cis)) > 0.02

    trans = elements.merge(genes, how="cross", suffixes=("", "_g"))
    trans = trans.loc[trans["chrom"] != trans["chrom_g"], ["grna_target", "response_id"]].copy()
    trans = trans.reset_index(drop=True)
    trans["p_value"] = rng.uniform(0, 1, len(trans)) ** 2
    trans["fold_change"] = 2.0 ** rng.normal(0, 0.3, len(trans))
    loud = trans["grna_target"] == LOUD
    trans.loc[loud, "p_value"] = 1e-6
    trans.loc[loud, "fold_change"] = 0.5
    trans["pass_qc"] = rng.random(len(trans)) > 0.02
    # Rows the engine could not test: no p-value and no screen.
    trans.loc[rng.choice(len(trans), 15, replace=False), "p_value"] = np.nan
    return cis.reset_index(drop=True), trans, elements, genes


def _with_lfc(result: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore"):
        return result.assign(log_2_fold_change=np.log2(result["fold_change"]))


def _screened(trans: pd.DataFrame) -> pd.DataFrame:
    """What the trans mode reports: fc >= 1 not resampled, the rest stopped early when p >= FLOOR."""
    out = trans.copy()
    p = out["p_value"].to_numpy()
    fc = out["fold_change"].to_numpy() >= 1
    early = ~fc & (p >= FLOOR)
    out["screen"] = np.where(fc, "fold_change", np.where(early, "early_stop", "none")).astype(
        object
    )
    out.loc[np.isnan(p), "screen"] = "none"
    out["p_value_floor"] = np.where(out["screen"] == "early_stop", FLOOR, np.nan)
    out.loc[out["screen"] != "none", "p_value"] = np.nan
    return out


def _check(cis, trans, elements, genes, **kw):
    kw = {"cutoff": CUTOFF, "n_bootstrap": N_BOOT, "seed": 5} | kw
    return run_specificity_check(cis, trans, elements, genes, **kw)


def _as_categorical(result: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Ids as categoricals in a shuffled order, with categories no row uses."""
    rng = np.random.default_rng(seed)
    out = result.copy()
    for c in ("grna_target", "response_id"):
        cats = np.concatenate([pd.unique(out[c]), [f"unused_{c}_{k}" for k in range(3)]])
        out[c] = pd.Categorical(out[c], categories=rng.permutation(cats))
    return out


def _assert_same_tables(a, b):
    for name in TABLES:
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    assert a.background_rate == b.background_rate


@pytest.mark.parametrize("lfc_given", [True, False], ids=["log_2_fold_change", "fold_change"])
def test_a_screened_background_gives_the_same_tables(lfc_given):
    cis, trans, elements, genes = _screen(0)
    scr = _screened(trans)
    if lfc_given:
        cis, trans, scr = _with_lfc(cis), _with_lfc(trans), _with_lfc(scr)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = _check(cis, trans, elements, genes)
        got = _check(cis, scr, elements, genes)
    _assert_same_tables(full, got)

    # The background: every column the same, but p_value only where the row was resampled.
    assert list(full.background.columns) == [
        "grna_target",
        "response_id",
        "p_value",
        "log_2_fold_change",
        "background",
    ]
    assert list(got.background.columns) == [
        "grna_target",
        "response_id",
        "p_value",
        "log_2_fold_change",
        "screen",
        "p_value_floor",
        "background",
    ]
    shared = ["grna_target", "response_id", "log_2_fold_change", "background"]
    pd.testing.assert_frame_equal(full.background[shared], got.background[shared], check_exact=True)
    none = (got.background["screen"] == "none").to_numpy()
    assert np.array_equal(full.background["p_value"][none], got.background["p_value"][none])
    assert got.background["p_value"][~none].isna().all()

    # The fixture exercises every table, and both screens reach the background and the lookup.
    assert full.background["background"].sum() > 100
    assert set(full.broad_effect.loc[full.broad_effect["broad_effect"], "grna_target"]) == {LOUD}
    assert len(full.tss_targets) >= 4 and full.gene_lookup["lowers_nominal"].any()
    assert len(full.far_links) > 0
    counts = got.background["screen"].value_counts()
    assert counts["fold_change"] > 100 and counts["early_stop"] > 100
    tss_rows = scr.loc[scr["grna_target"].isin(set(full.tss_targets["grna_target"]))]
    assert {"fold_change", "early_stop", "none"} <= set(tss_rows["screen"])


def test_screened_rows_are_tested_and_carried():
    cis, trans, elements, genes = _screen(1)
    scr = _screened(trans)
    scr.loc[scr.index[:40], "pass_qc"] = False
    got = background.tested_pairs(scr, "trans_result")
    want = scr["pass_qc"] & (
        scr["p_value"].notna() | scr["screen"].isin(["fold_change", "early_stop"])
    )
    assert len(got) == int(want.sum())
    assert got["screen"].tolist() == scr.loc[want, "screen"].tolist()
    assert np.array_equal(
        got["p_value_floor"].to_numpy(), scr.loc[want, "p_value_floor"].to_numpy(), equal_nan=True
    )
    # Without the screen columns the frame is the one a discovery result always gave.
    plain = background.tested_pairs(_with_lfc(trans), "trans_result")
    assert list(plain.columns) == ["grna_target", "response_id", "p_value", "log_2_fold_change"]
    keep = trans["pass_qc"] & trans["p_value"].notna()
    want_plain = _with_lfc(trans).loc[keep, list(plain.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(plain, want_plain, check_exact=True)
    # A row with no p-value and no screen was not tested.
    assert (
        not background.tested_pairs(scr.assign(screen="none"), "trans_result")["p_value"]
        .isna()
        .any()
    )


def test_a_screened_row_that_could_be_a_call_is_refused():
    cis, trans, elements, genes = _screen(2)
    scr = _screened(trans)
    links = cis_links(cis, elements, genes, CUTOFF)

    # Stopped early at FLOOR, so not known to be above a cutoff at or over it.
    with pytest.raises(ValueError, match="early_stop.*not above the cutoff 0.06"):
        background_pairs(scr, links, elements, genes, 0.06)
    with pytest.raises(ValueError, match="early_stop"):
        _check(cis, scr, elements, genes, cutoff=0.06)

    # Screened on its fold change, yet with a negative effect.
    bad = scr.copy()
    row = bad.index[(bad["screen"] == "fold_change") & bad["pass_qc"]][0]
    bad.loc[row, "fold_change"] = 0.9
    with pytest.raises(ValueError, match="screen 'fold_change' and a negative"):
        background_pairs(bad, links, elements, genes, CUTOFF)

    # No floor at all.
    with pytest.raises(ValueError, match="missing or not above"):
        background_pairs(scr.drop(columns="p_value_floor"), links, elements, genes, CUTOFF)

    # An effect that is not negative is never a call, whatever its floor.
    up = scr.copy()
    rows = up.index[up["screen"] == "fold_change"][:5]
    up.loc[rows, ["screen", "p_value_floor"]] = ["early_stop", 0.001]
    got = background_pairs(up, links, elements, genes, CUTOFF)
    assert len(got) == len(background_pairs(scr, links, elements, genes, CUTOFF))


def test_a_tss_target_stopped_below_the_nominal_level_is_refused():
    cis, trans, elements, genes = _screen(3)
    scr = _screened(trans)
    full = _check(cis, trans, elements, genes)
    tss = full.tss_targets
    of_tss = (
        scr["grna_target"].isin(set(tss["grna_target"]))
        & (scr["screen"] == "early_stop")
        & scr["pass_qc"]
    )
    low = scr.copy()
    low.loc[low.index[of_tss][:3], "p_value_floor"] = 0.03  # above CUTOFF, below 0.05

    # The background alone accepts it: 0.03 is above the cutoff.
    links = cis_links(cis, elements, genes, CUTOFF)
    background_pairs(low, links, elements, genes, CUTOFF)
    with pytest.raises(ValueError, match=r"TSS targets.*at least the nominal level 0.05"):
        gene_lookup(tss, cis, low, CUTOFF)
    with pytest.raises(ValueError, match="TSS targets"):
        _check(cis, low, elements, genes)
    # At a lower nominal level it is safe again.
    gene_lookup(tss, cis, low, CUTOFF, nominal=0.02)


def test_categorical_ids_give_exactly_the_object_result():
    cis, trans, elements, genes = _screen(4)
    scr = _screened(trans)
    obj = _check(cis, scr, elements, genes)
    cat = _check(_as_categorical(cis, 1), _as_categorical(scr, 2), elements, genes)
    _assert_same_tables(obj, cat)
    pd.testing.assert_frame_equal(obj.background, cat.background, check_exact=True)
    for frame in (cat.links, cat.background, cat.broad_effect, cat.per_gene):
        for c in ("grna_target", "response_id"):
            if c in frame:
                assert frame[c].dtype == object
    got = background.tested_pairs(_as_categorical(scr, 3), "trans_result")
    pd.testing.assert_frame_equal(
        got, background.tested_pairs(scr, "trans_result"), check_exact=True
    )


def test_fold_change_stands_in_for_log_2_fold_change():
    cis, trans, elements, genes = _screen(5)
    trans.loc[trans.index[:3], ["p_value", "fold_change", "pass_qc"]] = [1e-4, 0.0, True]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fc_only = background.tested_pairs(trans, "trans_result")
    given = background.tested_pairs(_with_lfc(trans), "trans_result")
    pd.testing.assert_frame_equal(fc_only, given, check_exact=True)
    # A fold change of 0 is an infinitely negative effect, and called.
    zero = fc_only.loc[:2]
    assert np.isneginf(zero["log_2_fold_change"]).all()
    assert background.called_down(zero, CUTOFF).all()

    _assert_same_tables(
        _check(cis, trans, elements, genes),
        _check(_with_lfc(cis), _with_lfc(trans), elements, genes),
    )
    # A log_2_fold_change column wins over fold_change.
    own = trans.assign(log_2_fold_change=-7.0)
    assert (background.tested_pairs(own, "trans_result")["log_2_fold_change"] == -7.0).all()
    with pytest.raises(KeyError, match="log_2_fold_change"):
        background.tested_pairs(trans.drop(columns="fold_change"), "trans_result")


def test_genes_without_background_tests_warn_and_count_as_rate_zero():
    cis, trans, elements, genes = _screen(6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(cis, trans, elements, genes)  # every gene has background tests: no warning

    lonely = ["G0", "G9"]
    with pytest.warns(UserWarning, match="Drop them from cis_result") as record:
        res = _check(cis, trans.loc[~trans["response_id"].isin(lonely)], elements, genes, seed=8)
    theirs = res.links["response_id"].isin(lonely)
    message = next(str(r.message) for r in record if "cis gene(s)" in str(r.message))
    assert message.startswith(
        f"2 cis gene(s) have no background test, so their background rate counts as 0: "
        f"{int(theirs.sum())} cis pair(s) and {int(res.links.loc[theirs, 'link'].sum())} link(s)"
    )
    assert "'G0', 'G9'" in message

    per_gene = res.per_gene.set_index("response_id")
    assert (per_gene.loc[lonely, "trans_pairs"] == 0).all()
    assert per_gene.loc[lonely, "background_rate"].isna().all()
    # The numbers are the bootstrap's own, the genes at rate 0.
    again = above_background_by_distance(
        res.links, res.background, n_bootstrap=N_BOOT, seed=np.random.default_rng(8)
    )
    pd.testing.assert_frame_equal(res.by_distance, again, check_exact=True)
    rate = res.background.groupby("response_id")["background"].mean()
    bins = pd.cut(res.links["distance"], [0, 1e3, 1e4, 5e4, 1e5, np.inf], include_lowest=True)
    matched = res.links["response_id"].map(rate).fillna(0).groupby(bins, observed=False).mean()
    np.testing.assert_allclose(res.by_distance["background"], matched.to_numpy(), rtol=1e-12)


def test_a_cross_chromosome_cis_pair_is_listed():
    cis, trans, elements, genes = _screen(7)
    crossed = pd.concat(
        [trans.loc[trans["grna_target"] == t].head(4) for t in ("E10", "E11", "E12")]
    ).assign(p_value=0.5, pass_qc=True)
    with pytest.raises(ValueError, match="different chromosomes") as err:
        cis_links(pd.concat([cis, crossed], ignore_index=True), elements, genes, CUTOFF)
    message = str(err.value)
    assert "12 cis_result pair(s), of 3 target(s)" in message
    listed = re.findall(r"\('(E\d+)', '(G\d+)', '(chr\d)', '(chr\d)'\)", message)
    assert len(listed) == 10
    e_chrom = elements.set_index("grna_target")["chrom"]
    g_chrom = genes.set_index("response_id")["chrom"]
    want = [
        (t, g, e_chrom[t], g_chrom[g])
        for t, g in crossed.head(10)[["grna_target", "response_id"]].itertuples(index=False)
    ]
    assert listed == want


def test_the_dense_bootstrap_products_are_bitwise_the_sparse_ones(monkeypatch):
    cis, trans, elements, genes = _screen(8)
    links = cis_links(cis, elements, genes, CUTOFF)
    bg = background_pairs(trans, links, elements, genes, CUTOFF)
    n_e, n_g = links["grna_target"].nunique(), links["response_id"].nunique()
    assert len(bg) > background._DENSE_ABOVE * n_e * n_g  # dense by default

    seen = []
    weighted = background._weighted_sums
    monkeypatch.setattr(
        background,
        "_weighted_sums",
        lambda m, w: seen.append(isinstance(m, np.ndarray)) or weighted(m, w),
    )
    runs = {}
    for name, above in (("default", 0.5), ("sparse", 2.0), ("dense", 0.0)):
        monkeypatch.setattr(background, "_DENSE_ABOVE", above)
        seen.clear()
        runs[name] = above_background_by_distance(links, bg, n_bootstrap=N_BOOT, seed=3)
        runs[name + "_types"] = sorted(set(seen))
    assert runs["sparse_types"] == [False]
    assert runs["dense_types"] == [True]
    assert runs["default_types"] == [False, True]  # tested dense, calls sparse
    for name in ("sparse", "dense"):
        pd.testing.assert_frame_equal(runs["default"], runs[name], check_exact=True)


def test_the_weighted_sums_match_the_released_formula_bit_for_bit():
    """0.5.0 stored every test in `calls`, zeros included, and multiplied sparse."""
    rng = np.random.default_rng(9)
    n_e, n_g, n_draws = 300, 80, 50
    e, g = np.nonzero(rng.random((n_e, n_g)) < 0.7)
    values = (rng.random(e.size) < 0.1).astype(float)
    released = sparse.csc_matrix((values, (e, g)), shape=(n_e, n_g))
    w = rng.multinomial(n_e, np.full(n_e, 1.0 / n_e), n_draws).astype(float)
    want = (released.T @ w.T).T
    nz = values != 0
    calls = sparse.csc_matrix((values[nz], (e[nz], g[nz])), shape=(n_e, n_g))
    for m in (calls, calls.toarray(), released.toarray()):
        assert np.array_equal(background._weighted_sums(m, w), want)


def test_the_check_reads_the_trans_result_once_and_leaves_it_alone(monkeypatch):
    cis, trans, elements, genes = _screen(10)
    scr = _as_categorical(_screened(trans))
    before = scr.copy(deep=True)
    calls = []
    tested = background._tested
    monkeypatch.setattr(
        background,
        "_tested",
        lambda result, name, *a, **k: calls.append(name) or tested(result, name, *a, **k),
    )
    rng = np.random.default_rng(4)
    res = _check(cis, scr, elements, genes, seed=rng)
    assert calls.count("trans_result") == 1
    pd.testing.assert_frame_equal(scr, before, check_exact=True)

    # The pieces the check computes once are what the public functions give.
    pd.testing.assert_frame_equal(
        res.links, cis_links(cis, elements, genes, CUTOFF), check_exact=True
    )
    pd.testing.assert_frame_equal(
        res.background, background_pairs(scr, res.links, elements, genes, CUTOFF), check_exact=True
    )
    pd.testing.assert_frame_equal(
        res.gene_lookup, gene_lookup(res.tss_targets, cis, scr, CUTOFF), check_exact=True
    )
    # Both bootstraps as 0.5.0 ran them, the second on a background without the flagged elements.
    rng = np.random.default_rng(4)
    first = above_background_by_distance(res.links, res.background, n_bootstrap=N_BOOT, seed=rng)
    flagged = set(res.broad_effect.loc[res.broad_effect["broad_effect"], "grna_target"])
    assert flagged
    second = above_background_by_distance(
        res.links.loc[~res.links["grna_target"].isin(flagged)],
        res.background.loc[~res.background["grna_target"].isin(flagged)],
        n_bootstrap=N_BOOT,
        seed=rng,
    )
    pd.testing.assert_frame_equal(res.by_distance, first, check_exact=True)
    pd.testing.assert_frame_equal(res.by_distance_without_broad, second, check_exact=True)
