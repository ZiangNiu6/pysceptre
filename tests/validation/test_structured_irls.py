"""The structured IRLS fits the same GLM as the outer-product IRLS.

`glm.design.structured_design` splits a design matrix into dense columns and
groups of indicator columns, and `glm.irls.fit_*_glm_structured` form the
weighted normal equations from that split instead of from the `(n, p^2)` array
of per-row outer products. In exact arithmetic the two are the same
computation; these tests hold them to floating-point agreement, the same
iteration count, and R's own fits. See docs/design.md, "Indicator columns in
the IRLS".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysceptre.glm import irls
from pysceptre.glm.design import structured_design
from pysceptre.glm.irls import (
    fit_binomial_glm_batch,
    fit_binomial_glm_structured,
    fit_poisson_glm_batch,
    fit_poisson_glm_structured,
    structured_linear_predictor,
    structured_normal_equations,
)
from pysceptre.glm.nb_theta import estimate_theta
from pysceptre.pipeline.api import run_calibration_check, run_discovery_analysis


def _design(n=30_000, seed=0):
    """Intercept, two continuous columns, a 6-level factor, a crossed 3-level factor,
    a lone binary covariate and a trailing continuous column, in a shuffled order."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 6, n)
    b = rng.integers(0, 3, n)
    cols = (
        [np.ones(n), rng.normal(size=n)]
        + [(a == k).astype(float) for k in range(1, 6)]
        + [rng.normal(size=n)]
        + [(b == k).astype(float) for k in range(1, 3)]
        + [(rng.random(n) < 0.3).astype(float), rng.normal(0.0, 0.5, n)]
    )
    return np.column_stack(cols), rng


def test_groups_are_the_factors():
    X, _ = _design()
    d = structured_design(X)
    groups = [g.columns.tolist() for g in d.groups]
    # The lone binary is 1 in rows of both factors, so it is a group of its own.
    assert groups == [[2, 3, 4, 5, 6], [8, 9], [10]]
    assert d.dense.tolist() == [0, 1, 7, 11]
    assert np.array_equal(d.to_matrix(), X)
    for g in d.groups:
        member = X[:, g.columns]
        assert (member.sum(axis=1) <= 1).all()
        assert np.array_equal(g.codes, np.where(member.any(1), member.argmax(1), g.n_levels))


def test_no_indicators_and_near_indicators_stay_dense():
    rng = np.random.default_rng(1)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=n), np.zeros(n), np.r_[np.ones(n - 1), 2.0]])
    d = structured_design(X)
    assert d.groups == ()
    assert d.dense.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_normal_equations_equal_the_full_products(seed):
    X, rng = _design(seed=seed)
    d = structured_design(X)
    w = rng.uniform(1e-3, 5.0, X.shape[0])
    z = rng.normal(size=X.shape[0])
    A, b = structured_normal_equations(d, w, w * z)
    np.testing.assert_allclose(A, X.T @ (w[:, None] * X), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(b, X.T @ (w * z), rtol=1e-12, atol=1e-9)
    beta = rng.normal(size=X.shape[1])
    np.testing.assert_allclose(
        structured_linear_predictor(d, beta), X @ beta, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("scale", [0.002, 0.1, 3.0, 40.0])
def test_poisson_fit_matches_the_outer_product_route(scale):
    X, rng = _design(seed=3)
    d = structured_design(X)
    beta = rng.normal(0.0, 0.3, X.shape[1])
    y = rng.poisson(scale * np.exp(X @ beta)).astype(float)
    s = fit_poisson_glm_structured(d, y)
    o = fit_poisson_glm_batch(X, y)
    assert s.n_iter == o.n_iter and s.converged == o.converged
    np.testing.assert_allclose(s.fitted_values, o.fitted_values, rtol=1e-9)
    np.testing.assert_allclose(s.deviance, o.deviance, rtol=1e-11)
    dfr = X.shape[0] - X.shape[1]
    t_s, m_s = estimate_theta(y, s.fitted_values, dfr)
    t_o, m_o = estimate_theta(y, o.fitted_values, dfr)
    assert m_s == m_o
    np.testing.assert_allclose(t_s, t_o, rtol=1e-8)


def test_poisson_mu_floor_is_honoured():
    X, rng = _design(seed=4)
    d = structured_design(X)
    y = np.zeros(X.shape[0])
    y[rng.choice(X.shape[0], 3, replace=False)] = 1.0
    for floor in (1e-10, np.finfo(float).eps):
        s = fit_poisson_glm_structured(d, y, mu_floor=floor)
        o = fit_poisson_glm_batch(X, y, mu_floor=floor)
        assert s.fitted_values.min() >= floor
        assert s.n_iter == o.n_iter
        # Three counts in 30,000 cells: levels without a count have no finite MLE, so their
        # fitted means run toward the floor along a direction neither route pins down, and
        # agree only at an absolute level.
        np.testing.assert_allclose(s.fitted_values, o.fitted_values, rtol=1e-7, atol=1e-11)


@pytest.mark.parametrize("n_trt", [40, 400, 3_000])
def test_binomial_fit_matches_the_outer_product_route(n_trt):
    X, rng = _design(seed=5)
    d = structured_design(X)
    y = np.zeros(X.shape[0])
    y[rng.choice(X.shape[0], n_trt, replace=False)] = 1.0
    s = fit_binomial_glm_structured(d, y)
    o = fit_binomial_glm_batch(X, y)
    assert s.n_iter == o.n_iter
    np.testing.assert_allclose(s.fitted_values, o.fitted_values, rtol=1e-9)


def test_structured_fits_match_r(ground_truth):
    X = np.array(ground_truth["X"])
    d = structured_design(X)
    n, p = X.shape
    for gene in ground_truth["genes"]:
        y = np.array(gene["y"], dtype=float)
        fit = fit_poisson_glm_structured(d, y)
        np.testing.assert_allclose(fit.coefs, gene["fitted_coefs"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(fit.fitted_values, gene["mu"], rtol=1e-6, atol=1e-8)
        theta, _ = estimate_theta(y=y, mu=fit.fitted_values, dfr=n - p)
        np.testing.assert_allclose(max(min(theta, 1000.0), 0.01), gene["theta"], rtol=1e-4)
    for target in ground_truth["targets"]:
        y = np.zeros(n)
        y[np.array(target["trt_idxs_1based"]) - 1] = 1.0
        fit = fit_binomial_glm_structured(d, y)
        np.testing.assert_allclose(
            fit.fitted_values, target["fitted_probabilities"], rtol=1e-6, atol=1e-8
        )


def test_wrong_length_response_is_refused():
    X, _ = _design(n=100)
    with pytest.raises(ValueError, match="rows"):
        fit_poisson_glm_structured(structured_design(X), np.zeros(99))


def _screen(seed=11, n=12_000, n_genes=8, n_targets=6):
    """A small complement screen whose covariates include a 4-level batch factor."""
    rng = np.random.default_rng(seed)
    batch = rng.integers(0, 4, n)
    X = np.column_stack(
        [np.ones(n), rng.normal(0.0, 0.3, n)] + [(batch == k).astype(float) for k in range(1, 4)]
    )
    mu = np.exp(0.4 + 0.6 * X[:, 1] + 0.2 * X[:, 2] - 0.3 * X[:, 4])
    counts = rng.negative_binomial(
        5.0, 5.0 / (5.0 + mu[None, :] * rng.uniform(0.3, 3.0, (n_genes, 1)))
    )
    genes = [f"g{i}" for i in range(n_genes)]
    targets = {f"t{j}": np.sort(rng.choice(n, 120, replace=False)) for j in range(n_targets)}
    counts = counts.astype(float)
    counts[0, targets["t0"]] = rng.negative_binomial(5.0, 5.0 / (5.0 + 0.3 * mu[targets["t0"]]))
    pairs = pd.DataFrame(
        [(g, t) for g in genes for t in targets], columns=["response_id", "grna_target"]
    )
    return counts, genes, X, targets, pairs


def test_discovery_with_the_structured_solver_matches():
    counts, genes, X, targets, pairs = _screen()
    runs = {
        solver: run_discovery_analysis(
            counts, genes, X, targets, pairs, seed=3, glm_solver=solver, target_chunk_size=3
        )
        for solver in ("outer", "structured")
    }
    a, b = runs["outer"], runs["structured"]
    for c in ("fold_change", "se_fold_change", "z_orig"):
        np.testing.assert_allclose(b[c], a[c], rtol=1e-8)
    # Statistics a few ulps apart give the same counts, so the same p-values and stages.
    np.testing.assert_array_equal(b["stage"], a["stage"])
    np.testing.assert_allclose(b["p_value"], a["p_value"], rtol=1e-6)


def test_glm_solver_is_validated():
    counts, genes, X, targets, pairs = _screen(n=2_000)
    with pytest.raises(ValueError, match="glm_solver"):
        run_discovery_analysis(counts, genes, X, targets, pairs, seed=1, glm_solver="qr")
    ntc = {"nt1": np.arange(1_900, 2_000)}
    with pytest.raises(ValueError, match="complement control group only"):
        run_discovery_analysis(
            counts,
            genes,
            X,
            {t: c[c < 1_900] for t, c in targets.items()},
            pairs,
            seed=1,
            moi="low",
            control_group="nt_cells",
            ntc_grna_cells=ntc,
            glm_solver="structured",
        )


@pytest.mark.skipif(not irls._HAVE_NUMBA, reason="needs numba")
def test_compiled_and_bincount_sums_agree_bit_for_bit():
    X, rng = _design(seed=6)
    d = structured_design(X)
    w = rng.uniform(1e-3, 5.0, X.shape[0])
    wz = w * rng.normal(size=X.shape[0])
    A1, b1 = structured_normal_equations(d, w, wz, compiled=True)
    A0, b0 = structured_normal_equations(d, w, wz, compiled=False)
    assert np.array_equal(A1, A0) and np.array_equal(b1, b0)


_COLUMNS = ["p_value", "fold_change", "se_fold_change", "z_orig", "stage"]


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a[["response_id", "grna_target"]].equals(b[["response_id", "grna_target"]]) and all(
        np.array_equal(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True) for c in _COLUMNS
    )


@pytest.mark.parametrize("mechanism", ["crt", "permutations"])
def test_structured_results_do_not_depend_on_chunks_or_workers(mechanism):
    counts, genes, X, targets, pairs = _screen(n=6_000)

    def run(n_jobs, chunk):
        return run_discovery_analysis(
            counts,
            genes,
            X,
            targets,
            pairs,
            seed=3,
            glm_solver="structured",
            resampling_mechanism=mechanism,
            n_jobs=n_jobs,
            target_chunk_size=chunk,
        )

    base = run(1, 2)
    for n_jobs, chunk in [(3, 2), (3, 200), (1, 5)]:
        assert _same(run(n_jobs, chunk), base), (n_jobs, chunk)


def test_structured_route_never_builds_the_outer_products(monkeypatch):
    from pysceptre.pipeline import discovery

    counts, genes, X, targets, pairs = _screen(n=4_000)
    expected = run_discovery_analysis(
        counts, genes, X, targets, pairs, seed=3, glm_solver="structured", target_chunk_size=2
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the outer-product route was called")

    for name in ("x_outer_flat", "fit_poisson_glm_batch", "fit_binomial_glm_batch"):
        monkeypatch.setattr(discovery, name, refuse)
    for n_jobs in (1, 2):
        got = run_discovery_analysis(
            counts,
            genes,
            X,
            targets,
            pairs,
            seed=3,
            glm_solver="structured",
            target_chunk_size=2,
            n_jobs=n_jobs,
        )
        assert _same(got, expected), n_jobs


def test_permutations_fit_no_targets(monkeypatch):
    from pysceptre.pipeline import discovery

    counts, genes, X, targets, pairs = _screen(n=4_000)
    calls = []
    original = discovery.fit_binomial_glm_structured

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(discovery, "fit_binomial_glm_structured", counted)
    for mechanism, expected in [("permutations", 0), ("crt", len(targets))]:
        calls.clear()
        run_discovery_analysis(
            counts,
            genes,
            X,
            targets,
            pairs,
            seed=3,
            glm_solver="structured",
            resampling_mechanism=mechanism,
        )
        assert len(calls) == expected, mechanism


def test_an_unknown_solver_is_refused_on_either_control_group():
    counts, genes, X, targets, pairs = _screen(n=2_000)
    with pytest.raises(ValueError, match="glm_solver must be one of"):
        run_discovery_analysis(
            counts,
            genes,
            X,
            {t: c[c < 1_900] for t, c in targets.items()},
            pairs,
            seed=1,
            moi="low",
            control_group="nt_cells",
            ntc_grna_cells={"nt1": np.arange(1_900, 2_000)},
            glm_solver="qr",
        )


def test_the_calibration_check_refuses_the_structured_solver_against_the_nt_cells():
    """As discovery does, although that check runs the complement engine inside the NT pool."""
    counts, genes, X, _, _ = _screen(n=2_000)
    ntc = {"nt1": np.arange(1_800, 1_900), "nt2": np.arange(1_900, 2_000)}
    args = dict(n_calibration_pairs=4, calibration_group_size=1, seed=1)
    with pytest.raises(ValueError, match="complement control group only"):
        run_calibration_check(
            counts,
            genes,
            X,
            ntc,
            moi="low",
            control_group="nt_cells",
            glm_solver="structured",
            **args,
        )
    for control_group in ("nt_cells", "complement"):
        with pytest.raises(ValueError, match="glm_solver must be one of"):
            run_calibration_check(
                counts,
                genes,
                X,
                ntc,
                moi="low",
                control_group=control_group,
                glm_solver="qr",
                **args,
            )
    # The complement control group takes it.
    result = run_calibration_check(counts, genes, X, ntc, glm_solver="structured", **args)
    assert len(result) == 4 and result["p_value"].between(0.0, 1.0).all()


@pytest.mark.skipif(not irls._HAVE_NUMBA, reason="needs numba")
def test_warming_the_compiled_sums_changes_nothing():
    X, rng = _design(n=2_000, seed=8)
    d = structured_design(X)
    w = rng.uniform(1e-3, 5.0, X.shape[0])
    before = structured_normal_equations(d, w, w, compiled=True)
    irls.warm_compiled_sums()
    after = structured_normal_equations(d, w, w, compiled=True)
    assert all(np.array_equal(x, y) for x, y in zip(before, after, strict=True))
