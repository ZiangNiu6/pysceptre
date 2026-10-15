"""Batched IRLS for Poisson (log link) and binomial (logit link) GLMs.

Ports the algorithm behind R's `stats::glm.fit`: iteratively reweighted
least squares on the working response, with deviance-based convergence
(`epsilon = 1e-8`, `maxit = 25`, matching R's defaults). Because both models
have concave, canonical-link log-likelihoods, IRLS converges to the unique
MLE regardless of implementation details -- exact line-for-line replication
of R's algorithm is not required for the fitted coefficients to match R's
`glm.fit` output to numerical precision.

Vectorized across a batch of response columns (k genes/targets) sharing one
design matrix X, since that's the actual batching opportunity in sceptre's
"complement" control-group mode (see pipeline/discovery.py): the p x p
weighted normal-equations system differs per column (because the IRLS
weights depend on that column's own fitted mu), but can be solved for all
k columns at once via a batched `numpy.linalg.solve`.

The per-iteration matmuls here are "thin" (p is a handful of covariates, so
every matmul has a tiny inner/output dimension even though n is large) and
called many times in a tight loop. Profiling showed multi-threaded OpenBLAS
is a net *loss* for this shape/call-pattern in this environment -- thread
coordination overhead per call dominates the actual (small) amount of work
each call does (measured: a 150-column batch over 100k cells went from
68.7s to 17.7s just by forcing BLAS to one thread). `threadpoolctl` scopes
that restriction to this module's own matmuls rather than clobbering BLAS
threading process-wide (which would also affect any other numpy/scipy work
happening in the same process outside pysceptre).

A second route, `fit_*_glm_structured`, fits one response on a
`glm.design.StructuredDesign`, forming the normal equations from the
design's 0/1 indicator columns by per-level sums instead of from
`x_outer_flat`. See docs/design.md, "Indicator columns in the IRLS".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from threadpoolctl import ThreadpoolController

from .design import StructuredDesign

try:
    from numba import njit

    @njit(cache=True, nogil=True)
    def _indicator_sums_compiled(codes, levels, offsets, wXd, w, wz, pair_g, pair_h, pair_off):
        """The per-level sums of `_indicator_sums_numpy`, in one pass over the cells.

        Each sum adds the cells in index order, as `np.bincount` does, so the
        two routes agree bit for bit; the reference level's sums are skipped.
        """
        n_groups, n = codes.shape
        d = wXd.shape[1]
        total = offsets[n_groups]
        diag = np.zeros(total)
        bsum = np.zeros(total)
        S = np.zeros((d, total))
        cross = np.zeros(pair_off[pair_g.shape[0]])
        for i in range(n):
            wi = w[i]
            for g in range(n_groups):
                c = codes[g, i]
                if c < levels[g]:
                    o = offsets[g] + c
                    diag[o] += wi
                    bsum[o] += wz[i]
                    for k in range(d):
                        S[k, o] += wXd[i, k]
            for q in range(pair_g.shape[0]):
                g = pair_g[q]
                h = pair_h[q]
                ch = codes[h, i]
                cg = codes[g, i]
                if cg < levels[g] and ch < levels[h]:
                    cross[pair_off[q] + cg * levels[h] + ch] += wi
        return diag, bsum, S, cross

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    _HAVE_NUMBA = False

_EPS = 1e-8
_MAXIT = 25
_MU_FLOOR = 1e-10


@dataclass
class GlmFitBatchResult:
    coefs: np.ndarray  # (k, p)
    fitted_values: np.ndarray  # (k, n)
    deviance: np.ndarray  # (k,)
    n_iter: np.ndarray  # (k,)
    converged: np.ndarray  # (k,) bool


def _as_2d(Y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Promote a single response vector to a batch of one.

    Shapes here are batch-axis-first -- `(k, n)`, k responses of n
    observations -- matching numpy's own batched-linalg convention
    (`np.linalg.solve` takes `(k, p, p)`), keeping each response contiguous,
    and letting the normal equations be assembled with no transposes at all.
    """
    if Y.ndim == 1:
        return Y[None, :], True
    return Y, False


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    # 2 * sum(y*log(y/mu) - (y - mu)), with the convention y*log(y/mu) = 0 at y = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * np.sum(term - (y - mu), axis=-1)


def _binomial_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        t2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
    return 2.0 * np.sum(t1 + t2, axis=-1)


def _binomial_saturated(y: np.ndarray) -> np.ndarray:
    """The `y`-only half of the binomial deviance, which the IRLS loop never
    changes.

    `2 * sum(y log y + (1-y) log(1-y))`. Exactly zero for 0/1 responses, which
    is every binomial fit in this package -- gRNA target indicators -- but
    computed rather than assumed so the deviance stays correct for
    proportions.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y), 0.0) + np.where(y < 1, (1 - y) * np.log1p(-y), 0.0)
    return 2.0 * np.sum(t, axis=-1)


def _binomial_deviance_from_eta(
    y: np.ndarray, mu: np.ndarray, eta: np.ndarray, saturated: np.ndarray
) -> np.ndarray:
    """The same deviance, rearranged around quantities the loop already holds.

    `y log(y/mu) + (1-y) log((1-y)/(1-mu))` expands to a saturated term in
    `y` alone plus `-(log(1-mu) + y * logit(mu))`, and `logit(mu)` is `eta`,
    which IRLS carries as loop state. So one `log1p` and a dot product
    replace two logs, two divisions and two `np.where`s over a `(k, n)`
    array, every iteration.

    Measured 45.3 ms -> 26.0 ms at day0's shape, 1.74x, on a term that was
    107.6 s of a 673.5 s serial CRT profile -- the largest single piece of
    the target fit.

    **`eta` is the unclipped logit while `mu` is clipped at `_MU_FLOOR`**, so
    the two disagree for any entry that clipped. That is 1 in 7.9M at day0's
    scale, and where it happens this form uses the unclipped value, which is
    the better-conditioned one. Agreement with the direct form was 1.66e-16
    relative.
    """
    return saturated - 2.0 * (np.sum(np.log1p(-mu), axis=-1) + np.sum(y * eta, axis=-1))


def _batched_wls_solve(
    X: np.ndarray, X_outer_flat: np.ndarray, w: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Solve, for each of k columns, (X^T diag(w_k) X) beta_k = X^T diag(w_k) z_k.

    X: (n, p) shared design matrix. X_outer_flat: (n, p*p), the precomputed
    (and iteration-independent) per-row outer product X[n,:] outer X[n,:],
    flattened -- passed in rather than recomputed every IRLS iteration.
    w, z: (k, n) per-response IRLS weights and working response.
    Returns beta: (k, p).

    Deliberately expressed as two plain matmuls (BLAS-backed) rather than a
    per-iteration `einsum` building a (k, p, n) intermediate: for realistic
    problem sizes (k in the hundreds to thousands of genes/targets, n in the
    hundreds of thousands of cells, p a handful of covariates), materializing
    a (k, p, n) tensor every one of up to 25 IRLS iterations dominated
    runtime by orders of magnitude (profiled: ~36s for a 292-gene x
    100k-cell batch) -- p is small, so there's no need to ever allocate
    anything of size k*p*n.
    """
    p = X.shape[1]
    # Batch-axis-first means both products come out already batched, and the
    # reshape below is a view rather than a copy -- with (n, k) inputs the
    # same computation needs a transpose on each of A and b.
    A = (w @ X_outer_flat).reshape(-1, p, p)  # (k, n) @ (n, p*p) -> (k, p, p)
    b = (w * z) @ X  # (k, n) @ (n, p) -> (k, p)
    return np.linalg.solve(A, b[:, :, None])[:, :, 0]  # (k, p)


def x_outer_flat(X: np.ndarray) -> np.ndarray:
    """`(n, p*p)` of per-row outer products of `X`, the one input to the IRLS
    that depends on nothing but the design matrix.

    Hoisted to a function so a caller can build it **once per run** and hand
    the same array to every fit. It was already hoisted out of the IRLS
    iteration -- the loop never changes it -- but not out of the call, so a
    day0 CRT run rebuilt it about 454 times: 237 gene fits at one gene each,
    plus 217 target chunks. At 567,690 cells and 11 covariates that is a
    550 MB array and a 550 MB temporary every time, for a matrix that is
    identical on every call.

    Sharing it is also what makes splitting a fit across workers affordable:
    sub-batches read one array instead of each constructing its own.
    """
    n, p = X.shape
    return (X[:, :, None] * X[:, None, :]).reshape(n, p * p)


def _fit_batch(
    X: np.ndarray,
    Y: np.ndarray,
    family: str,
    *,
    eps: float = _EPS,
    maxit: int = _MAXIT,
    X_outer_flat: np.ndarray | None = None,
    mu_floor: float = _MU_FLOOR,
) -> GlmFitBatchResult:
    n, p = X.shape
    Y, was_1d = _as_2d(Y)
    k, n_y = Y.shape
    if n_y != n:
        raise ValueError(f"X has {n} rows but Y has {n_y} columns")

    if family == "poisson":
        mu = Y + 0.1
        deviance_fn = _poisson_deviance
    elif family == "binomial":
        # **Started at the marginal rate, not the textbook `(y + 0.5) / 2`.**
        # That default puts every cell at mu = 0.25 for a 0/1 response, so
        # eta starts at -1.10; the gRNA-target fits this package runs have a
        # true rate near 0.001, so eta* is about -6.9 and IRLS spends most of
        # its nine iterations simply travelling there. Starting from the
        # intercept-only fit -- the marginal rate of each response -- begins
        # essentially at eta*, and only the covariate structure is left to
        # solve for.
        #
        # This is a starting point, not a change of estimand: IRLS converges
        # to the same MLE either way, to the same tolerance.
        rate = np.clip(Y.mean(axis=1, keepdims=True), _MU_FLOOR, 1.0 - _MU_FLOOR)
        mu = np.broadcast_to(rate, Y.shape).copy()
        deviance_fn = _binomial_deviance
    else:
        raise ValueError(f"unsupported family: {family}")

    eta = np.log(mu) if family == "poisson" else np.log(mu / (1 - mu))
    saturated = None if family == "poisson" else _binomial_saturated(Y)
    dev_old = np.full(k, np.inf)
    converged = np.zeros(k, dtype=bool)
    n_iter = np.zeros(k, dtype=int)
    beta = np.zeros((k, p))
    # Depends only on X, so a caller that fits repeatedly against the same
    # design matrix should build it once with `x_outer_flat` and pass it in.
    if X_outer_flat is None:
        X_outer_flat = x_outer_flat(X)

    # `mu` is carried forward as loop state (rather than recomputed via exp(eta)
    # at the top of every iteration) and every elementwise op below is
    # restricted to the not-yet-converged columns -- at real dataset scale
    # (hundreds of genes/targets x hundreds of thousands of cells), each full
    # (k, n) elementwise pass costs real wall-clock time, and IRLS otherwise
    # redundantly recomputes exp(eta) == mu twice per iteration.
    for it in range(1, maxit + 1):
        active = ~converged
        n_active = int(np.sum(active))
        if n_active == 0:
            break
        all_active = n_active == k  # avoid a full-array fancy-index *copy* in
        # the (extremely common) case where every column is still active --
        # a boolean-indexing copy of the whole (k, n) array every iteration
        # was measured to cost more than the work it saves once most/all
        # columns converge together, which is the typical case here since
        # every gene/target shares the same design matrix.

        # Row slices: each response is contiguous, so this copy is sequential.
        Y_a = Y if all_active else Y[active]
        eta_a = eta if all_active else eta[active]
        mu_a = mu if all_active else mu[active]

        if family == "poisson":
            dmu_deta_a = mu_a
            var_a = mu_a
        else:  # binomial
            dmu_deta_a = mu_a * (1 - mu_a)
            var_a = dmu_deta_a

        w_a = (dmu_deta_a**2) / var_a
        z_a = eta_a + (Y_a - mu_a) / dmu_deta_a

        beta_a = _batched_wls_solve(X, X_outer_flat, w_a, z_a)
        eta_new_a = beta_a @ X.T  # (k_a, p) @ (p, n) -> (k_a, n)

        mu_new_a = (
            np.clip(np.exp(eta_new_a), mu_floor, None)
            if family == "poisson"
            else np.clip(1.0 / (1.0 + np.exp(-eta_new_a)), _MU_FLOOR, 1 - _MU_FLOOR)
        )
        if family == "poisson":
            dev_new_a = deviance_fn(Y_a, mu_new_a)
        else:
            sat_a = saturated if all_active else saturated[active]
            dev_new_a = _binomial_deviance_from_eta(Y_a, mu_new_a, eta_new_a, sat_a)

        dev_old_a = dev_old if all_active else dev_old[active]
        newly_converged_a = np.abs(dev_new_a - dev_old_a) / (np.abs(dev_new_a) + 0.1) < eps

        if all_active:
            beta[:, :] = beta_a
            eta[:, :] = eta_new_a
            mu[:, :] = mu_new_a
            dev_old[:] = dev_new_a
            n_iter[:] = it
            converged[:] = newly_converged_a
        else:
            beta[active] = beta_a
            eta[active] = eta_new_a
            mu[active] = mu_new_a
            dev_old[active] = dev_new_a
            n_iter[active] = it
            converged[active] = newly_converged_a

    mu_final = mu

    if was_1d:
        return GlmFitBatchResult(
            coefs=beta[0],
            fitted_values=mu_final[0],
            deviance=dev_old[0],
            n_iter=n_iter[0],
            converged=converged[0],
        )
    return GlmFitBatchResult(
        coefs=beta, fitted_values=mu_final, deviance=dev_old, n_iter=n_iter, converged=converged
    )


_CONTROLLER: ThreadpoolController | None = None


def one_blas_thread():
    """`threadpool_limits(limits=1, user_api="blas")`, without rescanning the loaded
    libraries on every call: the scan cost more than a one-pair fit. The limit
    is process-wide, so concurrent fits should run inside one outer limit."""
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = ThreadpoolController()
    return _CONTROLLER.limit(limits=1, user_api="blas")


def fit_poisson_glm_batch(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    eps: float = _EPS,
    maxit: int = _MAXIT,
    X_outer_flat: np.ndarray | None = None,
    mu_floor: float = _MU_FLOOR,
) -> GlmFitBatchResult:
    """X: (n, p) shared design matrix. Y: (k, n) or (n,) response(s), one per row.

    `X_outer_flat`: optional, from `x_outer_flat(X)`. Pass it when fitting
    repeatedly against the same design matrix; it is rebuilt per call
    otherwise.

    `mu_floor`: lower bound on a fitted mean. R's `glm.fit` bounds it at the
    machine epsilon; the default here is 1e-10. See docs/design.md, "One
    Poisson GLM per gRNA".
    """
    with one_blas_thread():
        return _fit_batch(
            X, Y, "poisson", eps=eps, maxit=maxit, X_outer_flat=X_outer_flat, mu_floor=mu_floor
        )


def fit_binomial_glm_batch(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    eps: float = _EPS,
    maxit: int = _MAXIT,
    X_outer_flat: np.ndarray | None = None,
) -> GlmFitBatchResult:
    """X: (n, p) shared design matrix. Y: (k, n) or (n,) 0/1 indicator(s), one per row.

    `X_outer_flat`: optional, from `x_outer_flat(X)`. Pass it when fitting
    repeatedly against the same design matrix; it is rebuilt per call
    otherwise.
    """
    with one_blas_thread():
        return _fit_batch(X, Y, "binomial", eps=eps, maxit=maxit, X_outer_flat=X_outer_flat)


def warm_compiled_sums() -> None:
    """Compile the per-level sums now, or load them from numba's cache, if numba is installed.

    A forked worker inherits the compiled function, so calling this before
    forking spares each worker its own compile or cache load.
    """
    if _HAVE_NUMBA:
        none = np.zeros(0, dtype=np.int64)
        _indicator_sums_compiled(
            np.zeros((1, 1), dtype=np.int64),
            np.ones(1, dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.zeros((1, 1)),
            np.zeros(1),
            np.zeros(1),
            none,
            none,
            np.zeros(1, dtype=np.int64),
        )


def _indicator_layout(design: StructuredDesign):
    """Offsets of each group's levels, and the pairs of groups with their table offsets."""
    levels = design.levels
    offsets = np.zeros(levels.size + 1, dtype=np.int64)
    np.cumsum(levels, out=offsets[1:])
    pairs = [(g, h) for g in range(levels.size) for h in range(g + 1, levels.size)]
    pair_g = np.asarray([g for g, _ in pairs], dtype=np.int64)
    pair_h = np.asarray([h for _, h in pairs], dtype=np.int64)
    pair_off = np.zeros(len(pairs) + 1, dtype=np.int64)
    np.cumsum([levels[g] * levels[h] for g, h in pairs], out=pair_off[1:])
    return offsets, pair_g, pair_h, pair_off


def _indicator_sums_numpy(design, offsets, wXd, w, wz, pair_g, pair_h, pair_off):
    """Per-level sums of `w`, `wz` and each column of `wXd`, and each pair of groups'
    table of `w`, by `np.bincount`; the layout `_indicator_sums_compiled` returns."""
    total = int(offsets[-1])
    diag = np.zeros(total)
    bsum = np.zeros(total)
    S = np.zeros((wXd.shape[1], total))
    for g, group in enumerate(design.groups):
        L, codes, o = group.n_levels, group.codes, int(offsets[g])
        diag[o : o + L] = np.bincount(codes, weights=w, minlength=L + 1)[:L]
        bsum[o : o + L] = np.bincount(codes, weights=wz, minlength=L + 1)[:L]
        for k in range(wXd.shape[1]):
            S[k, o : o + L] = np.bincount(codes, weights=wXd[:, k], minlength=L + 1)[:L]
    cross = np.zeros(int(pair_off[-1]))
    for q in range(pair_g.size):
        g, h = design.groups[pair_g[q]], design.groups[pair_h[q]]
        Lg, Lh = g.n_levels, h.n_levels
        table = np.bincount(
            g.codes * (Lh + 1) + h.codes, weights=w, minlength=(Lg + 1) * (Lh + 1)
        ).reshape(Lg + 1, Lh + 1)[:Lg, :Lh]
        cross[pair_off[q] : pair_off[q + 1]] = table.ravel()
    return diag, bsum, S, cross


def structured_normal_equations(
    design: StructuredDesign, w: np.ndarray, wz: np.ndarray, *, compiled: bool | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """`(X^T diag(w) X, X^T wz)` for a `StructuredDesign`, in the design's column order.

    The dense block is two BLAS products; every block that touches an
    indicator group is a sequential per-level sum: the cross products with
    the dense columns, the group's diagonal, and the table of each pair of
    groups. That is `O(n * (d^2 + G*d + G^2))` work for `d` dense columns and
    `G` groups, against `O(n * p^2)` for the full products, and no `(n, p^2)`
    array. Equal to `X.T @ (w[:, None] * X)`
    and `X.T @ wz` in exact arithmetic, not bit for bit.

    `compiled` picks the one-pass numba kernel (the default when numba is
    installed) or `np.bincount`; both add every sum's terms in cell order, so
    they agree bit for bit. See docs/design.md, "Indicator columns in the IRLS".
    """
    p = design.p
    A = np.zeros((p, p))
    b = np.zeros(p)
    D = design.dense
    wXd = design.X_dense * w[:, None]
    if D.size:
        A[np.ix_(D, D)] = design.X_dense.T @ wXd
        b[D] = design.X_dense.T @ wz
    if not design.groups:
        return A, b
    offsets, pair_g, pair_h, pair_off = _indicator_layout(design)
    use_compiled = _HAVE_NUMBA if compiled is None else compiled
    if use_compiled and not _HAVE_NUMBA:
        raise RuntimeError("compiled=True needs numba")
    sums = _indicator_sums_compiled if use_compiled else None
    if sums is not None:
        diag, bsum, S, cross = sums(
            design.codes_matrix, design.levels, offsets, wXd, w, wz, pair_g, pair_h, pair_off
        )
    else:
        diag, bsum, S, cross = _indicator_sums_numpy(
            design, offsets, wXd, w, wz, pair_g, pair_h, pair_off
        )
    for g, group in enumerate(design.groups):
        cols, sl = group.columns, slice(int(offsets[g]), int(offsets[g + 1]))
        A[cols, cols] = diag[sl]
        b[cols] = bsum[sl]
        if D.size:
            A[np.ix_(D, cols)] = S[:, sl]
            A[np.ix_(cols, D)] = S[:, sl].T
    for q in range(pair_g.size):
        g, h = design.groups[pair_g[q]], design.groups[pair_h[q]]
        table = cross[pair_off[q] : pair_off[q + 1]].reshape(g.n_levels, h.n_levels)
        A[np.ix_(g.columns, h.columns)] = table
        A[np.ix_(h.columns, g.columns)] = table.T
    return A, b


def structured_linear_predictor(design: StructuredDesign, beta: np.ndarray) -> np.ndarray:
    """`X @ beta` for a `StructuredDesign`: the dense columns' product plus each group's level."""
    eta = design.X_dense @ beta[design.dense] if design.dense.size else np.zeros(design.n)
    for g in design.groups:
        eta += np.append(beta[g.columns], 0.0)[g.codes]
    return eta


def _fit_structured(
    design: StructuredDesign,
    y: np.ndarray,
    family: str,
    *,
    eps: float,
    maxit: int,
    mu_floor: float,
) -> GlmFitBatchResult:
    """`_fit_batch` for one response, with the normal equations of a `StructuredDesign`.

    The same IRLS step for step: the same starting point, weights, working
    response, deviance and stopping rule, and the same floor on the mean.
    Only `X^T W X`, `X^T W z` and `X beta` are formed differently.
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (design.n,):
        raise ValueError(f"the design has {design.n} rows but y has shape {y.shape}")
    if family == "poisson":
        mu = y + 0.1
        eta = np.log(mu)
        saturated = None
    elif family == "binomial":
        # Started at the marginal rate, as `_fit_batch` starts.
        rate = float(np.clip(y.mean(), _MU_FLOOR, 1.0 - _MU_FLOOR))
        mu = np.full(design.n, rate)
        eta = np.log(mu / (1 - mu))
        saturated = float(_binomial_saturated(y[None, :])[0])
    else:
        raise ValueError(f"unsupported family: {family}")

    dev_old = np.inf
    converged = False
    n_iter = 0
    beta = np.zeros(design.p)
    for it in range(1, maxit + 1):
        if family == "poisson":
            dmu_deta = mu
            var = mu
        else:
            dmu_deta = mu * (1 - mu)
            var = dmu_deta
        w = (dmu_deta**2) / var
        z = eta + (y - mu) / dmu_deta
        A, b = structured_normal_equations(design, w, w * z)
        beta = np.linalg.solve(A, b)
        eta = structured_linear_predictor(design, beta)
        if family == "poisson":
            mu = np.clip(np.exp(eta), mu_floor, None)
            dev_new = float(_poisson_deviance(y, mu))
        else:
            mu = np.clip(1.0 / (1.0 + np.exp(-eta)), _MU_FLOOR, 1 - _MU_FLOOR)
            dev_new = float(_binomial_deviance_from_eta(y, mu, eta, saturated))
        n_iter = it
        converged = abs(dev_new - dev_old) / (abs(dev_new) + 0.1) < eps
        dev_old = dev_new
        if converged:
            break
    return GlmFitBatchResult(
        coefs=beta, fitted_values=mu, deviance=dev_old, n_iter=n_iter, converged=converged
    )


def fit_poisson_glm_structured(
    design: StructuredDesign,
    y: np.ndarray,
    *,
    eps: float = _EPS,
    maxit: int = _MAXIT,
    mu_floor: float = _MU_FLOOR,
) -> GlmFitBatchResult:
    """One Poisson GLM by IRLS on a `StructuredDesign`; the result of `fit_poisson_glm_batch`
    for a single response, up to floating-point summation order.

    `y`: `(n,)` counts. Returns a `GlmFitBatchResult` with `coefs` `(p,)` and
    `fitted_values` `(n,)`. See docs/design.md, "Indicator columns in the IRLS".
    """
    with one_blas_thread():
        return _fit_structured(design, y, "poisson", eps=eps, maxit=maxit, mu_floor=mu_floor)


def fit_binomial_glm_structured(
    design: StructuredDesign,
    y: np.ndarray,
    *,
    eps: float = _EPS,
    maxit: int = _MAXIT,
) -> GlmFitBatchResult:
    """One logistic GLM by IRLS on a `StructuredDesign`; the result of
    `fit_binomial_glm_batch` for a single 0/1 response, up to floating-point
    summation order. See docs/design.md, "Indicator columns in the IRLS".
    """
    with one_blas_thread():
        return _fit_structured(design, y, "binomial", eps=eps, maxit=maxit, mu_floor=_MU_FLOOR)
