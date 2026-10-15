"""Width-1 gene fits computed once, saved, and reused exactly as the engine would compute them.

`fit_gene_cache` must return the fits the complement engine computes for
itself, bit for bit; `save_gene_cache` and `load_gene_cache` must round-trip
every float bit for bit; and `load_gene_cache` must refuse fits it cannot
vouch for: other data, a covariate matrix in another memory layout, other
engine settings, a gene in two files, or another numeric environment. The
environment fingerprint and its key are tested at the end. See
docs/design.md, "The trans background".
"""

from __future__ import annotations

import copy
import itertools
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pysceptre.crt import sampler
from pysceptre.pipeline import discovery, gene_cache
from pysceptre.pipeline.api import run_discovery_analysis
from pysceptre.pipeline.fingerprint import environment_fingerprint, numeric_environment_key
from pysceptre.pipeline.gene_cache import (
    GeneFitCache,
    fit_gene_cache,
    load_gene_cache,
    save_gene_cache,
)

N_CELLS = 3_000
SEED = 5
# Genes the pairs name, deliberately not in matrix order; g1 and g3 are untested.
TESTED = ["g7", "g2", "g0", "g6", "g4", "g5"]


@pytest.fixture(autouse=True)
def _quiet_degenerate_fits():
    """g6 and g7 are degenerate on purpose; their summary warning is expected."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="degenerate gene fits")
        yield


@pytest.fixture(scope="module")
def screen():
    """Eight genes over 3,000 cells: five negative binomial, one Poisson, two very sparse.

    The near-empty g6 has its dispersion clamped at the lower bound and g7,
    forty single counts, falls back to the pilot estimate, so every field of a
    fit varies across genes, including the type of `theta`.
    """
    rng = np.random.default_rng(20261014)
    batch = rng.integers(0, 2, N_CELLS)
    cov = np.column_stack([np.ones(N_CELLS), rng.normal(0.0, 0.4, N_CELLS), batch.astype(float)])
    mu = np.exp(cov @ np.array([0.0, 0.7, 0.3]))
    base = np.array([1.0, 3.0, 0.5, 8.0, 2.0])
    theta = np.array([3.0, 10.0, 1.5, 6.0, 20.0])
    nb = rng.negative_binomial(
        theta[:, None], theta[:, None] / (theta[:, None] + base[:, None] * mu[None, :])
    )
    poisson = rng.poisson(4.0 * mu)[None, :]
    near_empty = np.zeros((1, N_CELLS))
    near_empty[0, rng.choice(N_CELLS, 3, replace=False)] = [1, 2, 1]
    sparse_ones = np.zeros((1, N_CELLS))
    sparse_ones[0, rng.choice(N_CELLS, 40, replace=False)] = 1
    resp = np.vstack([nb, poisson, near_empty, sparse_ones]).astype(float)
    genes = [f"g{i}" for i in range(resp.shape[0])]
    cells = {
        f"t{j}": np.sort(rng.choice(N_CELLS, int(rng.integers(40, 81)), replace=False))
        for j in range(4)
    }
    pairs = pd.DataFrame(
        [(g, t) for g in TESTED for t in cells], columns=["response_id", "grna_target"]
    )
    return resp, genes, cov, cells, pairs


def _bits(x) -> bytes:
    return np.asarray(x, dtype=np.float64).tobytes()


def assert_same_fit(a, b, gene=""):
    """Every field equal, floats bit for bit (so NaN only matches NaN with the same bits)."""
    assert a.fitted_coefs.dtype == b.fitted_coefs.dtype == np.float64, gene
    assert a.fitted_coefs.shape == b.fitted_coefs.shape, gene
    assert _bits(a.fitted_coefs) == _bits(b.fitted_coefs), gene
    for field in ("theta", "min_eigenvalue", "max_eigenvalue"):
        assert _bits(getattr(a, field)) == _bits(getattr(b, field)), (gene, field)
    for field in ("theta_method", "theta_clamped", "glm_converged", "n_covariates"):
        assert getattr(a, field) == getattr(b, field), (gene, field)


def assert_same_fits(a: dict, b: dict):
    assert list(a) == list(b)
    for gene in a:
        assert_same_fit(a[gene], b[gene], gene)


def _engine_gene_fits(monkeypatch, screen):
    """What `fit_all_genes` returns inside `run_discovery_analysis`, and the result."""
    resp, genes, cov, cells, pairs = screen
    captured = []
    original = discovery.fit_all_genes

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append(out)
        return out

    with monkeypatch.context() as m:
        m.setattr(discovery, "fit_all_genes", spy)
        result = run_discovery_analysis(resp, genes, cov, cells, pairs, seed=SEED)
    assert len(captured) == 1
    return captured[0], result


def _rewrite(src, dst, drop=(), **changes):
    """Copy a saved cache with fingerprint entries dropped or replaced, as a crafted file would be."""
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fp = json.loads(str(arrays["fingerprint"][()]))
    for key in drop:
        del fp[key]
    fp.update(changes)
    arrays["fingerprint"] = np.array(json.dumps(fp))
    with open(dst, "wb") as f:
        np.savez(f, **arrays)
    return dst


# --- the cache is the engine's fits ---------------------------------------


def test_the_cache_holds_the_fits_the_engine_computes(screen, monkeypatch):
    resp, genes, cov, _, _ = screen
    engine_fits, _ = _engine_gene_fits(monkeypatch, screen)
    cache = fit_gene_cache(resp, genes, cov, genes=TESTED)
    assert isinstance(cache, GeneFitCache)
    # The engine fits only the tested genes, in matrix order, and so does the cache.
    assert list(engine_fits) == [g for g in genes if g in TESTED]
    assert_same_fits(cache.fits, engine_fits)
    # The data exercise the degenerate fields, not only the healthy path.
    assert cache.fits["g6"].theta_clamped
    assert cache.fits["g7"].theta_method != 1
    assert not cache.fits["g0"].theta_clamped and cache.fits["g0"].theta_method == 1


def test_fitting_all_genes_gives_each_the_same_fit_as_a_subset(screen):
    resp, genes, cov, _, _ = screen
    everything = fit_gene_cache(resp, genes, cov)
    subset = fit_gene_cache(resp, genes, cov, genes=TESTED)
    assert list(everything.fits) == genes
    assert_same_fits(subset.fits, {g: everything.fits[g] for g in subset.fits})


def test_worker_count_does_not_change_a_fit(screen):
    resp, genes, cov, _, _ = screen
    one = fit_gene_cache(resp, genes, cov, n_jobs=1)
    three = fit_gene_cache(resp, genes, cov, n_jobs=3)
    assert_same_fits(one.fits, three.fits)
    assert one.fingerprint == three.fingerprint


def test_the_fingerprint_describes_the_fit(screen):
    resp, genes, cov, _, _ = screen
    cache = fit_gene_cache(resp, genes, cov, genes=TESTED)
    fp = cache.fingerprint
    assert fp["n_cells"] == N_CELLS
    assert fp["n_covariates"] == cov.shape[1]
    assert fp["dfr"] == N_CELLS - cov.shape[1]
    assert fp["batch_width"] == 1
    assert fp["theta_bounds"] == list(discovery._THETA_BOUNDS)
    assert fp["glm_solver"] == "outer"
    assert cov.flags.c_contiguous and fp["covariate_layout"] == "C"
    assert set(fp["count_digests"]) == set(TESTED)
    row = resp[genes.index("g7")]
    assert fp["count_digests"]["g7"]["nnz"] == int(np.count_nonzero(row))
    assert fp["count_digests"]["g7"]["sum"] == float(row.sum())
    assert fp["environment"] == environment_fingerprint()
    json.dumps(fp)  # plain JSON throughout


def test_the_count_digest_does_not_depend_on_storage(screen):
    """Dense, CSR (also unsorted with a stored zero), CSC and integer input are the same counts."""
    resp, genes, cov, _, _ = screen
    reference = fit_gene_cache(resp, genes, cov)

    data, indices, indptr = [], [], [0]
    for row in resp:
        nz = np.flatnonzero(row)[::-1]  # descending, so unsorted
        zero = np.flatnonzero(row == 0)[:1]  # one stored zero
        indices.extend([*nz.tolist(), *zero.tolist()])
        data.extend([*row[nz].tolist(), *[0.0] * zero.size])
        indptr.append(len(indices))
    messy = sparse.csr_matrix(
        (np.array(data), np.array(indices), np.array(indptr)), shape=resp.shape
    )
    forms = {
        "csr": sparse.csr_matrix(resp),
        "messy_csr": messy,
        "csc": sparse.csc_matrix(resp),
        "int": resp.astype(np.int64),
    }
    for name, matrix in forms.items():
        other = fit_gene_cache(matrix, genes, cov)
        assert_same_fits(other.fits, reference.fits)
        assert other.fingerprint == reference.fingerprint, name


def test_fit_gene_cache_refuses_what_the_engine_refuses(screen):
    resp, genes, cov, _, _ = screen
    with pytest.raises(ValueError, match="rank deficient"):
        fit_gene_cache(resp, genes, np.column_stack([cov, cov[:, 1]]))
    with pytest.raises(ValueError, match="not in gene_ids"):
        fit_gene_cache(resp, genes, cov, genes=["g0", "nope"])
    with pytest.raises(ValueError, match="gene_ids has"):
        fit_gene_cache(resp, genes[:-1], cov)


# --- saving and loading ---------------------------------------------------


def test_save_and_load_round_trip_bit_for_bit(screen, tmp_path):
    resp, genes, cov, _, _ = screen
    cache = fit_gene_cache(resp, genes, cov, genes=TESTED)
    path = tmp_path / "fits.npz.partial"
    save_gene_cache(cache, path)
    assert path.exists() and not (tmp_path / "fits.npz.partial.npz").exists()

    with np.load(path, allow_pickle=False) as z:
        assert z["gene_ids"].dtype.kind == "U"
        assert z["coefs"].dtype == np.float64 and z["coefs"].shape == (len(TESTED), cov.shape[1])
        assert json.loads(str(z["fingerprint"][()])) == cache.fingerprint

    loaded = load_gene_cache(path, response_matrix=resp, gene_ids=genes, covariate_matrix=cov)
    assert_same_fits(loaded.fits, cache.fits)
    assert loaded.fingerprint == cache.fingerprint
    # A loaded cache saves and loads again unchanged.
    again = tmp_path / "again.npz"
    save_gene_cache(loaded, again)
    reloaded = load_gene_cache(again)
    assert_same_fits(reloaded.fits, cache.fits)
    assert reloaded.fingerprint == cache.fingerprint


def test_loaded_fits_reproduce_the_engines_result(screen, monkeypatch, tmp_path):
    """Handing the engine loaded fits in place of its own changes no bit of the result."""
    resp, genes, cov, cells, pairs = screen
    _, plain = _engine_gene_fits(monkeypatch, screen)
    path = tmp_path / "fits.npz"
    save_gene_cache(fit_gene_cache(resp, genes, cov, genes=TESTED), path)
    loaded = load_gene_cache(path)

    def from_cache(response_matrix, gene_ids, *args, **kwargs):
        return {g: loaded.fits[g] for g in gene_ids}

    with monkeypatch.context() as m:
        m.setattr(discovery, "fit_all_genes", from_cache)
        reused = run_discovery_analysis(resp, genes, cov, cells, pairs, seed=SEED)
    pd.testing.assert_frame_equal(plain, reused, check_exact=True)


def test_subset_caches_merge_into_the_full_one(screen, tmp_path):
    resp, genes, cov, _, _ = screen
    full = fit_gene_cache(resp, genes, cov)
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    save_gene_cache(fit_gene_cache(resp, genes, cov, genes=genes[:3]), first)
    save_gene_cache(fit_gene_cache(resp, genes, cov, genes=genes[3:]), second)
    merged = load_gene_cache(
        [first, second], response_matrix=resp, gene_ids=genes, covariate_matrix=cov
    )
    assert_same_fits(merged.fits, full.fits)
    assert merged.fingerprint == full.fingerprint
    # In the order the files are given.
    assert list(load_gene_cache([second, str(first)]).fits) == genes[3:] + genes[:3]


def test_an_empty_cache_round_trips(screen, tmp_path):
    resp, genes, cov, _, _ = screen
    path = tmp_path / "empty.npz"
    save_gene_cache(fit_gene_cache(resp, genes, cov, genes=[]), path)
    loaded = load_gene_cache(path, response_matrix=resp, gene_ids=genes, covariate_matrix=cov)
    assert loaded.fits == {}


# --- refusals -------------------------------------------------------------


@pytest.fixture
def saved(screen, tmp_path):
    resp, genes, cov, _, _ = screen
    path = tmp_path / "fits.npz"
    save_gene_cache(fit_gene_cache(resp, genes, cov, genes=TESTED), path)
    return path


def test_load_refuses_a_changed_covariate_entry(screen, saved):
    resp, genes, cov, _, _ = screen
    changed = cov.copy()
    changed[17, 1] = np.nextafter(changed[17, 1], np.inf)
    with pytest.raises(ValueError, match="covariate_matrix differs"):
        load_gene_cache(saved, response_matrix=resp, gene_ids=genes, covariate_matrix=changed)
    with pytest.raises(ValueError, match="covariate_matrix has shape"):
        load_gene_cache(saved, covariate_matrix=cov[:, :2])


def test_the_covariate_layout_is_recorded_beside_the_values(screen):
    resp, genes, cov, _, _ = screen
    wide = np.column_stack([cov, cov])
    assert gene_cache._covariate_layout(cov) == "C"
    assert gene_cache._covariate_layout(np.asfortranarray(cov)) == "F"
    with pytest.raises(ValueError, match="strided view"):
        gene_cache._covariate_layout(wide[:, : cov.shape[1]])
    # One row or one column is both; it is "C".
    assert gene_cache._covariate_layout(np.asfortranarray(cov[:, :1])) == "C"
    c = fit_gene_cache(resp, genes, cov, genes=["g0"])
    f = fit_gene_cache(resp, genes, np.asfortranarray(cov), genes=["g0"])
    with pytest.raises(ValueError, match="strided view"):
        fit_gene_cache(resp, genes, wide[:, : cov.shape[1]], genes=["g0"])
    assert c.fingerprint["covariate_sha256"] == f.fingerprint["covariate_sha256"]
    assert [x.fingerprint["covariate_layout"] for x in (c, f)] == ["C", "F"]


def test_load_refuses_another_covariate_layout(screen, saved, tmp_path):
    resp, genes, cov, _, _ = screen
    other = np.asfortranarray(cov)
    assert np.array_equal(other, cov)
    with pytest.raises(ValueError, match="memory layout 'F'; .* a 'C' one"):
        load_gene_cache(saved, response_matrix=resp, gene_ids=genes, covariate_matrix=other)
    with pytest.raises(ValueError, match="strided view"):
        load_gene_cache(
            saved,
            response_matrix=resp,
            gene_ids=genes,
            covariate_matrix=np.column_stack([cov, cov])[:, : cov.shape[1]],
        )
    # Fits made from a column-major matrix load only with a column-major one.
    f_path = tmp_path / "fortran.npz"
    save_gene_cache(fit_gene_cache(resp, genes, np.asfortranarray(cov), genes=["g1"]), f_path)
    load_gene_cache(f_path, covariate_matrix=np.asfortranarray(cov))
    with pytest.raises(ValueError, match="memory layout 'C'; .* a 'F' one"):
        load_gene_cache(f_path, covariate_matrix=cov)
    with pytest.raises(ValueError, match="different data or settings: covariate_layout"):
        load_gene_cache([saved, f_path])


def test_load_refuses_a_cache_that_does_not_record_the_layout(saved, tmp_path):
    crafted = _rewrite(saved, tmp_path / "no_layout.npz", drop=("covariate_layout",))
    with pytest.raises(ValueError, match="memory layout .* refit them with fit_gene_cache"):
        load_gene_cache(crafted)


def test_the_data_check_refuses_a_fingerprint_it_cannot_use(screen):
    resp, genes, cov, _, _ = screen
    fp = fit_gene_cache(resp, genes, cov, genes=["g0"]).fingerprint
    partial = {k: v for k, v in fp.items() if k not in ("covariate_layout", "count_digests")}
    with pytest.raises(ValueError, match=r"lacks \['covariate_layout'\]"):
        gene_cache._check_data(partial, covariate_matrix=cov)
    with pytest.raises(ValueError, match=r"lacks \['count_digests'\]"):
        gene_cache._check_data(partial, response_matrix=resp, gene_ids=genes)
    with pytest.raises(ValueError, match="give both or neither"):
        gene_cache._check_data(fp, response_matrix=resp)
    # Only the genes asked about are checked.
    changed = resp.copy()
    changed[genes.index("g0"), 0] += 1
    gene_cache._check_data(fp, response_matrix=changed, gene_ids=genes, genes=[])
    with pytest.raises(ValueError, match=r"counts of 1 cached gene\(s\) differ"):
        gene_cache._check_data(fp, response_matrix=changed, gene_ids=genes)


def test_load_refuses_a_changed_count(screen, saved):
    resp, genes, cov, _, _ = screen
    changed = resp.copy()
    row = genes.index("g4")
    changed[row, int(np.flatnonzero(changed[row])[0])] += 1
    with pytest.raises(ValueError, match=r"counts of 1 cached gene\(s\) differ.*'g4'"):
        load_gene_cache(saved, response_matrix=changed, gene_ids=genes, covariate_matrix=cov)
    # An untested gene's counts are not the cache's business.
    changed = resp.copy()
    changed[genes.index("g1"), 0] += 1
    load_gene_cache(saved, response_matrix=changed, gene_ids=genes)


def test_load_refuses_other_cells_or_genes(screen, saved):
    resp, genes, _, _, _ = screen
    with pytest.raises(ValueError, match="response_matrix has"):
        load_gene_cache(saved, response_matrix=resp[:, :-1], gene_ids=genes)
    renamed = [g if g != "g2" else "other" for g in genes]
    with pytest.raises(ValueError, match=r"not in gene_ids.*'g2'"):
        load_gene_cache(saved, response_matrix=resp, gene_ids=renamed)
    with pytest.raises(ValueError, match="give both or neither"):
        load_gene_cache(saved, response_matrix=resp)


def test_load_refuses_a_batch_width_other_than_one(saved, tmp_path):
    crafted = _rewrite(saved, tmp_path / "width8.npz", batch_width=8)
    with pytest.raises(ValueError, match="batch width 8"):
        load_gene_cache(crafted)


def test_load_refuses_other_theta_bounds(saved, tmp_path, monkeypatch):
    crafted = _rewrite(saved, tmp_path / "bounds.npz", theta_bounds=[0.01, 100.0])
    with pytest.raises(ValueError, match="theta bounds"):
        load_gene_cache(crafted)
    # Or the engine's bounds moved since the fits were made.
    monkeypatch.setattr(discovery, "_THETA_BOUNDS", (0.001, 1000.0))
    with pytest.raises(ValueError, match="theta bounds"):
        load_gene_cache(saved)


def test_load_refuses_a_gene_in_two_files(screen, saved, tmp_path):
    resp, genes, cov, _, _ = screen
    other = tmp_path / "other.npz"
    save_gene_cache(fit_gene_cache(resp, genes, cov, genes=["g1", "g2"]), other)
    with pytest.raises(ValueError, match=r"more than one file, e\.g\. 'g2'"):
        load_gene_cache([saved, other])


def test_load_refuses_files_fit_on_different_data(screen, saved, tmp_path):
    resp, genes, cov, _, _ = screen
    scaled = cov.copy()
    scaled[:, 1] *= 2.0
    other = tmp_path / "other.npz"
    save_gene_cache(fit_gene_cache(resp, genes, scaled, genes=["g1"]), other)
    with pytest.raises(ValueError, match="different data or settings: covariate_sha256"):
        load_gene_cache([saved, other])


def test_load_refuses_what_save_gene_cache_did_not_write(tmp_path):
    pkl = tmp_path / "gene_fits.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"g0": {"theta": 1.0}}, f)
    with pytest.raises(ValueError, match="not a gene-fit cache"):
        load_gene_cache(pkl)
    npy = tmp_path / "coefs.npy"
    np.save(npy, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="not an .npz archive"):
        load_gene_cache(npy)
    other = tmp_path / "other.npz"
    np.savez(other, coefs=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="lacks the array"):
        load_gene_cache(other)
    with pytest.raises(ValueError, match="no gene-fit cache files"):
        load_gene_cache([])


# --- the numeric environment ----------------------------------------------


def test_load_refuses_another_numeric_environment(saved, monkeypatch):
    elsewhere = copy.deepcopy(environment_fingerprint())
    elsewhere["numpy"] = "0.0.0"
    monkeypatch.setattr(gene_cache, "environment_fingerprint", lambda: elsewhere)
    with pytest.raises(ValueError, match=r"another numeric environment \(numpy '.+' in the cache"):
        load_gene_cache(saved)
    assert len(load_gene_cache(saved, check_environment=False).fits) == len(TESTED)


def test_a_mismatched_environment_key_is_refused(saved, monkeypatch):
    keys = itertools.count()
    monkeypatch.setattr(gene_cache, "numeric_environment_key", lambda fp: f"key{next(keys)}")
    with pytest.raises(ValueError, match="numeric environment keys differ"):
        load_gene_cache(saved)
    load_gene_cache(saved, check_environment=False)


def test_another_cpu_model_with_the_same_kernels_is_accepted(saved, monkeypatch):
    elsewhere = copy.deepcopy(environment_fingerprint())
    elsewhere["cpu_model"] = "Some Other CPU"
    elsewhere["python"] = "3.99.0"
    elsewhere["pysceptre"] = "9.9.9"
    monkeypatch.setattr(gene_cache, "environment_fingerprint", lambda: elsewhere)
    assert len(load_gene_cache(saved).fits) == len(TESTED)


def test_the_environment_fingerprint_fields():
    fp = environment_fingerprint()
    assert set(fp) == {
        "pysceptre",
        "python",
        "numpy",
        "scipy",
        "numba",
        "blas",
        "numpy_cpu_features",
        "cpu_model",
        "have_numba",
    }
    assert fp["numpy"] == np.__version__
    assert fp["have_numba"] is bool(sampler._HAVE_NUMBA)
    assert all(
        set(e) == {"internal_api", "version", "architecture", "filename"} for e in fp["blas"]
    )
    assert all("/" not in e["filename"] for e in fp["blas"])
    assert fp == environment_fingerprint()
    assert json.loads(json.dumps(fp)) == fp


def test_the_numeric_key_follows_kernels_not_the_cpu_model():
    fp = environment_fingerprint()
    key = numeric_environment_key(fp)
    assert numeric_environment_key(json.loads(json.dumps(fp))) == key

    def changed(**edits):
        other = copy.deepcopy(fp)
        other.update(edits)
        return numeric_environment_key(other)

    same = [
        changed(cpu_model="AMD EPYC 7502 32-Core Processor"),
        changed(cpu_model=None),
        changed(python="3.10.0", pysceptre="0.0.0"),
        changed(blas=[{**e, "filename": "renamed.so"} for e in fp["blas"]]),
        changed(blas=list(reversed(fp["blas"]))),
    ]
    assert all(k == key for k in same)

    features = [f for f in fp["numpy_cpu_features"] or [] if not f.startswith("AVX512")]
    different = [
        changed(numpy="0.0.0"),
        changed(scipy="0.0.0"),
        changed(numba="0.0.0"),
        changed(have_numba=not fp["have_numba"]),
        changed(blas=[{**e, "architecture": "SomethingElse"} for e in fp["blas"]]),
        changed(blas=[{**e, "version": "0.0.0"} for e in fp["blas"]]),
        changed(numpy_cpu_features=features + ["AVX512F", "AVX512_SKX"]),
        changed(numpy_cpu_features=features),
    ]
    if not any(f.startswith("AVX512") for f in fp["numpy_cpu_features"] or []):
        different.pop()  # removing AVX-512 changes nothing where there was none
    assert key not in different


# --- the solver -------------------------------------------------------------


def test_the_solver_is_recorded_and_checked(screen, saved, tmp_path):
    resp, genes, cov, _, _ = screen
    structured = tmp_path / "structured.npz"
    save_gene_cache(
        fit_gene_cache(resp, genes, cov, genes=["g1"], glm_solver="structured"), structured
    )
    assert load_gene_cache(structured).fingerprint["glm_solver"] == "structured"
    assert load_gene_cache(saved, glm_solver="outer").fingerprint["glm_solver"] == "outer"
    with pytest.raises(ValueError, match="glm_solver='structured', not 'outer'"):
        load_gene_cache(structured, glm_solver="outer")
    with pytest.raises(ValueError, match="different data or settings"):
        load_gene_cache([saved, structured])


def test_a_cache_without_its_solver_is_refused(saved, tmp_path):
    with np.load(saved, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fp = json.loads(str(arrays["fingerprint"][()]))
    del fp["glm_solver"]
    arrays["fingerprint"] = np.array(json.dumps(fp))
    old = tmp_path / "old.npz"
    with open(old, "wb") as f:
        np.savez(f, **arrays)
    with pytest.raises(ValueError, match="lacks.*glm_solver"):
        load_gene_cache(old)


def test_a_fingerprint_that_is_not_an_object_is_refused(saved, tmp_path):
    with np.load(saved, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["fingerprint"] = np.array(json.dumps([1, 2]))
    bad = tmp_path / "bad.npz"
    with open(bad, "wb") as f:
        np.savez(f, **arrays)
    with pytest.raises(ValueError, match="not a gene-fit cache"):
        load_gene_cache(bad)
