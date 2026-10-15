"""What a run's floating-point results depend on, beyond its inputs.

`environment_fingerprint` records the versions, BLAS libraries and CPU a
process computes with; `numeric_environment_key` reduces it to the part that
selects arithmetic kernels, so results saved by one process can be checked
before another process reuses them. Two processes with the same key run the
same kernels; the CPU model is recorded but not part of the key, so two CPU
types that dispatch the same kernels share it. See docs/design.md, "The trans
background".
"""

from __future__ import annotations

import hashlib
import json
import os
import platform

__all__ = ["environment_fingerprint", "numeric_environment_key"]


def _pysceptre_version() -> str:
    from .. import __version__

    return __version__


def _numba_version() -> str | None:
    try:
        import numba
    except Exception:
        return None
    return str(numba.__version__)


def _blas_libraries() -> list[dict]:
    """The loaded BLAS libraries, as threadpoolctl reports them, in a fixed order."""
    # numpy loads its BLAS on import; scipy's is loaded with scipy.linalg, so
    # import it here to list the same libraries whatever ran before.
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_info

    entries = [
        {
            "internal_api": info.get("internal_api"),
            "version": info.get("version"),
            "architecture": info.get("architecture"),
            "filename": os.path.basename(info.get("filepath") or ""),
        }
        for info in threadpool_info()
        if info.get("user_api") == "blas"
    ]
    return sorted(entries, key=lambda e: json.dumps(e, sort_keys=True))


def _numpy_cpu_features() -> list[str] | None:
    """The CPU features numpy's runtime dispatch found enabled, sorted."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__
        except ImportError:
            return None
    return sorted(name for name, enabled in __cpu_features__.items() if enabled)


def _cpu_model() -> str | None:
    """The first `model name` of /proc/cpuinfo, or None where there is none."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                key, sep, value = line.partition(":")
                if sep and key.strip() == "model name":
                    return value.strip()
    except OSError:
        return None
    return None


def environment_fingerprint() -> dict:
    """The current process's numeric environment, as a JSON-serializable dict.

    Returns:
        A dict with `pysceptre`, `python`, `numpy`, `scipy` and `numba`
        versions (`numba` None when it cannot be imported); `blas`, one entry
        per loaded BLAS library with its `internal_api`, `version`,
        `architecture` and the `filename` of its path; `numpy_cpu_features`,
        the CPU features numpy's dispatch found enabled (None when numpy does
        not expose them); `cpu_model` from /proc/cpuinfo (None when
        unavailable); and `have_numba`, whether the CRT sampler runs its numba
        kernel.
    """
    import numpy as np
    import scipy

    from ..crt import sampler

    return {
        "pysceptre": _pysceptre_version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": _numba_version(),
        "blas": _blas_libraries(),
        "numpy_cpu_features": _numpy_cpu_features(),
        "cpu_model": _cpu_model(),
        "have_numba": bool(sampler._HAVE_NUMBA),
    }


def _numeric_components(fp: dict) -> dict:
    """The parts of an `environment_fingerprint` that `numeric_environment_key` hashes.

    numpy, scipy and numba versions; each BLAS library's `internal_api`,
    `version` and `architecture`; the enabled AVX-512 feature names, which
    select among numpy's dispatched kernels; and `have_numba`. Not the CPU
    model, the Python or pysceptre version, nor the BLAS file names.
    """
    blas = sorted(
        ([e.get("internal_api"), e.get("version"), e.get("architecture")] for e in fp["blas"]),
        key=json.dumps,
    )
    avx512 = sorted(f for f in (fp.get("numpy_cpu_features") or []) if f.startswith("AVX512"))
    return {
        "numpy": fp["numpy"],
        "scipy": fp["scipy"],
        "numba": fp["numba"],
        "blas": blas,
        "avx512": avx512,
        "have_numba": bool(fp["have_numba"]),
    }


def numeric_environment_key(fp: dict) -> str:
    """sha256 hex digest of the parts of `fp` that select arithmetic kernels.

    Equal keys mean the same library versions, BLAS kernels and numpy CPU
    dispatch, which is what a bitwise comparison of results needs. `fp` is an
    `environment_fingerprint`, fresh or read back from JSON.
    """
    encoded = json.dumps(_numeric_components(fp), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
