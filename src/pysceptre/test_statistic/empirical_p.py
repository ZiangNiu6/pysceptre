"""Port of `compute_empirical_p_value` (shared_low_level_functions.cpp).

side: -1 = left, 0 = both, 1 = right (matches sceptre's side_code convention).
"""

from __future__ import annotations

import numpy as np


def compute_empirical_p_value(null_statistics: np.ndarray, z_orig: float, side: int) -> float:
    null_statistics = np.asarray(null_statistics)
    B = null_statistics.size
    if side == -1:
        counter = np.sum(z_orig >= null_statistics)
        return (1.0 + counter) / (1.0 + B)
    if side == 1:
        counter = np.sum(z_orig <= null_statistics)
        return (1.0 + counter) / (1.0 + B)
    # two-sided
    return 2.0 * min(
        compute_empirical_p_value(null_statistics, z_orig, -1),
        compute_empirical_p_value(null_statistics, z_orig, 1),
    )


def empirical_p_from_counts(c_left: int, c_right: int, B: int, side: int) -> float:
    """`compute_empirical_p_value` from the two counts instead of the null statistics.

    `c_left` counts the nulls with `z_orig >= null` and `c_right` those with
    `z_orig <= null`; a NaN null counts on neither side. With the counts over
    all `B` nulls this equals `compute_empirical_p_value(nulls, z_orig, side)`
    exactly, float for float. With counts over the first nulls only and `B`
    the full number, it is a lower bound on that p-value, since the counts can
    only grow. See docs/design.md, "The trans background".
    """
    left = (1.0 + c_left) / (1.0 + B)
    right = (1.0 + c_right) / (1.0 + B)
    if side == -1:
        return float(left)
    if side == 1:
        return float(right)
    return float(2.0 * min(left, right))
