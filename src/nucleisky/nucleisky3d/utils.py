"""utils.py Small, generic utilities for 3D workflows."""

import math
from typing import Any
import zlib
import numpy as np


def _stable_u32(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    return zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF


def _is_finite_number(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _rel_err(a, b):
    a = float(a)
    b = float(b)
    return abs(a - b) / max(abs(b), 1e-12)


def compute_min_inliers_stable(
    Nc_eff,
    min_inliers_abs,
    min_inliers_frac,
    *,
    hard_floor=3,
    cap_frac=0.80,
):
    """Robustly choose the minimum inlier count from absolute and fractional rules."""
    Nc_eff = int(max(0, Nc_eff))
    min_abs = int(max(0, min_inliers_abs))
    min_frac = int(math.floor(float(min_inliers_frac) * Nc_eff + 0.5))

    if Nc_eff <= 0:
        return int(max(hard_floor, min_abs, min_frac))

    cap = int(max(hard_floor, math.floor(float(cap_frac) * Nc_eff)))
    need = int(max(hard_floor, min_abs, min_frac))
    return int(min(need, cap))
