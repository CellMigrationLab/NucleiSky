
"""utils.py Small, generic utilities."""

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
    a = float(a); b = float(b)
    return abs(a - b) / max(abs(b), 1e-12)
