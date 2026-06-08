"""Compatibility import path for the public 2D API.

The implementation lives in :mod:`nucleisky.nucleisky2d`; this package keeps
``import nucleisky2d`` and ``from nucleisky2d.pipeline import ...`` working for
notebooks and user scripts.
"""

from __future__ import annotations

from nucleisky import nucleisky2d as _impl
from nucleisky.nucleisky2d import *  # noqa: F401,F403

__all__ = list(getattr(_impl, "__all__", []))
__path__ = _impl.__path__
