"""Compatibility import path for the public 3D API.

The implementation lives in :mod:`nucleisky.nucleisky3d`; this package keeps
``import nucleisky3d`` and ``from nucleisky3d.pipeline import ...`` working for
notebooks and user scripts.
"""

from __future__ import annotations

from nucleisky import nucleisky3d as _impl
from nucleisky.nucleisky3d import *  # noqa: F401,F403

__all__ = list(getattr(_impl, "__all__", []))
__path__ = _impl.__path__
