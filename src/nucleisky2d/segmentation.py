"""Compatibility import path for the public 2D segmentation API."""

from __future__ import annotations

from nucleisky.nucleisky2d.segmentation import (
    _remove_small_holes_compat,
    _remove_small_objects_compat,
)
from nucleisky.nucleisky2d.segmentation import *  # noqa: F401,F403
