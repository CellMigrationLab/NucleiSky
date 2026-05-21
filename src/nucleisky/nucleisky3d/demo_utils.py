"""Notebook-only demo/synthetic utilities for 3D workflows."""

from __future__ import annotations

from typing import Iterable, Tuple
import warnings

import numpy as np
from scipy.ndimage import map_coordinates


def _normalize_voxel_size(voxel_size_um: float | Iterable[float]) -> np.ndarray:
    if isinstance(voxel_size_um, (tuple, list, np.ndarray)):
        if len(voxel_size_um) != 3:
            raise ValueError("voxel_size_um must be a float or length-3 tuple.")
        vox = np.asarray(voxel_size_um, dtype=float).reshape(3,)
    else:
        vox = np.asarray([float(voxel_size_um)] * 3, dtype=float)

    if not np.isfinite(vox).all() or np.any(vox <= 0):
        raise ValueError("voxel_size_um must contain positive finite values.")
    return vox


def _random_z_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a random rotation matrix strictly around the Z-axis (XY plane)."""
    theta = rng.uniform(0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    
    # Coordinates are in (Z, Y, X) order.
    # Z is unchanged. Y and X rotate.
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )
    return R


def generate_random_subvolume_3d(
    img_full: np.ndarray,
    crop_shape_zyx: Tuple[int, int, int],
    scale_range: Tuple[float, float],
    voxel_size_um: float | Iterable[float],
    rng: np.random.Generator | None = None,
):
    """
    Generate a synthetic 3D crop with known (scale, R, t) using trilinear interpolation.

    Notes
    -----
    Sampling geometry (crop shape, random center, rotation and scale) is computed in
    voxel/pixel index space. ``voxel_size_um`` is only used to report physical units
    in the return values (``crop_voxel_size_um`` and ``ground_truth['t']``).
    If the full volume is smaller than the strict in-bounds requirement, sampling still
    proceeds (using reflect padding at boundaries) and a ``RuntimeWarning`` is emitted.

    Parameters
    ----------
    img_full : np.ndarray
        Full reference volume in (Z, Y, X).
    crop_shape_zyx : tuple
        Output crop shape (Z, Y, X) in voxels.
    scale_range : tuple
        Uniform scale factor range (min, max) relative to the full volume.
    voxel_size_um : float or tuple
        Physical voxel size of the full volume (µm) in (Z, Y, X).
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    crop : np.ndarray
        Synthetic crop volume (Z, Y, X).
    crop_voxel_size_um : np.ndarray
        Voxel size for the crop in µm (Z, Y, X).
    ground_truth : dict
        Dictionary containing {"scale", "R", "t"} in µm space.
    """
    if rng is None:
        rng = np.random.default_rng()

    full = np.asarray(img_full)
    if full.ndim != 3:
        raise ValueError(f"img_full must be 3D (Z,Y,X). Got shape={full.shape}")

    scale_min, scale_max = map(float, scale_range)
    if not (0 < scale_min < scale_max):
        raise ValueError("scale_range must be (min, max) with 0 < min < max.")

    scale = float(rng.uniform(scale_min, scale_max))
    
    # Use the constrained Z-axis rotation here
    R = _random_z_rotation_matrix(rng)

    crop_z, crop_y, crop_x = (int(v) for v in crop_shape_zyx)
    if crop_z <= 0 or crop_y <= 0 or crop_x <= 0:
        raise ValueError("crop_shape_zyx must contain positive integers.")

    half_diag = np.linalg.norm([crop_z, crop_y, crop_x]) / (2.0 * scale_min)
    margin_target = int(np.ceil(half_diag))

    full_z, full_y, full_x = full.shape
    margin_max = int(min((full_z - 1) // 2, (full_y - 1) // 2, (full_x - 1) // 2))
    margin = min(margin_target, margin_max)

    if margin < margin_target:
        warnings.warn(
            "Full volume is smaller than the strict in-bounds requirement for the chosen "
            "crop/scale/rotation range. Sampling will still proceed and out-of-bounds "
            "coordinates are handled with reflect padding. "
            f"full shape={full.shape}, requested margin={margin_target}, used margin={margin}. "
            "crop_shape_zyx is interpreted in voxels.",
            RuntimeWarning,
            stacklevel=2,
        )

    if margin > 0:
        cz = int(rng.integers(margin, full_z - margin))
        cy = int(rng.integers(margin, full_y - margin))
        cx = int(rng.integers(margin, full_x - margin))
    else:
        # Very small volumes (axis length <= 2): allow any center index.
        cz = int(rng.integers(0, full_z))
        cy = int(rng.integers(0, full_y))
        cx = int(rng.integers(0, full_x))

    zz, yy, xx = np.mgrid[0:crop_z, 0:crop_y, 0:crop_x]
    z0 = (crop_z - 1) / 2.0
    y0 = (crop_y - 1) / 2.0
    x0 = (crop_x - 1) / 2.0

    zc = (zz - z0).reshape(-1)
    yc = (yy - y0).reshape(-1)
    xc = (xx - x0).reshape(-1)

    coords = np.stack([zc, yc, xc], axis=1) / scale
    coords_rot = coords @ R.T

    z_full = (cz + coords_rot[:, 0]).reshape(crop_z, crop_y, crop_x)
    y_full = (cy + coords_rot[:, 1]).reshape(crop_z, crop_y, crop_x)
    x_full = (cx + coords_rot[:, 2]).reshape(crop_z, crop_y, crop_x)

    crop = map_coordinates(
        full.astype(np.float32, copy=False),
        [z_full, y_full, x_full],
        order=1,
        mode="reflect",
    )
    crop = crop.astype(full.dtype, copy=False)

    voxel_size_um = _normalize_voxel_size(voxel_size_um)
    crop_voxel_size_um = voxel_size_um / scale

    t_um = np.array([cz, cy, cx], dtype=float) * voxel_size_um

    ground_truth = {
        "scale": scale,
        "R": R,
        "t": t_um,
    }

    return crop, crop_voxel_size_um, ground_truth
