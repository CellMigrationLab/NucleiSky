"""preprocess.py Preprocessing utilities for 3D volumes."""

from __future__ import annotations

import numpy as np
from skimage.transform import rescale
from scipy.ndimage import zoom


def _normalize_voxel_size_zyx(voxel_size_um, name: str) -> np.ndarray:
    if isinstance(voxel_size_um, (tuple, list, np.ndarray)):
        if len(voxel_size_um) != 3:
            raise ValueError(f"{name} must be a float or length-3 tuple. Got {voxel_size_um}")
        vox = np.asarray(voxel_size_um, dtype=float).reshape(3,)
    else:
        vox = np.asarray([float(voxel_size_um)] * 3, dtype=float)

    if not np.isfinite(vox).all() or np.any(vox <= 0):
        raise ValueError(f"{name} must contain positive finite values. Got {voxel_size_um}")
    return vox


def choose_common_target_um_per_voxel(
    voxel_size_full_um_zyx,
    voxel_size_crop_um_zyx,
    strategy="coarsest",
    manual_target_um=None,
) -> np.ndarray:
    """Choose a common target voxel size (µm/voxel) for pre-segmentation scaling."""
    vf = _normalize_voxel_size_zyx(voxel_size_full_um_zyx, "voxel_size_full_um_zyx")
    vc = _normalize_voxel_size_zyx(voxel_size_crop_um_zyx, "voxel_size_crop_um_zyx")

    strat = str(strategy).strip().lower()
    alias = {
        "match_crop_to_full": "match_full",
        "match_full_to_crop": "match_crop",
        "max": "coarsest",
        "min": "finest",
        "manual": "custom",
    }
    strat = alias.get(strat, strat)

    if strat in ("match_full", "full", "keep_full"):
        return vf
    if strat in ("match_crop", "crop", "keep_crop"):
        return vc
    if strat in ("coarsest", "coarse"):
        return np.maximum(vf, vc)
    if strat in ("finest", "fine"):
        return np.minimum(vf, vc)
    if strat in ("mean", "average", "avg"):
        return 0.5 * (vf + vc)
    if strat in ("custom", "user"):
        if manual_target_um is None:
            raise ValueError("strategy='custom' requires manual_target_um > 0.")
        return _normalize_voxel_size_zyx(manual_target_um, "manual_target_um")

    raise ValueError(
        f"Unknown strategy='{strategy}'. "
        "Use one of: 'coarsest', 'finest', 'match_full', 'match_crop', 'custom'."
    )


def rescale_to_target_um_per_voxel(
    volume_zyx,
    current_um_per_voxel_zyx,
    target_um_per_voxel_zyx,
    *,
    order=1,
    dtype_out=np.float32,
    max_upsample=4.0,
    min_downsample=0.25,
):
    """Rescale a 3D volume so its effective voxel size becomes target_um_per_voxel_zyx."""
    x = np.asarray(volume_zyx)
    if x.ndim != 3:
        raise ValueError(f"volume_zyx must be a 3D array. Got shape={x.shape}")

    cur = _normalize_voxel_size_zyx(current_um_per_voxel_zyx, "current_um_per_voxel_zyx")
    tgt = _normalize_voxel_size_zyx(target_um_per_voxel_zyx, "target_um_per_voxel_zyx")

    max_upsample = float(max_upsample)
    min_downsample = float(min_downsample)
    if max_upsample < 1.0:
        raise ValueError("max_upsample must be >= 1.0.")
    if not (0 < min_downsample <= 1.0):
        raise ValueError("min_downsample must be in (0, 1].")

    s_req = cur / tgt
    if np.allclose(s_req, 1.0, rtol=1e-6, atol=1e-8):
        return x.astype(dtype_out, copy=False), np.ones(3, dtype=float)

    s = np.clip(s_req, min_downsample, max_upsample).astype(float)
    
    # FIX: Re-introduced anti-aliasing via skimage.transform.rescale (supports 3D)
    anti_alias = bool(np.any(s < 1.0) and int(order) > 0)
    
    x_rs = rescale(
        x,
        scale=tuple(float(v) for v in s),
        order=int(order),
        preserve_range=True,
        anti_aliasing=anti_alias,
        channel_axis=None,
    )
    return x_rs.astype(dtype_out, copy=False), s


def scale_normalize_pair_for_segmentation(
    img_full,
    img_crop,
    voxel_size_full_um_zyx,
    voxel_size_crop_um_zyx,
    *,
    strategy="coarsest",
    manual_target_um=None,
    max_upsample=4.0,
    min_downsample=0.25,
    order=1,
    dtype_out=np.float32,
):
    """Rescale full+crop volumes before segmentation so nuclei have comparable voxel-scale size."""
    img_full = np.asarray(img_full)
    img_crop = np.asarray(img_crop)
    if img_full.ndim != 3 or img_crop.ndim != 3:
        raise ValueError("img_full and img_crop must both be 3D arrays.")

    vf = _normalize_voxel_size_zyx(voxel_size_full_um_zyx, "voxel_size_full_um_zyx")
    vc = _normalize_voxel_size_zyx(voxel_size_crop_um_zyx, "voxel_size_crop_um_zyx")

    target_req = choose_common_target_um_per_voxel(
        vf, vc, strategy=strategy, manual_target_um=manual_target_um,
    )

    img_full_seg, sf_full = rescale_to_target_um_per_voxel(
        img_full, vf, target_req,
        order=order, dtype_out=dtype_out,
        max_upsample=max_upsample, min_downsample=min_downsample,
    )
    img_crop_seg, sf_crop = rescale_to_target_um_per_voxel(
        img_crop, vc, target_req,
        order=order, dtype_out=dtype_out,
        max_upsample=max_upsample, min_downsample=min_downsample,
    )

    voxel_full_seg = vf / sf_full
    voxel_crop_seg = vc / sf_crop

    return (
        img_full_seg,
        img_crop_seg,
        tuple(float(v) for v in voxel_full_seg),
        tuple(float(v) for v in voxel_crop_seg),
        tuple(float(v) for v in sf_full),
        tuple(float(v) for v in sf_crop),
        tuple(float(v) for v in target_req),
    )


def rescale_label_mask_nearest_3d(mask3d, scale_factor_zyx, *, output_shape=None):
    """
    Rescale a 3D label mask using nearest-neighbor interpolation.
    """
    m = np.asarray(mask3d)
    if m.ndim != 3:
        raise ValueError("rescale_label_mask_nearest_3d expects a 3D label volume (ZYX).")

    sf = _normalize_voxel_size_zyx(scale_factor_zyx, "scale_factor_zyx")

    if output_shape is not None:
        out_z, out_y, out_x = map(int, output_shape[:3])
        if out_z <= 0 or out_y <= 0 or out_x <= 0:
            raise ValueError("output_shape must be positive.")
        
        zf = (out_z / m.shape[0], out_y / m.shape[1], out_x / m.shape[2])
        out = zoom(m, zoom=zf, order=0, mode="nearest")
        out = out[:out_z, :out_y, :out_x]
        
        if out.shape != (out_z, out_y, out_x):
            pad_z = max(0, out_z - out.shape[0])
            pad_y = max(0, out_y - out.shape[1])
            pad_x = max(0, out_x - out.shape[2])
            out = np.pad(out, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        return out.astype(m.dtype, copy=False)

    out = zoom(m, zoom=tuple(sf), order=0, mode="nearest")
    return out.astype(m.dtype, copy=False)


def ij_percentile_normalize(volume: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """ImageJ-style percentile normalization."""
    volume = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(volume, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    volume_scaled = (volume - lo) / (hi - lo)
    volume_scaled = np.clip(volume_scaled, 0, 1)
    return volume_scaled


def require_3d_image(arr: np.ndarray, *, label: str):
    """Validates that the input is a 3D ZYX grayscale volume."""
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"{label} must be a 3D volume (ZYX) at this stage. Got shape={a.shape} (ndim={a.ndim}).")
    if a.size == 0:
        raise ValueError(f"{label} is empty. shape={a.shape}")
    if not np.isfinite(a).any():
        raise ValueError(f"{label} has no finite pixels (all NaN/Inf).")
    return a.astype(np.float32, copy=False)


def require_3d_label_mask(mask, *, label, expected_shape=None):
    """Validates that the input is a 3D ZYX integer label volume."""
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError(f"{label} must be a 3D label volume. Got shape={m.shape} (ndim={m.ndim}).")
    if expected_shape is not None and tuple(m.shape) != tuple(expected_shape):
        raise ValueError(f"{label} shape {m.shape} does not match expected shape {expected_shape}.")
    if not np.issubdtype(m.dtype, np.integer):
        print(f"⚠️ {label} dtype is {m.dtype}, casting to int32.")
        m = m.astype(np.int32, copy=False)
    if np.min(m) < 0:
        raise ValueError(f"{label} contains negative labels (min={int(np.min(m))}). Labels must be >= 0.")
    if int(np.max(m)) == 0:
        raise ValueError(f"{label} contains no objects (max label = 0).")
    return m.astype(np.int32, copy=False)
