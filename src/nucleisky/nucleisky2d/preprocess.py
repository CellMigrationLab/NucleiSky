
"""preprocess.py Pre-processing: rescaling, normalization, axes coercion, validation."""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from skimage.transform import rescale
from .types import BBox

from pathlib import Path
from scipy.ndimage import zoom

try:
    import torch
except Exception:
    torch = None

_ALLOWED_AX_LETTERS = set("TZCYX")


def choose_common_target_um_per_px(
    pixel_size_full_um,
    pixel_size_crop_um,
    strategy="coarsest",
    manual_target_um=None,
):
    """
    Decide the requested common target pixel size (µm/px) for pre-segmentation rescaling.

    Required options (case-insensitive):
      - "coarsest":   target = max(full_um_per_px, crop_um_per_px)  (mostly downsample)
      - "finest":     target = min(full_um_per_px, crop_um_per_px)  (upsamples coarser)
      - "match_full": target = full_um_per_px
      - "match_crop": target = crop_um_per_px
      - "custom":     target = manual_target_um (must be provided)

    Backward-compatible aliases supported:
      - "max" -> "coarsest"
      - "min" -> "finest"
      - "match_crop_to_full" -> "match_full"
      - "match_full_to_crop" -> "match_crop"
      - "manual" -> "custom"
    """


    pf = float(pixel_size_full_um)
    pc = float(pixel_size_crop_um)

    if not (np.isfinite(pf) and np.isfinite(pc)) or pf <= 0 or pc <= 0:
        raise ValueError("Pixel sizes must be positive finite floats (µm/px).")

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
        return pf
    if strat in ("match_crop", "crop", "keep_crop"):
        return pc
    if strat in ("coarsest", "coarse"):
        return max(pf, pc)
    if strat in ("finest", "fine"):
        return min(pf, pc)
    if strat in ("mean", "average", "avg"):
        return 0.5 * (pf + pc)
    if strat in ("custom", "user"):
        if manual_target_um is None or float(manual_target_um) <= 0:
            raise ValueError("strategy='custom' requires manual_target_um > 0.")
        return float(manual_target_um)

    raise ValueError(
        f"Unknown strategy='{strategy}'. "
        "Use one of: 'coarsest', 'finest', 'match_full', 'match_crop', 'custom'."
    )


def rescale_to_target_um_per_px(
    img2d,
    current_um_per_px,
    target_um_per_px,
    *,
    order=1,
    dtype_out=np.float32,
    max_upsample=4.0,
    min_downsample=0.25,
):
    """
    Rescale a 2D image so that its *requested* effective pixel size becomes target_um_per_px.

    Requested scale factor:
        s_req = current_um_per_px / target_um_per_px
      - s_req > 1: upsample
      - s_req < 1: downsample

    Safeguards:
      - max_upsample (>=1): cap on upsampling factor
      - min_downsample in (0,1]: floor on downsampling factor

    Returns
    -------
    img_rescaled : np.ndarray
    scale_factor_used : float
        The actual scale factor applied (may be clipped).
    """

    x = np.asarray(img2d)
    cur = float(current_um_per_px)
    tgt = float(target_um_per_px)

    if cur <= 0 or tgt <= 0:
        raise ValueError("Pixel sizes must be positive floats (µm/px).")

    max_upsample = float(max_upsample)
    min_downsample = float(min_downsample)
    if max_upsample < 1.0:
        raise ValueError("max_upsample must be >= 1.0.")
    if not (0 < min_downsample <= 1.0):
        raise ValueError("min_downsample must be in (0, 1].")

    s_req = cur / tgt
    if np.isclose(s_req, 1.0, rtol=1e-6, atol=1e-8):
        return x.astype(dtype_out, copy=False), 1.0

    s = float(np.clip(s_req, min_downsample, max_upsample))
    anti_alias = bool(s < 1.0 and int(order) > 0)

    x_rs = rescale(
        x,
        scale=s,
        order=int(order),
        preserve_range=True,
        anti_aliasing=anti_alias,
        channel_axis=None,
    )
    return x_rs.astype(dtype_out, copy=False), float(s)


def scale_normalize_pair_for_segmentation(
    img_full,
    img_crop,
    pixel_size_full_um,
    pixel_size_crop_um,
    *,
    strategy="coarsest",
    manual_target_um=None,
    max_upsample=4.0,
    min_downsample=0.25,
    order=1,
    dtype_out=np.float32,
):
    """
    Rescale full + crop BEFORE segmentation so nuclei have comparable pixel diameter.

    Policy choices (strategy):
      - "coarsest", "finest", "match_full", "match_crop", "custom"
        (see choose_common_target_um_per_px)

    Important:
      - The *requested* target may be modified effectively if scaling is clipped by
        max_upsample/min_downsample. We return the effective µm/px for each image.

    Returns
    -------
    img_full_seg, img_crop_seg,
    pixel_size_full_seg_um, pixel_size_crop_seg_um,
    scale_factor_full, scale_factor_crop,
    target_um_per_px_requested
    """

    img_full = np.asarray(img_full)
    img_crop = np.asarray(img_crop)

    pf = float(pixel_size_full_um)
    pc = float(pixel_size_crop_um)

    target_req = choose_common_target_um_per_px(
        pf, pc,
        strategy=strategy,
        manual_target_um=manual_target_um,
    )

    img_full_seg, sf_full = rescale_to_target_um_per_px(
        img_full, pf, target_req,
        order=order,
        dtype_out=dtype_out,
        max_upsample=max_upsample,
        min_downsample=min_downsample,
    )
    img_crop_seg, sf_crop = rescale_to_target_um_per_px(
        img_crop, pc, target_req,
        order=order,
        dtype_out=dtype_out,
        max_upsample=max_upsample,
        min_downsample=min_downsample,
    )

    # Effective pixel sizes after potentially clipped scaling:
    pix_full_seg = pf / float(sf_full)
    pix_crop_seg = pc / float(sf_crop)

    return (
        img_full_seg, img_crop_seg,
        float(pix_full_seg), float(pix_crop_seg),
        float(sf_full), float(sf_crop),
        float(target_req),
    )


def rescale_label_mask_nearest(mask2d, scale_factor, *, output_shape=None):
    """
    Rescale a 2D label mask using nearest-neighbor interpolation.

    Parameters
    ----------
    mask2d : (H,W) array-like
        Integer label image.
    scale_factor : float
        Multiplicative scale applied to (Y,X). (>1 upsample, <1 downsample)
    output_shape : tuple or None
        If provided, overrides scale_factor-based sizing and forces output to this shape.

    Returns
    -------
    mask_rs : (H2,W2) ndarray
    """

    m = np.asarray(mask2d)
    if m.ndim != 2:
        raise ValueError("rescale_label_mask_nearest expects a 2D label image.")
    sf = float(scale_factor)
    if sf <= 0:
        raise ValueError("scale_factor must be > 0.")

    if output_shape is not None:
        out_h, out_w = map(int, output_shape[:2])
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output_shape must be positive.")
        zf = (out_h / m.shape[0], out_w / m.shape[1])
        out = zoom(m, zoom=zf, order=0)
        out = out[:out_h, :out_w]
        if out.shape != (out_h, out_w):
            pad_h = out_h - out.shape[0]
            pad_w = out_w - out.shape[1]
            out = np.pad(out, ((0, max(0, pad_h)), (0, max(0, pad_w))),
                         mode="constant", constant_values=0)
        return out.astype(m.dtype, copy=False)

    out = zoom(m, zoom=(sf, sf), order=0)
    return out.astype(m.dtype, copy=False)


def ij_percentile_normalize(img, p_low=2, p_high=98):
    """
    ImageJ-style percentile normalization:
    - compute low/high percentiles on the raw image
    - linearly scale so p_low -> 0, p_high -> 1
    - clip outside [0,1]
    """
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (p_low, p_high))
    if hi <= lo:  # fallback in pathological cases
        return np.zeros_like(img, dtype=np.float32)
    img_scaled = (img - lo) / (hi - lo)
    img_scaled = np.clip(img_scaled, 0, 1)
    return img_scaled


def _as_array(img_or_path):
    """
    Accept either a numpy array or a path.

    Notebook hygiene:
      - Avoid relying on a mutable global name `imread`.
      - Always use tifffile.imread for TIFF stability.
    """
    from tifffile import imread as _tif_imread

    if isinstance(img_or_path, (str, Path)):
        return _tif_imread(str(img_or_path))
    return np.asarray(img_or_path)

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_2d(img, channel=0):
    x = np.asarray(img)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        # CHW vs HWC
        if x.shape[0] in (1, 2, 3, 4) and x.shape[1] > 16 and x.shape[2] > 16 and (x.shape[0] < x.shape[-1]):
            return x[int(channel)]
        return x[:, :, int(channel)]
    raise ValueError(f"Unsupported image shape for 2D extraction: {x.shape}")


def _coerce_label_2d(label_like, target="nuclei"):
    """
    Convert model output to a (H,W) int label image.
    Handles tuples/lists, dicts, torch tensors, and shapes like:
      (1,1,H,W), (1,H,W), (C,H,W), (H,W,C)
    """
    if isinstance(label_like, (tuple, list)) and len(label_like) > 0:
        label_like = label_like[0]

    if isinstance(label_like, dict):
        if target in label_like:
            label_like = label_like[target]
        else:
            label_like = next(iter(label_like.values()))

    arr = _to_numpy(label_like)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(np.int32, copy=False)

    if arr.ndim == 3:
        # channel-first vs channel-last
        if arr.shape[0] in (2, 3) and arr.shape[1] > 16 and arr.shape[2] > 16:
            c_axis = 0
        elif arr.shape[-1] in (2, 3) and arr.shape[0] > 16 and arr.shape[1] > 16:
            c_axis = 2
        else:
            arr2 = np.squeeze(arr)
            if arr2.ndim == 2:
                return arr2.astype(np.int32, copy=False)
            raise ValueError(f"Unexpected output shape after squeeze: {arr.shape}")

        idx = 0 if str(target).lower().startswith("nuc") else 1
        if c_axis == 0:
            if arr.shape[0] <= idx: idx = 0
            out = arr[idx]
        else:
            if arr.shape[-1] <= idx: idx = 0
            out = arr[..., idx]

        out = np.squeeze(out)
        if out.ndim != 2:
            raise ValueError(f"Could not coerce output to 2D. Final shape: {out.shape}")
        return out.astype(np.int32, copy=False)

    raise ValueError(f"Unexpected output shape: {arr.shape}")


def _coerce_tile_overlap(overlap, bsize: int) -> float:
    """
    overlap can be:
      - None -> default 0.1
      - fraction in [0,1) -> used directly
      - pixels >= 1 -> converted to fraction overlap/bsize
    """
    if overlap is None:
        return 0.1
    ov = float(overlap)
    if 0.0 <= ov < 1.0:
        return ov
    # pixels
    frac = ov / float(bsize)
    # keep strictly < 1
    if frac >= 1.0:
        return 0.1
    return frac


def _as_tzcyx(arr, axes: str):
    """
    Normalize to TZCYX.
    Supported: YX, CYX, ZYX, ZCYX, TYX, TCYX, TZYX, TZCYX
    """
    a = np.asarray(arr)
    ax = str(axes).upper().strip()

    if ax == "YX":
        if a.ndim != 2:
            raise ValueError(f"Expected YX (2D), got shape={a.shape}")
        return a[None, None, None, :, :]
    if ax == "CYX":
        if a.ndim != 3:
            raise ValueError(f"Expected CYX (3D), got shape={a.shape}")
        return a[None, None, :, :, :]
    if ax == "ZYX":
        if a.ndim != 3:
            raise ValueError(f"Expected ZYX (3D), got shape={a.shape}")
        return a[None, :, None, :, :]
    if ax == "ZCYX":
        if a.ndim != 4:
            raise ValueError(f"Expected ZCYX (4D), got shape={a.shape}")
        return a[None, :, :, :, :]
    if ax == "TYX":
        if a.ndim != 3:
            raise ValueError(f"Expected TYX (3D), got shape={a.shape}")
        return a[:, None, None, :, :]
    if ax == "TCYX":
        if a.ndim != 4:
            raise ValueError(f"Expected TCYX (4D), got shape={a.shape}")
        return a[:, None, :, :, :]
    if ax == "TZYX":
        if a.ndim != 4:
            raise ValueError(f"Expected TZYX (4D), got shape={a.shape}")
        return a[:, :, None, :, :]
    if ax == "TZCYX":
        if a.ndim != 5:
            raise ValueError(f"Expected TZCYX (5D), got shape={a.shape}")
        return a

    raise ValueError("Unsupported axes. Use one of: YX, CYX, ZYX, ZCYX, TYX, TCYX, TZYX, TZCYX")


def _from_tzcyx(tzcyx, axes: str):
    ax = str(axes).upper().strip()
    if ax == "YX":
        return tzcyx[0, 0, 0]
    if ax == "CYX":
        return tzcyx[0, 0]
    if ax == "ZYX":
        return tzcyx[0, :, 0]
    if ax == "ZCYX":
        return tzcyx[0]
    if ax == "TYX":
        return tzcyx[:, 0, 0]
    if ax == "TCYX":
        return tzcyx[:, 0]
    if ax == "TZYX":
        return tzcyx[:, :, 0]
    if ax == "TZCYX":
        return tzcyx
    raise ValueError(f"Unsupported axes='{axes}'")


def _broadcast_outer_dims(ref_tzcyx, mov_tzcyx):
    """
    Both are TZCYX. Broadcast T and Z if one side is 1.
    """
    ref = np.asarray(ref_tzcyx)
    mov = np.asarray(mov_tzcyx)

    Tr, Zr, Cr, Hr, Wr = ref.shape
    Tm, Zm, Cm, Hm, Wm = mov.shape

    # broadcast T
    T = max(Tr, Tm)
    if Tr != T:
        if Tr == 1:
            ref = np.repeat(ref, repeats=T, axis=0)
        else:
            raise ValueError(f"Cannot broadcast T: ref T={Tr}, mov T={Tm}")
    if Tm != T:
        if Tm == 1:
            mov = np.repeat(mov, repeats=T, axis=0)
        else:
            raise ValueError(f"Cannot broadcast T: ref T={Tr}, mov T={Tm}")

    # broadcast Z
    Tr, Zr, Cr, Hr, Wr = ref.shape
    Tm, Zm, Cm, Hm, Wm = mov.shape
    Z = max(Zr, Zm)
    if Zr != Z:
        if Zr == 1:
            ref = np.repeat(ref, repeats=Z, axis=1)
        else:
            raise ValueError(f"Cannot broadcast Z: ref Z={Zr}, mov Z={Zm}")
    if Zm != Z:
        if Zm == 1:
            mov = np.repeat(mov, repeats=Z, axis=1)
        else:
            raise ValueError(f"Cannot broadcast Z: ref Z={Zr}, mov Z={Zm}")

    return ref, mov


def _cast_like(x_float, ref_dtype):
    """
    Cast float array back to ref_dtype safely.
    - bool: threshold at 0.5
    - integer: round + clip to dtype range
    - float: cast without copy when possible
    """
    x = np.asarray(x_float)

    if np.issubdtype(ref_dtype, np.bool_):
        return (x > 0.5).astype(ref_dtype, copy=False)

    if np.issubdtype(ref_dtype, np.integer):
        info = np.iinfo(ref_dtype)
        x_round = np.rint(x)
        x_clip = np.clip(x_round, info.min, info.max)
        return x_clip.astype(ref_dtype, copy=False)

    return x.astype(ref_dtype, copy=False)


def _guess_axes_from_shape(shape):
    """Heuristic only; user must confirm."""
    ndim = len(shape)
    if ndim == 2:
        return "YX", ["YX"]
    if ndim == 3:
        a0, _, _ = shape
        cand = ["CYX", "ZYX", "TYX"]
        if a0 <= 8:  return "CYX", cand
        if a0 <= 50: return "TYX", cand
        return "ZYX", cand
    if ndim == 4:
        a0, a1, _, _ = shape
        cand = ["ZCYX", "TCYX", "TZYX"]
        if a1 <= 8:
            if a0 <= 50: return "TCYX", cand
            return "ZCYX", cand
        return "TZYX", cand
    if ndim == 5:
        return "TZCYX", ["TZCYX"]
    return "", []


def _normalize_to_tzcyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """
    Accept axes as any permutation/subset of {T,Z,C,Y,X} and expand missing dims as size-1.
    Enforces:
      - no duplicate letters
      - must include Y and X (we always warp in XY)
      - axes length must match arr.ndim
    """
    a = np.asarray(arr)
    ax = str(axes).upper().strip()

    if not ax:
        raise ValueError("Axes string is empty.")
    if not set(ax).issubset(_ALLOWED_AX_LETTERS):
        raise ValueError(f"Axes must use only TZCYX letters. Got: {axes!r}")
    if len(set(ax)) != len(ax):
        raise ValueError(f"Axes contains duplicate letters: {axes!r}")
    if "Y" not in ax or "X" not in ax:
        raise ValueError(f"Axes must include Y and X. Got axes={axes!r}")
    if len(ax) != a.ndim:
        raise ValueError(f"Axes length ({len(ax)}) must match array ndim ({a.ndim}). Axes={ax}, shape={a.shape}")

    present = {ch: i for i, ch in enumerate(ax)}
    order_present = [present[ch] for ch in "TZCYX" if ch in present]  # reorder to canonical
    a2 = np.transpose(a, axes=order_present) if order_present != list(range(a.ndim)) else a

    ax2 = "".join([ch for ch in "TZCYX" if ch in present])  # axes of a2
    for i, ch in enumerate("TZCYX"):
        if ch not in ax2:
            a2 = np.expand_dims(a2, axis=i)
            ax2 = ax2[:i] + ch + ax2[i:]

    # final sanity
    if a2.ndim != 5:
        raise RuntimeError(f"Internal axes normalization failed: got ndim={a2.ndim}, expected 5.")
    return a2  # TZCYX

def _pad_tzcyx_to(arr_tzcyx: np.ndarray, T_out: int, Z_out: int) -> np.ndarray:
    """Pads without cropping: T=start, Z=center."""
    arr = np.asarray(arr_tzcyx)
    if arr.ndim != 5:
        raise ValueError("Expected TZCYX for padding.")
    T, Z, C, H, W = arr.shape
    if T > T_out or Z > Z_out:
        raise ValueError(f"Cannot pad to smaller: ({T},{Z}) -> ({T_out},{Z_out})")
    t0 = 0
    z0 = (Z_out - Z) // 2
    out = np.zeros((T_out, Z_out, C, H, W), dtype=arr.dtype)
    out[t0:t0+T, z0:z0+Z] = arr
    return out


def _bytes_gb(n: int) -> float:
    return float(n) / (1024**3)


def _safe_float32(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.float32:
        return a
    return a.astype(np.float32, copy=False)


def _normalize_bbox_to_y0y1x0x1(bbox, *, bbox_convention: str):
    """
    Accept bbox in either:
      - "y0y1x0x1" : (y0,y1,x0,x1)
      - "y0x0y1x1" : (y0,x0,y1,x1)
    Return canonical list [y0,y1,x0,x1] (ints).
    """
    if bbox is None:
        return None
    if isinstance(bbox, BBox):
        return list(bbox.as_y0y1x0x1())

    b = np.asarray(bbox).reshape(-1).tolist()
    if len(b) != 4:
        return None

    conv = str(bbox_convention).lower().strip()
    if conv == "y0y1x0x1":
        y0, y1, x0, x1 = b
    elif conv == "y0x0y1x1":
        y0, x0, y1, x1 = b
    else:
        raise ValueError("bbox_convention must be 'y0y1x0x1' or 'y0x0y1x1'")

    return [int(y0), int(y1), int(x0), int(x1)]


def require_2d(arr, *, label: str):
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"{label} must be a 2D image (YX) at this stage. Got shape={a.shape} (ndim={a.ndim}).")
    if a.size == 0:
        raise ValueError(f"{label} is empty. shape={a.shape}")
    if not np.isfinite(a).any():
        raise ValueError(f"{label} has no finite pixels (all NaN/Inf).")
    return a


def require_2d_image(arr: np.ndarray, *, label: str):
    """
    Step-1 policy:
      - No implicit Z/T slicing.
      - Accept:
          * 2D grayscale: (Y, X)
          * 2D RGB/RGBA: (Y, X, 3) or (Y, X, 4)
      - If RGB/RGBA: convert to a single 2D grayscale plane (luma) so downstream
        code that expects (H, W) will work.

    Returns:
      - For eager array-likes: a 2D float32 ndarray of shape (Y, X)
      - For lazy 2D arrays (e.g. zarr/dask-like with shape/ndim): returns the input
        object unchanged to avoid eager materialization.
    """
    ndim = getattr(arr, "ndim", None)
    shape = getattr(arr, "shape", None)

    # Lazy-friendly fast path: validate shape/ndim without forcing __array__.
    if ndim == 2 and shape is not None and not isinstance(arr, np.ndarray):
        shp = tuple(shape)
        if len(shp) != 2:
            raise ValueError(
                f"{label} must be a single 2D plane: (Y, X) grayscale or (Y, X, 3/4) RGB/RGBA. "
                f"Got shape={shp} (ndim={len(shp)}).\n"
                f"If your file contains Z/T stacks, select one plane upstream and re-run Step 1."
            )
        if shp[0] <= 0 or shp[1] <= 0:
            raise ValueError(f"{label} is empty (shape={shp}).")
        return arr

    a = np.asarray(arr)

    if a.ndim == 2:
        out = a.astype(np.float32, copy=False)

    elif a.ndim == 3 and a.shape[-1] in (3, 4):
        # Convert RGB/RGBA -> grayscale using luma weights; ignore alpha if present.
        rgb = a[..., :3].astype(np.float32, copy=False)
        w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # R,G,B
        out = rgb @ w  # (Y,X,3) dot (3,) -> (Y,X)

    else:
        raise ValueError(
            f"{label} must be a single 2D plane: (Y, X) grayscale or (Y, X, 3/4) RGB/RGBA. "
            f"Got shape={a.shape} (ndim={a.ndim}).\n"
            f"If your file contains Z/T stacks, select one plane upstream and re-run Step 1."
        )

    if out.size == 0:
        raise ValueError(f"{label} is empty (shape={out.shape}).")
    if not np.isfinite(out).any():
        raise ValueError(f"{label} has no finite pixels (all NaN/Inf).")

    return out


def require_2d_label_mask(mask, *, label, expected_shape=None):
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"{label} must be a 2D label image. Got shape={m.shape} (ndim={m.ndim}).")
    if expected_shape is not None and tuple(m.shape) != tuple(expected_shape):
        raise ValueError(f"{label} shape {m.shape} does not match expected shape {expected_shape}.")
    if not np.issubdtype(m.dtype, np.integer):
        # allow safe cast but alert via print
        print(f"⚠️ {label} dtype is {m.dtype}, casting to int32.")
        m = m.astype(np.int32, copy=False)
    if np.min(m) < 0:
        raise ValueError(f"{label} contains negative labels (min={int(np.min(m))}). Labels must be >= 0.")
    if int(np.max(m)) == 0:
        raise ValueError(f"{label} contains no objects (max label = 0).")
    return m.astype(np.int32, copy=False)


def _require_df_columns(df, cols, *, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}. Available: {list(df.columns)[:30]} ...")


def _require_defined_symbols(names):
    missing = [n for n in names if n not in globals()]
    if missing:
        raise RuntimeError(
            "Missing required functions/objects for Step 2: "
            + ", ".join(missing)
            + ". Ensure earlier cells defining these are executed."
        )


def require_positive_float(x, *, label):
    v = float(x)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{label} must be > 0. Got {x}.")
    return v


