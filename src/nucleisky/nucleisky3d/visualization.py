"""visualization.py QC plotting for 3D alignment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .export import warp_crop_to_full_volume, warp_crop_to_full_bbox_volume
from .matching.geometry import bbox_full_px_from_similarity_um_3d
from .preprocess import ij_percentile_normalize


# ----------------------------
# Small helpers (shared)
# ----------------------------
def _as_zyx(vol: np.ndarray, name: str) -> np.ndarray:
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"{name} must be 3D (Z,Y,X). Got shape={vol.shape}")
    return vol


def _mip_z(vol_zyx: np.ndarray) -> np.ndarray:
    """Z-MIP -> (Y,X)."""
    vol_zyx = _as_zyx(vol_zyx, "volume")
    return np.max(vol_zyx, axis=0)


def _display_step(h: int, w: int, max_dim: int) -> int:
    return int(max(1, int(np.ceil(max(h, w) / float(max_dim)))))


def _downsample_for_display(img: np.ndarray, *, max_dim: int) -> tuple[np.ndarray, int]:
    """
    Strided downsample (view) for display. Works for:
      - 2D (H,W)
      - RGB (H,W,3)
    """
    arr = np.asarray(img)
    if arr.ndim < 2:
        return arr, 1
    h, w = arr.shape[:2]
    step = _display_step(h, w, max_dim=max_dim)
    if step <= 1:
        return arr, 1
    return arr[::step, ::step, ...], step


def imshow_safe3d(
    ax: plt.Axes,
    img: np.ndarray,
    *,
    title: str,
    cmap: str = "gray",
    max_dim: int = 2500,
) -> np.ndarray:
    """Downsample + normalize image for safer QC display on large data."""
    disp, _ = _downsample_for_display(img, max_dim=max_dim)

    if disp.ndim == 3 and disp.shape[-1] in (3, 4):
        disp_n = ij_percentile_normalize(disp[..., :3], p_low=1, p_high=99).astype(
            np.float32, copy=False
        )
        ax.imshow(disp_n)
    else:
        disp_n = ij_percentile_normalize(disp, p_low=1, p_high=99).astype(
            np.float32, copy=False
        )
        ax.imshow(disp_n, cmap=cmap)

    ax.set_title(title)
    return disp_n


# Backward-compatible aliases.
imshow_safe = imshow_safe3d
_imshow_safe = imshow_safe3d

def _normalize_pair_2d(
    a: np.ndarray,
    b: np.ndarray,
    p_low: float,
    p_high: float,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize a & b to [0,1] using a shared (vmin,vmax) computed from both images.
    This makes overlays/errors comparable.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    am = np.isfinite(a)
    bm = np.isfinite(b)

    if am.any():
        alo, ahi = np.percentile(a[am], [p_low, p_high])
    else:
        alo, ahi = 0.0, 1.0
    if bm.any():
        blo, bhi = np.percentile(b[bm], [p_low, p_high])
    else:
        blo, bhi = 0.0, 1.0

    vmin = float(min(alo, blo))
    vmax = float(max(ahi, bhi))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin + eps):
        vmin, vmax = 0.0, 1.0

    a_n = np.clip((a - vmin) / (vmax - vmin + eps), 0.0, 1.0)
    b_n = np.clip((b - vmin) / (vmax - vmin + eps), 0.0, 1.0)
    return a_n, b_n


def _overlay_target_green_source_magenta(target_n: np.ndarray, source_n: np.ndarray) -> np.ndarray:
    """RGB overlay: target=G, source=R+B (magenta)."""
    h, w = target_n.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 1] = target_n
    rgb[..., 0] = source_n
    rgb[..., 2] = source_n
    return rgb


def _auto_ds_full(full_shape_zyx: tuple[int, int, int], max_full_warp_voxels: int) -> int:
    """
    Isotropic downsample factor so full warp stays under max_full_warp_voxels.
    """
    full_voxels = int(np.prod(full_shape_zyx))
    if full_voxels <= int(max_full_warp_voxels):
        return 1
    ds = int(np.ceil((full_voxels / float(max_full_warp_voxels)) ** (1.0 / 3.0)))
    return max(1, ds)


def _get_transform_and_bbox(record_or_result: dict) -> tuple[float, np.ndarray, np.ndarray, tuple[int, int, int, int, int, int] | None]:
    """
    Accepts either:
      - NucleiSky3D output: best_scale, best_R, best_t, best_bbox
      - saved record: scale, R_zyx, t_um_zyx, bbox_full_px_z0z1y0y1x0x1
    """
    if not isinstance(record_or_result, dict):
        raise ValueError("record_or_result must be a dict.")

    def _first(*keys):
        for k in keys:
            v = record_or_result.get(k, None)
            if v is not None:
                return v
        return None

    scale = _first("scale", "best_scale")
    R = _first("R_zyx", "best_R")
    t = _first("t_um_zyx", "best_t", "best_t_um_zyx")
    bbox = _first("bbox_full_px_z0z1y0y1x0x1", "best_bbox")

    if scale is None or R is None or t is None:
        raise ValueError(
            "record_or_result must contain (scale, R_zyx, t_um_zyx) "
            "or (best_scale, best_R, best_t)."
        )

    scale = float(scale)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3,)

    bbox_tup = None
    if bbox is not None:
        bbox_tup = tuple(int(v) for v in bbox)

    return scale, R, t, bbox_tup


# ----------------------------
# Main: one figure per matcher
# ----------------------------
def plot_warp_overlay3d(
    img_full_zyx: np.ndarray,
    img_crop_zyx: np.ndarray,
    record_or_result: dict,
    *,
    pixel_size_full_um_zyx,
    pixel_size_crop_um_zyx,
    roi_margin_um: float = 0.0,
    z_full: int | None = None,
    z_crop: int | None = None,
    clip_percentiles: tuple[float, float] = (1.0, 99.8),
    max_display_dim: int = 512,
    max_full_warp_voxels: int = 64_000_000,
    include_roi: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
):
    """
    One-figure QC overlay for 3D matching (single plot per matcher).

    Rows:
      - FULL: single Z slice
      - FULL: Z-MIP
      - ROI:  single Z slice            (if include_roi=True)
      - ROI:  Z-MIP                     (if include_roi=True)

    Columns:
      - Crop (source)
      - Full/ROI (target)
      - Overlay (G=target, M=source)
      - |diff| error

    Note: FULL warps are downsampled automatically if full volume is huge (QC-only).
    """
    full = _as_zyx(img_full_zyx, "img_full_zyx")
    crop = _as_zyx(img_crop_zyx, "img_crop_zyx")

    full_shape = tuple(int(v) for v in full.shape)
    crop_shape = tuple(int(v) for v in crop.shape)

    voxel_full = np.asarray(pixel_size_full_um_zyx, dtype=float).reshape(3,)
    voxel_crop = np.asarray(pixel_size_crop_um_zyx, dtype=float).reshape(3,)

    p_low, p_high = float(clip_percentiles[0]), float(clip_percentiles[1])

    scale, R, t, bbox_rec = _get_transform_and_bbox(record_or_result)

    # Always compute bbox0 (margin=0) for consistent defaults/rects.
    bbox0 = bbox_rec or tuple(
        int(v) for v in bbox_full_px_from_similarity_um_3d(
            crop_shape_px=crop_shape,
            pixel_size_full_um_zyx=voxel_full,
            pixel_size_crop_um_zyx=voxel_crop,
            scale=scale,
            R_zyx=R,
            t_um_zyx=t,
            margin_um=0.0,
            full_shape_px=full_shape,  # clamp
        )
    )
    bbox_roi = tuple(
        int(v) for v in bbox_full_px_from_similarity_um_3d(
            crop_shape_px=crop_shape,
            pixel_size_full_um_zyx=voxel_full,
            pixel_size_crop_um_zyx=voxel_crop,
            scale=scale,
            R_zyx=R,
            t_um_zyx=t,
            margin_um=float(roi_margin_um),
            full_shape_px=full_shape,  # clamp
        )
    )

    cz0, cz1, cy0, cy1, cx0, cx1 = bbox0
    rz0, rz1, ry0, ry1, rx0, rx1 = bbox_roi

    # Default slices: full at bbox-center, crop at center
    if z_full is None:
        z_full = int(np.clip((cz0 + cz1) // 2, 0, full_shape[0] - 1))
    else:
        z_full = int(np.clip(z_full, 0, full_shape[0] - 1))

    if z_crop is None:
        z_crop = int(crop_shape[0] // 2)
    else:
        z_crop = int(np.clip(z_crop, 0, crop_shape[0] - 1))

    # -------- FULL warp (QC-only, maybe downsampled) --------
    ds_full = _auto_ds_full(full_shape, max_full_warp_voxels=max_full_warp_voxels)
    full_ds = full[::ds_full, ::ds_full, ::ds_full]
    voxel_full_ds = voxel_full * float(ds_full)
    full_shape_ds = tuple(int(v) for v in full_ds.shape)

    aligned_full_ds = warp_crop_to_full_volume(
        crop,
        full_shape_zyx=full_shape_ds,
        pixel_size_full_um=tuple(float(v) for v in voxel_full_ds),
        pixel_size_crop_um=tuple(float(v) for v in voxel_crop),
        best_scale=scale,
        best_R=R,
        best_t=t,
        order=1,
        mode="constant",
        cval=0.0,
        output_dtype=np.float32,
    )

    z_full_ds = int(np.clip(z_full // ds_full, 0, full_shape_ds[0] - 1))

    crop_slice = crop[z_crop]
    crop_mip = _mip_z(crop)

    full_slice = full_ds[z_full_ds]
    warp_slice = aligned_full_ds[z_full_ds]
    full_mip = _mip_z(full_ds)
    warp_mip = _mip_z(aligned_full_ds)

    full_slice_n, warp_slice_n = _normalize_pair_2d(full_slice, warp_slice, p_low, p_high)
    full_mip_n, warp_mip_n = _normalize_pair_2d(full_mip, warp_mip, p_low, p_high)

    ov_full_slice = _overlay_target_green_source_magenta(full_slice_n, warp_slice_n)
    ov_full_mip = _overlay_target_green_source_magenta(full_mip_n, warp_mip_n)
    err_full_slice = np.abs(full_slice_n - warp_slice_n)
    err_full_mip = np.abs(full_mip_n - warp_mip_n)

    crop_slice_n = ij_percentile_normalize(crop_slice, p_low=p_low, p_high=p_high).astype(np.float32, copy=False)
    crop_mip_n = ij_percentile_normalize(crop_mip, p_low=p_low, p_high=p_high).astype(np.float32, copy=False)

    # Display downsampling (keep step for ROI rect coordinates)
    crop_slice_show, _ = _downsample_for_display(crop_slice_n, max_dim=max_display_dim)
    crop_mip_show, _ = _downsample_for_display(crop_mip_n, max_dim=max_display_dim)

    full_slice_show, ds_fs = _downsample_for_display(full_slice_n, max_dim=max_display_dim)
    full_mip_show, ds_fm = _downsample_for_display(full_mip_n, max_dim=max_display_dim)

    ov_full_slice_show, _ = _downsample_for_display(ov_full_slice, max_dim=max_display_dim)
    ov_full_mip_show, _ = _downsample_for_display(ov_full_mip, max_dim=max_display_dim)

    err_full_slice_show, _ = _downsample_for_display(err_full_slice, max_dim=max_display_dim)
    err_full_mip_show, _ = _downsample_for_display(err_full_mip, max_dim=max_display_dim)

    # ROI rectangle in the FULL panels:
    # bbox_roi is native full px -> convert to full_ds px -> then to display px.
    rx0_ds = rx0 // ds_full
    rx1_ds = rx1 // ds_full
    ry0_ds = ry0 // ds_full
    ry1_ds = ry1 // ds_full

    # -------- ROI warp (native ROI resolution) --------
    roi_rows = []
    if include_roi:
        aligned_roi, bbox_used = warp_crop_to_full_bbox_volume(
            crop,
            full_shape_zyx=full_shape,
            bbox_zyx=bbox_roi,
            pixel_size_full_um=tuple(float(v) for v in voxel_full),
            pixel_size_crop_um=tuple(float(v) for v in voxel_crop),
            best_scale=scale,
            best_R=R,
            best_t=t,
            order=1,
            mode="constant",
            cval=0.0,
            output_dtype=np.float32,
        )
        uz0, uz1, uy0, uy1, ux0, ux1 = (int(v) for v in bbox_used)
        full_roi = full[uz0:uz1, uy0:uy1, ux0:ux1]

        z_full_roi = int(np.clip(z_full - uz0, 0, max(0, full_roi.shape[0] - 1)))

        roi_slice = full_roi[z_full_roi]
        roi_warp_slice = aligned_roi[z_full_roi]
        roi_mip = _mip_z(full_roi)
        roi_warp_mip = _mip_z(aligned_roi)

        roi_slice_n, roi_warp_slice_n = _normalize_pair_2d(roi_slice, roi_warp_slice, p_low, p_high)
        roi_mip_n, roi_warp_mip_n = _normalize_pair_2d(roi_mip, roi_warp_mip, p_low, p_high)

        ov_roi_slice = _overlay_target_green_source_magenta(roi_slice_n, roi_warp_slice_n)
        ov_roi_mip = _overlay_target_green_source_magenta(roi_mip_n, roi_warp_mip_n)
        err_roi_slice = np.abs(roi_slice_n - roi_warp_slice_n)
        err_roi_mip = np.abs(roi_mip_n - roi_warp_mip_n)

        roi_slice_show, _ = _downsample_for_display(roi_slice_n, max_dim=max_display_dim)
        roi_mip_show, _ = _downsample_for_display(roi_mip_n, max_dim=max_display_dim)
        ov_roi_slice_show, _ = _downsample_for_display(ov_roi_slice, max_dim=max_display_dim)
        ov_roi_mip_show, _ = _downsample_for_display(ov_roi_mip, max_dim=max_display_dim)
        err_roi_slice_show, _ = _downsample_for_display(err_roi_slice, max_dim=max_display_dim)
        err_roi_mip_show, _ = _downsample_for_display(err_roi_mip, max_dim=max_display_dim)

        roi_rows = [
            (crop_slice_show, roi_slice_show, ov_roi_slice_show, err_roi_slice_show, "ROI: single Z slice"),
            (crop_mip_show, roi_mip_show, ov_roi_mip_show, err_roi_mip_show, "ROI: Z-MIP projection"),
        ]

    # -------- Plot --------
    rows = [
        (crop_slice_show, full_slice_show, ov_full_slice_show, err_full_slice_show, "FULL: single Z slice"),
        (crop_mip_show, full_mip_show, ov_full_mip_show, err_full_mip_show, "FULL: Z-MIP projection"),
        *roi_rows,
    ]
    nrows = len(rows)

    fig, axes = plt.subplots(nrows, 4, figsize=(22, 5 * nrows))

    if nrows == 1:
        axes = np.asarray([axes])  # keep shape (1,4)

    if title is None:
        title = (
            f"3D Warp Overlay | scale={scale:.4g} | z_full={z_full} z_crop={z_crop} | "
            f"roi_margin_um={float(roi_margin_um):.3g} | ds_full={ds_full}"
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    col_titles = ["Crop (source)", "Target", "Overlay (G=target, M=source)", "|diff| error"]

    def _imshow(ax, img, cmap=None):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(img)
        ax.axis("off")

    for i, (a, b, ov, err, row_label) in enumerate(rows):
        _imshow(axes[i, 0], a, cmap="gray")
        _imshow(axes[i, 1], b, cmap="gray")
        _imshow(axes[i, 2], ov, cmap=None)
        _imshow(axes[i, 3], err, cmap="magma")

        # Row label on left
        axes[i, 0].text(
            -0.06, 0.5, row_label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

    # Column titles on top row
    for j, ttxt in enumerate(col_titles):
        axes[0, j].set_title(ttxt, fontsize=11, fontweight="bold")

    # ROI rectangle on FULL target panels (row 0 and row 1, col 1)
    rect0 = Rectangle(
        (rx0_ds / ds_fs, ry0_ds / ds_fs),
        max(1.0, (rx1_ds - rx0_ds) / ds_fs),
        max(1.0, (ry1_ds - ry0_ds) / ds_fs),
        fill=False,
        linewidth=2,
        edgecolor="yellow",
    )
    axes[0, 1].add_patch(rect0)

    rect1 = Rectangle(
        (rx0_ds / ds_fm, ry0_ds / ds_fm),
        max(1.0, (rx1_ds - rx0_ds) / ds_fm),
        max(1.0, (ry1_ds - ry0_ds) / ds_fm),
        fill=False,
        linewidth=2,
        edgecolor="yellow",
    )
    axes[1, 1].add_patch(rect1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# Convenience alias (matches your requested name)
plot_warp_overlay3D = plot_warp_overlay3d


