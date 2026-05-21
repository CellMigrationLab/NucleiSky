"""visualization.py ."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .preprocess import ij_percentile_normalize, _safe_float32
from .export import warp_and_save_metrics


def show_alignment_original_and_rescaled(
    res,
    *,
    ij_percentile_normalize=ij_percentile_normalize,
    img_full_orig=None,
    img_crop_orig=None,
    pixel_size_full_orig_um=None,
    pixel_size_crop_orig_um=None,
    save_dir=None,
    margin_um=5.0,
    also_warp_full_to_crop=True,
    compute_warp=True,
    verbose=False,
):
    """
    Display alignment overlays on ORIGINAL images only.

    If save_dir is provided, saves into:
      save_dir/original/
    """

    if res is None or not isinstance(res, dict):
        raise ValueError("res must be a dict returned by NucleiSky().")

    best_scale = res.get("best_scale", None)
    best_R = res.get("best_R", None)
    best_t = res.get("best_t", None)

    if best_scale is None or best_R is None or best_t is None:
        if verbose:
            print("No transform present in result; cannot display alignment.")
        return None

    if img_full_orig is None or img_crop_orig is None:
        raise ValueError("img_full_orig and img_crop_orig must be provided (seg/rescaled context removed).")

    if pixel_size_full_orig_um is None or pixel_size_crop_orig_um is None:
        raise ValueError("pixel_size_full_orig_um and pixel_size_crop_orig_um must be provided.")

    # Resolve save paths
    subdir = None
    if save_dir is not None:
        subdir = Path(save_dir) / "original"
        subdir.mkdir(parents=True, exist_ok=True)
    
    out = warp_and_save_metrics(
        img_full=img_full_orig,          
        crop_img_proc=img_crop_orig,     
        ij_percentile_normalize=ij_percentile_normalize,
        pixel_size_full_um=float(pixel_size_full_orig_um),
        pixel_size_patch_um=float(pixel_size_crop_orig_um),
        best_scale=float(best_scale),
        best_R=np.asarray(best_R, float),
        best_t=np.asarray(best_t, float),
        margin_um=float(margin_um),
        also_warp_full_to_crop=bool(also_warp_full_to_crop),
        compute_warp=bool(compute_warp),
        save_dir=subdir,
        save_prefix="original",
        return_plot_data=True,
    )


    # warp_and_save_metrics returns (bbox, plot_data) when return_plot_data=True
    if isinstance(out, tuple) and len(out) == 2:
        bbox, plot_data = out
        if plot_data:
            plot_warp_overlay(plot_data, save_dir=subdir, save_prefix="original")
        return bbox

    # Fallback: bbox or None
    return out




def _ij_norm_float32(img: np.ndarray) -> np.ndarray:
    """
    Call ij_percentile_normalize but ensure input/output are float32 to reduce RAM.
    """
    x = _safe_float32(img)
    y = ij_percentile_normalize(x)
    return _safe_float32(y)


def imshow_safe(ax, img, *, title: str, cmap: str = "gray", max_dim: int = 2500):
    """
    Downsample + float32-normalize for display to avoid OOM in Jupyter.
    Works for (H,W) and (H,W,C).
    Lazy-safe: strides if image is large/lazy.
    """
    # Downsample first (lazily) to avoid loading huge array
    disp = _downsample_for_display(img, max_dim=max_dim)

    # If RGB, don't pass cmap
    if disp.ndim == 3 and disp.shape[-1] in (3, 4):
        # Normalize each channel jointly (simple, stable)
        disp_n = _ij_norm_float32(disp[..., :3])
        ax.imshow(disp_n)
    else:
        disp_n = _ij_norm_float32(disp)
        ax.imshow(disp_n, cmap=cmap)

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return disp
    
def _downsample_for_display(img, max_dim: int = 2500):
    """
    Return a view (strided) downsampled version of img for visualization only.
    Lazy-safe: checks shape without loading.
    """
    # Lazy shape check
    if hasattr(img, "shape"):
        shape = img.shape
    else:
        shape = np.asarray(img).shape

    H, W = shape[:2]
    s = max(H, W) / float(max_dim)
    
    if s <= 1.0:
        return np.asarray(img) # Safe to load if small
    
    step = int(np.ceil(s))
    
    # Lazy stride slicing: works for Zarr/Dask/Numpy without loading full array
    return np.asarray(img[::step, ::step, ...])

def downsample_points_for_display(pts_yx: np.ndarray, max_points: int = 200_000, seed: int = 0) -> np.ndarray:
    """
    pts_yx: (N,2) array in (y,x). Returns subset for display only.
    """
    pts = np.asarray(pts_yx)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return pts
    n = int(len(pts))
    if n <= max_points:
        return pts
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    return pts[idx]


def plot_warp_overlay(
    plot_data: dict,
    save_dir=None,
    save_prefix="match"
):
    """
    Visualize results produced by warp_and_save_metrics.
    """
    if plot_data is None:
        return

    # Unpack data
    crop_orig_n = plot_data["crop_orig_n"]
    full_roi_n = plot_data["full_roi_n"]
    crop_warp_n = plot_data["crop_warp_n"]
    full_warp_n = plot_data["full_warp_n"]
    err_1 = plot_data["err_1"]
    err_2 = plot_data["err_2"]
    ssim_val_1 = plot_data["ssim_val_1"]
    ssim_val_2 = plot_data["ssim_val_2"]
    h1, w1 = plot_data["dims_1"]
    h2, w2 = plot_data["dims_2"]
    
    if save_dir is not None:
        save_dir = Path(save_dir)

    # Prepare overlays
    # Ensure dimensions match for overlay creation
    def _make_ov(bg, fg, h, w):
        ov = np.zeros((h, w, 3), dtype=np.float32)
        ov[..., 1] = bg[:h, :w] # Green channel = Background (Target)
        ov[..., 0] = fg[:h, :w] # Red   channel = Foreground (Warped Source)
        ov[..., 2] = fg[:h, :w] # Blue  channel = Foreground (Warped Source) -> Magentaish overlap
        return ov

    ov1 = _make_ov(full_roi_n, crop_warp_n, h1, w1)

    if full_warp_n is not None:
        ov2 = _make_ov(crop_orig_n, full_warp_n, h2, w2)

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))

        axes[0, 0].imshow(crop_orig_n, cmap="gray"); axes[0, 0].set_title("Crop (source)")
        axes[0, 1].imshow(full_roi_n, cmap="gray"); axes[0, 1].set_title("Full ROI (target)")
        axes[0, 2].imshow(ov1); axes[0, 2].set_title("Overlay (crop→full ROI)\nG=Target, M=Source")
        
        if err_1 is not None:
            im0 = axes[0, 3].imshow(err_1, cmap="magma", vmin=0, vmax=1)
            axes[0, 3].set_title(f"Error (1-SSIM)\nSSIM={ssim_val_1:.3f}" if ssim_val_1 else "Error")
            plt.colorbar(im0, ax=axes[0, 3], fraction=0.046, pad=0.04)
        else:
            axes[0, 3].axis("off")

        axes[1, 0].imshow(full_warp_n, cmap="gray"); axes[1, 0].set_title("Warp(full→crop)")
        axes[1, 1].imshow(crop_orig_n, cmap="gray"); axes[1, 1].set_title("Crop (target)")
        axes[1, 2].imshow(ov2); axes[1, 2].set_title("Overlay (full→crop)\nG=Target, M=Source")
        
        if err_2 is not None:
            im1 = axes[1, 3].imshow(err_2, cmap="magma", vmin=0, vmax=1)
            axes[1, 3].set_title(f"Error (1-SSIM)\nSSIM={ssim_val_2:.3f}" if ssim_val_2 else "Error")
            plt.colorbar(im1, ax=axes[1, 3], fraction=0.046, pad=0.04)
        else:
            axes[1, 3].axis("off")

        for ax in axes.ravel():
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            fig.savefig(str(save_dir / f"{save_prefix}_overlay.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        axes[0].imshow(crop_orig_n, cmap="gray"); axes[0].set_title("Crop (source)"); axes[0].axis("off")
        axes[1].imshow(full_roi_n, cmap="gray"); axes[1].set_title("Full ROI (target)"); axes[1].axis("off")
        axes[2].imshow(ov1); axes[2].set_title("Overlay (crop→full ROI)"); axes[2].axis("off")
        
        if err_1 is not None:
            im = axes[3].imshow(err_1, cmap="magma", vmin=0, vmax=1)
            axes[3].set_title(f"Error (1-SSIM)\nSSIM={ssim_val_1:.3f}" if ssim_val_1 else "Error")
            axes[3].axis("off")
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        else:
            axes[3].axis("off")
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            fig.savefig(str(save_dir / f"{save_prefix}_overlay.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
