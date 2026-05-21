"""segmentation.py Segmentation backends."""

from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
import logging
import os

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import (
    gaussian,
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.measure import label
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed
from .preprocess import _to_2d, _coerce_label_2d

logger = logging.getLogger(__name__)

# Constants
MAX_TILE = None          
TILE_OVERLAP = None 
DEFAULT_DIAMETER = None

_THRESHOLD_FUNCS = {
    "otsu": threshold_otsu,
    "li": threshold_li,
    "yen": threshold_yen,
    "triangle": threshold_triangle,
    "isodata": threshold_isodata,
}


def _remove_small_objects_compat(mask: np.ndarray, min_object_size: int) -> np.ndarray:
    """Compatibility wrapper preserving legacy '< min_size' semantics."""
    thr = int(min_object_size)
    if thr <= 0:
        return np.asarray(mask, dtype=bool)
    # New skimage API removes objects with size <= max_size.
    # Legacy min_size removed objects with size < min_size.
    return remove_small_objects(np.asarray(mask, dtype=bool), max_size=max(0, thr - 1))


def _remove_small_holes_compat(mask: np.ndarray, min_hole_size: int) -> np.ndarray:
    """Compatibility wrapper preserving legacy '< area_threshold' semantics."""
    thr = int(min_hole_size)
    if thr <= 0:
        return np.asarray(mask, dtype=bool)
    # New skimage API fills holes with size <= max_size.
    # Legacy area_threshold filled holes with size < area_threshold.
    return remove_small_holes(np.asarray(mask, dtype=bool), max_size=max(0, thr - 1))

class Segmentor:
    """
    Manages segmentation models and execution to ensure thread safety 
    and proper resource management.
    """
    def __init__(self):
        self._cellpose_model = None
        self._cellpose_backend = None
        self._instanseg_cache = {}

    def get_cellpose_model(self, pretrained_model="cpsam", model_type=None):
        """Lazy loader for Cellpose model."""
        if self._cellpose_model is None:
            self._cellpose_model, self._cellpose_backend = self._init_cellpose_model(
                pretrained_model=pretrained_model,
                model_type=model_type,
            )
        return self._cellpose_model, self._cellpose_backend

    def get_instanseg_model(self, model_name, verbosity=0):
        """Cached loader for InstanSeg models."""
        key = (str(model_name), int(verbosity))
        if key not in self._instanseg_cache:
            from instanseg import InstanSeg
            logger.info(f"Loading InstanSeg model: {model_name}")
            m = InstanSeg(model_name, verbosity=int(verbosity))
            self._instanseg_cache[key] = m
        return self._instanseg_cache[key]

    def _init_cellpose_model(self, pretrained_model="cpsam", model_type=None):
        """
        Cellpose v4 initializer with GPU->CPU fallback.
        Defaults to pretrained_model='cpsam' per v4 docs.
        """
        from cellpose.models import CellposeModel

        devnull = open(os.devnull, "w")
        try:
            logger.info(f"Initializing Cellpose (GPU) model: {pretrained_model}")
            with redirect_stdout(devnull), redirect_stderr(devnull):
                model = CellposeModel(
                    gpu=True,
                    pretrained_model=pretrained_model,
                    model_type=model_type,
                )
            return model, "CellposeModel(gpu)"
        except Exception as e:
            logger.warning(f"Cellpose GPU init failed ({e}), falling back to CPU.")
            with redirect_stdout(devnull), redirect_stderr(devnull):
                model = CellposeModel(
                    gpu=False,
                    pretrained_model=pretrained_model,
                    model_type=model_type,
                )
            return model, "CellposeModel(cpu)"
        finally:
            devnull.close()

    def segment_instanseg(
        self,
        img,
        pixel_size_um,
        model_name="brightfield_nuclei",
        target="nuclei",
        mode="auto",                 # "auto" | "small" | "medium"
        auto_medium_pixels=6_000_000,
        verbosity=0,
        robust_normalize=True,
        cleanup_fragments=True,
        resolve_cell_and_nucleus=False,
        use_mean_threshold=False,
        mean_threshold=0.3,
    ):
        """
        Run InstanSeg segmentation.
        """
        inst = self.get_instanseg_model(model_name=model_name, verbosity=verbosity)

        x = np.asarray(img)

        # normalize to uint8 if requested (keeps RAM reasonable too)
        if robust_normalize and x.dtype != np.uint8:
            x_in = _robust_uint8(x)
        else:
            x_in = x

        # Ensure correct channel count for common brightfield models
        if "brightfield" in str(model_name).lower():
            x_in = _instanseg_force_channels(x_in, 3)
        else:
            x_in = _instanseg_as_hwc(x_in)

        px = float(pixel_size_um)

        # Choose mode based on spatial dimensions (H*W)
        shape = x_in.shape
        n_pix = shape[0] * shape[1] # H * W

        m = mode
        if m == "auto":
            m = "medium" if n_pix >= int(auto_medium_pixels) else "small"

        # Build kwargs safely (InstanSeg signatures may vary)
        common_kwargs = {
            "cleanup_fragments": bool(cleanup_fragments),
            "resolve_cell_and_nucleus": bool(resolve_cell_and_nucleus),
        }
        if use_mean_threshold:
            common_kwargs["mean_threshold"] = float(mean_threshold)

        # Convert HWC -> CHW for InstanSeg (standard PyTorch image format)
        if x_in.ndim == 3 and x_in.shape[-1] <= 4:
             x_in = np.moveaxis(x_in, -1, 0)

        if m == "small":
            try:
                out = inst.eval_small_image(x_in, px, target=str(target), **common_kwargs)
            except TypeError:
                out = inst.eval_small_image(x_in, px)
        elif m == "medium":
            try:
                out = inst.eval_medium_image(x_in, px, target=str(target), **common_kwargs)
            except TypeError:
                out = inst.eval_medium_image(x_in, px)
        else:
            raise ValueError(f"Unknown InstanSeg mode: {mode}")

        labels2d = _coerce_label_2d(out, target=target)
        return labels2d.astype(np.int32, copy=False)

    def segment_cellpose(
        self,
        img2d,
        tile_size=MAX_TILE,
        overlap=TILE_OVERLAP,
        diameter=DEFAULT_DIAMETER,
        batch_size=1,
        normalize=True,
        invert=False,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=15,
        pretrained_model="cpsam",
    ):
        """
        Run Cellpose segmentation.
        """
        from cellpose import transforms
        cp_model, _ = self.get_cellpose_model(pretrained_model=pretrained_model)

        x = np.asarray(img2d)
        if x.ndim != 2:
            raise ValueError("segment_with_cellpose expects a 2D image.")
        x = x.astype(np.float32, copy=False)

        # Cellpose v4 expects images with 3 channels
        x3 = transforms.convert_image(x, channel_axis=None, z_axis=None, do_3D=False)
        x3 = np.asarray(x3, dtype=np.float32)  # H x W x 3

        # Decide bsize (model tile size)
        use_kwargs = {}
        if tile_size is not None:
            bsize = int(tile_size)
            if bsize <= 0:
                raise ValueError("tile_size must be a positive int or None.")
            # Transformer/SAM-style backbones are safest at training bsize=256
            if _is_transformer_backbone(cp_model) and bsize != 256:
                logger.warning(f"Cellpose transformer backbone detected; overriding bsize {bsize} -> 256 (recommended).")
                bsize = 256
            use_kwargs["bsize"] = bsize
            use_kwargs["tile_overlap"] = float(_coerce_tile_overlap(overlap, bsize))
        else:
            # Let Cellpose pick defaults (bsize=256, tile_overlap=0.1)
            if overlap is not None:
                bsize = 256
                use_kwargs["bsize"] = bsize
                use_kwargs["tile_overlap"] = float(_coerce_tile_overlap(overlap, bsize))

        masks, *_ = cp_model.eval(
            [x3],
            batch_size=int(batch_size),
            channel_axis=-1,
            normalize=normalize,
            invert=invert,
            diameter=diameter,
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
            min_size=int(min_size),
            **use_kwargs,
        )

        m = masks[0] if isinstance(masks, (list, tuple)) else masks
        return np.asarray(m, dtype=np.int32)

    def segment_threshold(
        self,
        img,
        threshold_method="otsu",
        channel=0,
        gaussian_sigma=1.0,
        min_object_size=80,
        min_hole_size=80,
        do_watershed=True,
        peak_min_distance=5,
        watershed_compactness=0.0,
        foreground="bright",
    ):
        """
        Run auto-threshold based segmentation.
        """
        x = _to_2d(img, channel=channel).astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xs = gaussian(x, sigma=float(gaussian_sigma), preserve_range=True)

        method = str(threshold_method).lower().strip()
        if method not in _THRESHOLD_FUNCS:
            raise ValueError(f"Unknown threshold_method '{threshold_method}'. Options: {list(_THRESHOLD_FUNCS)}")
        t = _THRESHOLD_FUNCS[method](xs)

        fg = str(foreground).lower().strip()
        if fg == "bright":
            mask = xs > t
        elif fg == "dark":
            mask = xs < t
        else:
            raise ValueError("foreground must be 'bright' or 'dark'")

        mask = _remove_small_objects_compat(mask, int(min_object_size))
        mask = _remove_small_holes_compat(mask, int(min_hole_size))

        if not bool(do_watershed):
            return label(mask).astype(np.int32)

        dist = ndi.distance_transform_edt(mask)
        peaks = peak_local_max(dist, min_distance=int(peak_min_distance), labels=mask, exclude_border=False)
        markers = np.zeros_like(dist, dtype=np.int32)
        for i, (r, c) in enumerate(peaks, start=1):
            markers[r, c] = i
        markers = label(markers > 0).astype(np.int32)

        labels_ws = watershed(-dist, markers=markers, mask=mask, compactness=float(watershed_compactness))
        return labels_ws.astype(np.int32)

# --- Helper Functions (Stateless) ---

def _is_transformer_backbone(model) -> bool:
    # v4 exposes model.backbone ("default" vs "transformer")
    bb = str(getattr(model, "backbone", "")).lower()
    if "transformer" in bb:
        return True
    # extra safety: SAM/VIT module name
    net_mod = getattr(getattr(model, "net", None), "__class__", type("X",(object,),{})).__module__
    return "vit" in str(net_mod).lower() or "sam" in str(net_mod).lower()

def _coerce_tile_overlap(overlap, bsize: int) -> float:
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

def _instanseg_as_hwc(x):
    """Ensure HWC ordering for 3D arrays; leave 2D unchanged."""
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise ValueError(f"InstanSeg input must be 2D or 3D. Got shape={x.shape}")

    # Heuristic: CHW if first dim is small and other dims are image-like
    if x.shape[0] in (1, 2, 3, 4) and (x.shape[1] > 16 and x.shape[2] > 16) and (x.shape[0] < x.shape[-1]):
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    return x

def _instanseg_force_channels(x, n_channels=3):
    """
    Force HWC with exactly n_channels.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return np.stack([x] * int(n_channels), axis=-1)

    x = _instanseg_as_hwc(x)
    C = x.shape[-1]

    n_channels = int(n_channels)
    if C == n_channels:
        return x
    if C == 1:
        return np.repeat(x, n_channels, axis=-1)
    if C < n_channels:
        # pad by repeating channel 0
        pad = np.repeat(x[..., :1], n_channels - C, axis=-1)
        return np.concatenate([x, pad], axis=-1)
    # C > n_channels
    return x[..., :n_channels]

def _robust_uint8(x, p_low=0.5, p_high=99.5):
    x = np.asarray(x)
    if x.ndim == 2:
        xf = x.astype(np.float32, copy=False)
        lo, hi = np.percentile(xf, [p_low, p_high])
        if not np.isfinite(lo): lo = float(np.min(xf))
        if not np.isfinite(hi): hi = float(np.max(xf))
        if hi <= lo: hi = lo + 1.0
        y = (np.clip(xf, lo, hi) - lo) / (hi - lo)
        return (y * 255.0).astype(np.uint8)
    if x.ndim == 3:
        out = np.zeros(x.shape, dtype=np.uint8)
        for c in range(x.shape[-1]):
            out[..., c] = _robust_uint8(x[..., c], p_low=p_low, p_high=p_high)
        return out
    raise ValueError(f"_robust_uint8 expects 2D or 3D (HxWxC). Got {x.shape}")


# --- Module-Level Compatibility API ---

_GLOBAL_SEGMENTOR = Segmentor()


def get_global_segmentor() -> Segmentor:
    """Public accessor for the shared module-level segmentor instance."""
    return _GLOBAL_SEGMENTOR

def segment_with_instanseg(img, pixel_size_um, **kwargs):
    """Legacy wrapper using global segmentor instance."""
    return _GLOBAL_SEGMENTOR.segment_instanseg(img, pixel_size_um, **kwargs)

def segment_with_cellpose(img2d, **kwargs):
    """Legacy wrapper using global segmentor instance."""
    return _GLOBAL_SEGMENTOR.segment_cellpose(img2d, **kwargs)

def segment_with_auto_threshold(img, **kwargs):
    """Legacy wrapper using global segmentor instance."""
    return _GLOBAL_SEGMENTOR.segment_threshold(img, **kwargs)

def segment_nuclei_dispatch(img, method, pixel_size_um, settings=None, segmentor: Optional[Segmentor] = None):
    """
    Main dispatch function.
    
    Args:
        segmentor: Optional Segmentor instance. If None, uses the shared module-level instance.
    """
    settings = settings or {}
    m = str(method).lower().strip()
    
    # Use provided segmentor or fall back to global default
    seg_impl = segmentor if segmentor is not None else _GLOBAL_SEGMENTOR

    if m == "cellpose":
        s = settings.get("cellpose", {})
        return seg_impl.segment_cellpose(img, **s)

    if m == "instanseg":
        s = settings.get("instanseg", {})
        px = float(s.get("pixel_size_um", pixel_size_um))
        return seg_impl.segment_instanseg(
            img,
            pixel_size_um=px,
            model_name=str(s.get("model_name", "brightfield_nuclei")),
            target=str(s.get("target", "nuclei")),
            mode=str(s.get("mode", "auto")),
            auto_medium_pixels=int(s.get("auto_medium_pixels", 6_000_000)),
            verbosity=int(s.get("verbosity", 0)),
            robust_normalize=bool(s.get("robust_normalize", True)),
            cleanup_fragments=bool(s.get("cleanup_fragments", True)),
            resolve_cell_and_nucleus=bool(s.get("resolve_cell_and_nucleus", False)),
            use_mean_threshold=bool(s.get("use_mean_threshold", False)),
            mean_threshold=float(s.get("mean_threshold", 0.3)),
        )

    if m == "threshold":
        s = settings.get("threshold", {})
        return seg_impl.segment_threshold(
            img,
            threshold_method=str(s.get("threshold_method", "otsu")),
            channel=int(s.get("channel", 0)),
            gaussian_sigma=float(s.get("gaussian_sigma", 1.0)),
            min_object_size=int(s.get("min_object_size", 80)),
            min_hole_size=int(s.get("min_hole_size", 80)),
            do_watershed=bool(s.get("do_watershed", True)),
            peak_min_distance=int(s.get("peak_min_distance", 5)),
            watershed_compactness=float(s.get("watershed_compactness", 0.0)),
            foreground=str(s.get("foreground", "bright")),
        )

    raise ValueError(f"Unknown segmentation method: {method}")
