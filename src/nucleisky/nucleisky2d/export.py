"""export.py Warping/export utilities (ImageJ stacks, overlays, debug visualizations)."""

from pathlib import Path
import numpy as np
import shutil
import math
from scipy.ndimage import affine_transform
from tifffile import imwrite
import traceback

# --- LAZY LOADING IMPORTS ---
try:
    import zarr
    import numcodecs
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from .io import (
    save_tiff,
    save_json,
    save_nucleisky_transform,
    append_transform_jsonl,
    _safe_path,
    _is_zarr_store_path,
    load_image,
    get_pixel_size_um_from_tiff,
)

from .preprocess import (
    _as_tzcyx,
    _from_tzcyx,
    _cast_like,
    _broadcast_outer_dims,
    _pad_tzcyx_to,
    ij_percentile_normalize,
)

from .utils import _rel_err

from .matching.geometry import invert_affine_px
from .matching.geometry import similarity_um_to_affine_px, bbox_full_px_from_similarity_um
from .types import BBox

_ALLOWED_AX_LETTERS = set("TZCYX")

# -------------------------------------------------------------------------
#  CORE WARPING (In-Memory Helper)
# -------------------------------------------------------------------------

def warp_dataset_with_transform(
    moving,
    *,
    A_px,
    b_px,
    out_shape_yx,
    axes="YX",
    order=1,
    mode="constant",
    cval=0.0,
):
    """
    Warp 'moving' into destination grid out_shape_yx.
    
    WARNING: This loads 'moving' into RAM via _as_tzcyx. 
    Ensure input is a sliced ROI if the original image is massive.
    """
    mov_tzcyx = _as_tzcyx(moving, axes=axes)
    T, Z, C, _, _ = mov_tzcyx.shape
    H_out, W_out = map(int, out_shape_yx)
    
    if H_out <= 0 or W_out <= 0:
        return np.zeros((T, Z, C, 0, 0), dtype=mov_tzcyx.dtype)

    # Invert affine for backward mapping (scipy.ndimage convention)
    A_mat = np.asarray(A_px, float).reshape(2, 2)
    b_vec = np.asarray(b_px, float).reshape(2,)
    if abs(np.linalg.det(A_mat)) < 1e-12: raise ValueError("Singular affine matrix")
    M = np.linalg.inv(A_mat)
    offset = -M @ b_vec

    out = np.zeros((T, Z, C, H_out, W_out), dtype=mov_tzcyx.dtype)
    ord_i = int(order)
    
    for t in range(T):
        for z in range(Z):
            for c in range(C):
                img_f = mov_tzcyx[t, z, c].astype(np.float32, copy=False)
                warped_f = affine_transform(
                    img_f, matrix=M, offset=offset, output_shape=(H_out, W_out),
                    order=ord_i, mode=str(mode), cval=float(cval), prefilter=(ord_i > 1)
                )
                out[t, z, c] = _cast_like(warped_f, mov_tzcyx.dtype)

    return _from_tzcyx(out, axes=axes)


def compute_mapped_bbox_y0y1x0x1(
    A_px,
    b_px,
    *,
    src_shape_yx,
    dst_shape_yx=None,
    margin_px=0,
) -> BBox:
    Hs, Ws = map(int, src_shape_yx)

    A = np.asarray(A_px, float).reshape(2, 2)
    b = np.asarray(b_px, float).reshape(2,)

    corners = np.array(
        [
            [0.0, 0.0],
            [float(Hs), 0.0],
            [0.0, float(Ws)],
            [float(Hs), float(Ws)],
        ],
        dtype=float,
    )

    dst = corners @ A.T + b[None, :]
    m = int(margin_px)

    y0 = int(np.floor(dst[:, 0].min())) - m
    y1 = int(np.ceil(dst[:, 0].max())) + m
    x0 = int(np.floor(dst[:, 1].min())) - m
    x1 = int(np.ceil(dst[:, 1].max())) + m

    bb = BBox(y0, y1, x0, x1)

    if dst_shape_yx is not None:
        bb = bb.clamp(tuple(map(int, dst_shape_yx)), min_size=1)
    elif bb.empty:
        bb = BBox(bb.y0, bb.y0 + 1, bb.x0, bb.x0 + 1)

    return bb

def _normalize_bbox_y0y1x0x1(bbox_full_px):
    """
    Accept BBox, dict, tuple/list and return (y0,y1,x0,x1) ints.
    """
    if bbox_full_px is None:
        return None

    if isinstance(bbox_full_px, BBox):
        return tuple(map(int, bbox_full_px.as_y0y1x0x1()))

    if isinstance(bbox_full_px, dict):
        if all(k in bbox_full_px for k in ("y0", "y1", "x0", "x1")):
            return (int(bbox_full_px["y0"]), int(bbox_full_px["y1"]), int(bbox_full_px["x0"]), int(bbox_full_px["x1"]))
        if "y0y1x0x1" in bbox_full_px:
            v = bbox_full_px["y0y1x0x1"]
            if isinstance(v, (list, tuple)) and len(v) == 4:
                return tuple(map(int, v))
        raise ValueError(f"Unsupported bbox dict format: keys={list(bbox_full_px.keys())}")

    if isinstance(bbox_full_px, (list, tuple)) and len(bbox_full_px) == 4:
        return tuple(map(int, bbox_full_px))

    raise TypeError(f"Unsupported bbox_full_px type: {type(bbox_full_px)}")


def _normalize_transform_dict(
    res: dict,
    *,
    pixel_size_full_um: float | None,
    pixel_size_crop_um: float | None,
) -> dict:
    """
    Normalize transform dictionaries from either matcher outputs or saved records.

    Accepted inputs:
      - matcher outputs: best_scale, best_R, best_t (um)
      - saved records: scale, R_yx, t_um_yx and/or A_px, b_px

    Returns a dict with keys:
      - A_px, b_px
      - best_scale, best_R, best_t (if available)
      - bbox_full_px (if provided in any compatible format)
    """
    if not isinstance(res, dict):
        raise TypeError(f"res must be a dict, got {type(res)}")

    keys = set(res.keys())

    def _get_similarity():
        if {"best_scale", "best_R", "best_t"}.issubset(keys):
            return (
                float(res["best_scale"]),
                np.asarray(res["best_R"], float).reshape(2, 2),
                np.asarray(res["best_t"], float).reshape(2,),
            )
        if {"scale", "R_yx", "t_um_yx"}.issubset(keys):
            return (
                float(res["scale"]),
                np.asarray(res["R_yx"], float).reshape(2, 2),
                np.asarray(res["t_um_yx"], float).reshape(2,),
            )
        return None

    A_px = None
    b_px = None
    if {"A_px", "b_px"}.issubset(keys):
        A_px = np.asarray(res["A_px"], float).reshape(2, 2)
        b_px = np.asarray(res["b_px"], float).reshape(2,)

    sim = _get_similarity()
    best_scale = best_R = best_t = None
    if sim is not None:
        best_scale, best_R, best_t = sim

    if A_px is None or b_px is None:
        if sim is None:
            raise KeyError(
                "res missing transform params. Expected "
                "('best_scale','best_R','best_t') or ('scale','R_yx','t_um_yx') "
                "or ('A_px','b_px'). "
                f"Available keys: {sorted(keys)}"
            )
        if pixel_size_full_um is None:
            pixel_size_full_um = res.get("pixel_size_full_um")
        if pixel_size_crop_um is None:
            pixel_size_crop_um = res.get("pixel_size_crop_um")
        if pixel_size_full_um is None or pixel_size_crop_um is None:
            raise ValueError(
                "pixel_size_full_um and pixel_size_crop_um are required to derive A_px/b_px "
                "from similarity params."
            )

        A_px, b_px = similarity_um_to_affine_px(
            best_scale,
            best_R,
            best_t,
            pixel_size_src_um=float(pixel_size_crop_um),
            pixel_size_dst_um=float(pixel_size_full_um),
        )

    bbox_full_px = None
    if "bbox_full_px" in res:
        bbox_full_px = res["bbox_full_px"]
    elif "bbox_full_px_y0y1x0x1" in res:
        bbox_full_px = res["bbox_full_px_y0y1x0x1"]

    return {
        "A_px": A_px,
        "b_px": b_px,
        "best_scale": best_scale,
        "best_R": best_R,
        "best_t": best_t,
        "bbox_full_px": bbox_full_px,
    }


# -------------------------------------------------------------------------
#  LAZY UTILS
# -------------------------------------------------------------------------

def _get_shape_lazy(img):
    """
    Return shape without forcing a full materialization.

    IMPORTANT: If the object has no .shape, do NOT fall back to np.asarray(img)
    because that can load the entire dataset into RAM (breaks laziness).
    """
    sh = getattr(img, "shape", None)
    if sh is None:
        raise TypeError(f"Object of type {type(img)} has no .shape; refusing to np.asarray() for lazy safety.")
    return tuple(sh)

def _axes_index_map(axes: str):
    axes = str(axes).upper().strip()
    if len(set(axes)) != len(axes):
        raise ValueError(f"Invalid axes (duplicate letters): {axes}")
    for ch in axes:
        if ch not in _ALLOWED_AX_LETTERS:
            raise ValueError(f"Invalid axes letter '{ch}' in {axes}. Allowed: {_ALLOWED_AX_LETTERS}")
    return {ch: i for i, ch in enumerate(axes)}, axes

def _slice_block_lazy(img_obj, *, axes: str,
                      t: slice | None = None,
                      z: slice | None = None,
                      c: slice | None = None,
                      y: slice | None = None,
                      x: slice | None = None):
    """
    Slice a block from a lazy array without collapsing dimensions.
    Always uses slices (not integers) so dimensionality is stable.
    """
    pos, axes_u = _axes_index_map(axes)
    shape = _get_shape_lazy(img_obj)
    sl = [slice(None)] * len(shape)

    if "T" in pos and t is not None: sl[pos["T"]] = t
    if "Z" in pos and z is not None: sl[pos["Z"]] = z
    if "C" in pos and c is not None: sl[pos["C"]] = c
    if "Y" in pos and y is not None: sl[pos["Y"]] = y
    if "X" in pos and x is not None: sl[pos["X"]] = x

    return np.asarray(img_obj[tuple(sl)])

def _parse_lazy_shape(shape, ax):
    """Map shape tuple to Z,C,H,W based on axes string."""
    ax = ax.upper()
    dims = {k: 1 for k in "ZCHW"}
    if len(shape) == 2: return 1, 1, shape[0], shape[1]
    
    map_ax = {'Y':'H', 'X':'W'}
    for i, char in enumerate(ax):
        target = map_ax.get(char, char)
        if target in dims and i < len(shape):
            dims[target] = shape[i]
    return dims['Z'], dims['C'], dims['H'], dims['W']

def _slice_roi_lazy(img_obj, y0, y1, x0, x1, axes="YX"):
    """
    Lazy-safe XY slicing: never materializes the full array as a fallback.
    Requires axes to include Y and X.
    """
    pos, axes_u = _axes_index_map(axes)
    if "Y" not in pos or "X" not in pos:
        raise ValueError(f"axes must include Y and X for ROI slicing. Got axes={axes_u}")

    return _slice_block_lazy(
        img_obj,
        axes=axes_u,
        y=slice(int(y0), int(y1)),
        x=slice(int(x0), int(x1)),
    )

def _assert_axes_match_shape(img_obj, axes: str) -> tuple[tuple[int, ...], str]:
    axes_u = str(axes).upper().strip()
    sh = _get_shape_lazy(img_obj)
    if len(sh) != len(axes_u):
        raise ValueError(f"axes '{axes_u}' (len={len(axes_u)}) does not match shape {sh} (ndim={len(sh)}).")
    pos, _ = _axes_index_map(axes_u)
    if "Y" not in pos or "X" not in pos:
        raise ValueError(f"axes must include Y and X. Got axes='{axes_u}'")
    return sh, axes_u

def _dim_from_axes(shape: tuple[int, ...], axes: str, ch: str, default: int = 1) -> int:
    pos, axes_u = _axes_index_map(axes)
    return int(shape[pos[ch]]) if ch in pos else int(default)

def _dtype_lazy(img_obj, default=np.float32):
    dt = getattr(img_obj, "dtype", None)
    return np.dtype(dt) if dt is not None else np.dtype(default)

def _choose_out_dtype(img_full, img_crop, *, prefer_float32_for_int: bool = False):
    dt = np.result_type(_dtype_lazy(img_full), _dtype_lazy(img_crop))
    if prefer_float32_for_int and dt.kind in "iu":
        return np.dtype(np.float32)
    return np.dtype(dt)


# -------------------------------------------------------------------------
#  OME-ZARR WRITER & STREAMER
# -------------------------------------------------------------------------

def _init_ome_zarr(
    path: Path,
    shape_tzcyx,
    dtype,
    pixel_size_xy_um: float,
    name="image",
    *,
    z_spacing_um: float | None = None,
    time_step_ms: float | None = None,
):
    """Initialize empty OME-Zarr on disk (single-scale)."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    if hasattr(zarr, "DirectoryStore"):
        store = zarr.DirectoryStore(str(path))
    elif hasattr(zarr, "storage") and hasattr(zarr.storage, "DirectoryStore"):
        store = zarr.storage.DirectoryStore(str(path))
    elif hasattr(zarr, "storage") and hasattr(zarr.storage, "LocalStore"):
        store = zarr.storage.LocalStore(str(path))
    else:
        raise AttributeError("zarr does not provide a DirectoryStore/LocalStore backend.")
    root = zarr.group(store=store)

    chunks = (1, 1, 1, 1024, 1024)
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

    ds = root.create_dataset(
        "0",
        shape=tuple(map(int, shape_tzcyx)),
        chunks=chunks,
        compressor=compressor,
        dtype=np.dtype(dtype),
        dimension_separator="/",
    )

    # OME-NGFF axes metadata
    axes_meta = [
        {"name": "t", "type": "time", "unit": "millisecond"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    # Scale transform (units correspond to axes_meta)
    t_scale = float(time_step_ms) if (time_step_ms is not None and np.isfinite(time_step_ms) and time_step_ms > 0) else 1.0
    z_scale = float(z_spacing_um) if (z_spacing_um is not None and np.isfinite(z_spacing_um) and z_spacing_um > 0) else 1.0
    xy = float(pixel_size_xy_um) if (pixel_size_xy_um is not None and np.isfinite(pixel_size_xy_um) and pixel_size_xy_um > 0) else 1.0
    scale = [t_scale, z_scale, 1.0, xy, xy]

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": name,
        "axes": axes_meta,
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        }],
    }]

    C = int(shape_tzcyx[2])
    root.attrs["omero"] = {
        "name": name,
        "channels": [{"active": True, "label": f"Ch{i}"} for i in range(C)],
    }
    return ds


def _copy_lazy_chunked(
    source,
    dest_zarr,
    *,
    axes="YX",
    t_start=0,
    z_start=0,
    c_start=0,
    tile_yx=1024,   # default matches your zarr chunk (1024,1024)
    z_chunk=1,
    c_chunk=1,
    t_chunk=1,
):
    """
    Streams 'source' into 'dest_zarr' without loading full Z/C stacks.

    dest_zarr is assumed to be TZCYX.
    'axes' describes the ordering of 'source'.

    Writes into dest indices:
      T: [t_start: ...]
      Z: [z_start: ...]
      C: [c_start: ...]
      Y/X: full extent of source

    Notes:
    - If source lacks T/Z/C axes, it is treated as size-1 in those dims.
    - Uses slice() everywhere to keep dims stable.
    """

    pos, axes_u = _axes_index_map(axes)
    shape = _get_shape_lazy(source)

    # Determine source sizes (t/z/c/y/x) from axes; missing dims => 1
    def _dim(ch, default=1):
        return int(shape[pos[ch]]) if ch in pos else int(default)

    T_src = _dim("T", 1)
    Z_src = _dim("Z", 1)
    C_src = _dim("C", 1)
    H_src = _dim("Y", shape[-2] if len(shape) >= 2 else 1)
    W_src = _dim("X", shape[-1] if len(shape) >= 1 else 1)

    # Destination capacity checks
    T_dst, Z_dst, C_dst, H_dst, W_dst = map(int, dest_zarr.shape)
    if H_src > H_dst or W_src > W_dst:
        raise ValueError(f"Source XY ({H_src}x{W_src}) larger than destination ({H_dst}x{W_dst}).")

    # Clamp how much we can actually write
    T_write = min(T_src, max(0, T_dst - int(t_start)))
    Z_write = min(Z_src, max(0, Z_dst - int(z_start)))
    C_write = min(C_src, max(0, C_dst - int(c_start)))

    if T_write <= 0 or Z_write <= 0 or C_write <= 0:
        return  # nothing fits

    # Iterate over blocks: T, Z, C, then XY tiles
    for t0 in range(0, T_write, int(t_chunk)):
        t1 = min(T_write, t0 + int(t_chunk))
        t_sl = slice(t0, t1) if "T" in pos else None

        for z0 in range(0, Z_write, int(z_chunk)):
            z1 = min(Z_write, z0 + int(z_chunk))
            z_sl = slice(z0, z1) if "Z" in pos else None

            for c0 in range(0, C_write, int(c_chunk)):
                c1 = min(C_write, c0 + int(c_chunk))
                c_sl = slice(c0, c1) if "C" in pos else None

                for y0 in range(0, H_src, int(tile_yx)):
                    y1 = min(H_src, y0 + int(tile_yx))
                    y_sl = slice(y0, y1) if "Y" in pos else slice(y0, y1)

                    for x0 in range(0, W_src, int(tile_yx)):
                        x1 = min(W_src, x0 + int(tile_yx))
                        x_sl = slice(x0, x1) if "X" in pos else slice(x0, x1)

                        # Lazy read of a bounded block (only this T/Z/C + this XY tile)
                        block = _slice_block_lazy(
                            source, axes=axes_u,
                            t=t_sl, z=z_sl, c=c_sl,
                            y=y_sl, x=x_sl
                        )

                        # Normalize to TZCYX; this is now bounded in size
                        block_tzcyx = _as_tzcyx(block, axes=axes_u)

                        # block_tzcyx has shapes:
                        #   T_block = (t1-t0) if T existed else 1
                        #   Z_block = (z1-z0) if Z existed else 1
                        #   C_block = (c1-c0) if C existed else 1
                        # and the same y/x tile sizes.
                        T_block = block_tzcyx.shape[0]
                        Z_block = block_tzcyx.shape[1]
                        C_block = block_tzcyx.shape[2]

                        # Destination slices
                        td0 = int(t_start) + t0
                        zd0 = int(z_start) + z0
                        cd0 = int(c_start) + c0

                        dest_zarr[
                            td0:td0 + T_block,
                            zd0:zd0 + Z_block,
                            cd0:cd0 + C_block,
                            y0:y1,
                            x0:x1
                        ] = block_tzcyx[:T_block, :Z_block, :C_block, :, :]

# -------------------------------------------------------------------------
#  TIFF WRITER
# -------------------------------------------------------------------------

def _maybe_float_to_uint16(data, *, enabled: bool):
    data = np.asarray(data)
    if (not enabled) or (not np.issubdtype(data.dtype, np.floating)): return data
    if data.size == 0: return data.astype(np.uint16)
    mn = np.nanmin(data); mx = np.nanmax(data)
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn: x = (data - mn) / (mx - mn)
    else: x = np.zeros_like(data, dtype=np.float32)
    return (np.clip(x, 0, 1) * 65535).astype(np.uint16)

def imagej_write_hyperstack_tzcyx(
    path: str | Path,
    data_tzcyx,
    *,
    unit: str = "um",
    pixel_size_um: float | None = None,
    z_spacing_um: float | None = None,
    time_interval: float | None = None,
    time_unit: str = "sec",
    as_uint16_if_float: bool = False,
):
    """
    Write ImageJ-compatible TIFF hyperstack (TZCYX).

    Improvements vs previous version:
    - auto BigTIFF when needed (prevents >4GB failures)
    - optional Z spacing and time interval stored in ImageJ metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data_tzcyx)
    if data.ndim != 5:
        raise ValueError(f"Expected TZCYX (5D), got shape={data.shape}")

    data = _maybe_float_to_uint16(data, enabled=bool(as_uint16_if_float))
    unit = str(unit).replace("µ", "u")

    meta = {"axes": "TZCYX", "unit": unit}

    # Optional ImageJ metadata fields
    if z_spacing_um is not None and np.isfinite(float(z_spacing_um)) and float(z_spacing_um) > 0:
        meta["spacing"] = float(z_spacing_um)
    if time_interval is not None and np.isfinite(float(time_interval)) and float(time_interval) > 0:
        # ImageJ expects "finterval" and "tunit" in many readers
        meta["finterval"] = float(time_interval)
        meta["tunit"] = str(time_unit)

    # Resolution tags: keep as px per (metadata unit) for ImageJ
    resolution = None
    if pixel_size_um is not None:
        pix = float(pixel_size_um)
        if pix > 0 and np.isfinite(pix):
            res = 1.0 / pix
            resolution = (res, res)

    # Prevent classic TIFF overflow
    bigtiff = bool(data.nbytes >= (4 * 1024**3))  # ~4GiB threshold

    imwrite(
        str(path),
        data,
        imagej=True,
        metadata=meta,
        resolution=resolution,
        bigtiff=bigtiff,
    )
    return str(path)


def write_ome_zarr_5d(path, data_tzcyx, pixel_size_um=None, chunk_size_2d=(1024, 1024), name="image",
                      *, z_spacing_um=None, time_step_ms=None):
    """
    Chunked write for a 5D TZCYX array.
    Note: data_tzcyx is already in memory; this avoids huge single assignments.
    """
    if not HAS_ZARR:
        raise ImportError("Zarr not installed")

    data = np.asarray(data_tzcyx)
    if data.ndim != 5:
        raise ValueError(f"Expected TZCYX (5D), got shape={data.shape}")

    ds = _init_ome_zarr(
        Path(path),
        data.shape,
        data.dtype,
        float(pixel_size_um) if pixel_size_um is not None else 1.0,
        name=name,
        z_spacing_um=z_spacing_um,
        time_step_ms=time_step_ms,
    )

    ty, tx = map(int, chunk_size_2d)
    T, Z, C, H, W = map(int, data.shape)

    for t in range(T):
        for z in range(Z):
            for c in range(C):
                for y0 in range(0, H, ty):
                    y1 = min(H, y0 + ty)
                    for x0 in range(0, W, tx):
                        x1 = min(W, x0 + tx)
                        ds[t, z, c, y0:y1, x0:x1] = data[t, z, c, y0:y1, x0:x1]

    return str(path)


# -------------------------------------------------------------------------
#  MAIN EXPORT FUNCTION
# -------------------------------------------------------------------------

def export_aligned_dataset(
    res: dict,
    *,
    out_dir: str | Path,
    img_full,
    img_crop,
    pixel_size_full_um: float,
    pixel_size_crop_um: float,
    axes_full="YX",
    axes_crop="YX",
    export_region: str = "roi",
    margin_px: int = 20,
    bbox_full_px: BBox | tuple | list | dict | None = None,
    bbox_convention: str = "y0y1x0x1",
    always_two_stacks: bool = False,
    pixel_size_equal_rtol: float = 1e-3,
    order_intensity: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    as_uint16_if_float: bool = False,
    format: str = "tiff", 
):
    """
    Export aligned reference + warped crop as TIFF or OME-Zarr.

    Supports matcher outputs (best_scale/best_R/best_t) and saved transform
    records (scale/R_yx/t_um_yx and/or A_px/b_px). ROI export is the default.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_region = str(export_region).lower().strip()
    fmt = str(format).lower().strip()
    use_zarr = "zarr" in fmt

    if use_zarr and not HAS_ZARR:
        raise ImportError("Zarr format requested but 'zarr' module not found.")

    # Validate axes early for clearer errors and correct shape inference
    full_shape, axes_full_u = _assert_axes_match_shape(img_full, axes_full)
    crop_shape, axes_crop_u = _assert_axes_match_shape(img_crop, axes_crop)

    # Helper: BBox margin
    def _clamp_bbox(y0, y1, x0, x1, H, W, m):
        y0, x0 = max(0, int(y0) - m), max(0, int(x0) - m)
        y1, x1 = min(H, int(y1) + m), min(W, int(x1) + m)
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        if x1 <= x0:
            x1 = min(W, x0 + 1)
        return int(y0), int(y1), int(x0), int(x1)

    # 1. Get Transform (normalize schema)
    norm = _normalize_transform_dict(
        res,
        pixel_size_full_um=pixel_size_full_um,
        pixel_size_crop_um=pixel_size_crop_um,
    )
    A_cf = np.asarray(norm["A_px"], float).reshape(2, 2)
    b_cf = np.asarray(norm["b_px"], float).reshape(2,)

    # 2. Get Lazy Shapes
    Zf = _dim_from_axes(full_shape, axes_full_u, "Z", 1)
    Cf = _dim_from_axes(full_shape, axes_full_u, "C", 1)
    Hf = _dim_from_axes(full_shape, axes_full_u, "Y", 1)
    Wf = _dim_from_axes(full_shape, axes_full_u, "X", 1)
    Zc = _dim_from_axes(crop_shape, axes_crop_u, "Z", 1)
    Cc = _dim_from_axes(crop_shape, axes_crop_u, "C", 1)
    Hc = _dim_from_axes(crop_shape, axes_crop_u, "Y", 1)
    Wc = _dim_from_axes(crop_shape, axes_crop_u, "X", 1)
    Z_out = max(Zf, Zc)
    C_out = Cf + Cc

    path_full = None

    # ======================================================
    # EXPORT FULL GRID
    # ======================================================
    if export_region == "full":
        # RAM Safety Check
        is_massive = (Hf * Wf) > (20000 * 20000) 
        if not use_zarr and is_massive:
            print(f"⚠️ SKIPPING Full-Grid TIFF: Image size {Hf}x{Wf} too large. Use format='zarr'.")
            return {"skipped": "full_tiff_too_large"}

        if use_zarr:
            # --- OME-ZARR STRATEGY: STREAM & PASTE ---
            path_full = out_dir / "aligned_on_full_px.zarr"

            # Infer dimensions from axes (supports TZCYX / ZCYX / CYX / YX)
            T_full = _dim_from_axes(full_shape, axes_full_u, "T", 1)
            Z_full = _dim_from_axes(full_shape, axes_full_u, "Z", 1)
            C_full = _dim_from_axes(full_shape, axes_full_u, "C", 1)
            Hf = _dim_from_axes(full_shape, axes_full_u, "Y", 1)
            Wf = _dim_from_axes(full_shape, axes_full_u, "X", 1)

            T_crop = _dim_from_axes(crop_shape, axes_crop_u, "T", 1)
            Z_crop = _dim_from_axes(crop_shape, axes_crop_u, "Z", 1)
            C_crop = _dim_from_axes(crop_shape, axes_crop_u, "C", 1)
            Hc = _dim_from_axes(crop_shape, axes_crop_u, "Y", 1)
            Wc = _dim_from_axes(crop_shape, axes_crop_u, "X", 1)

            T_out = max(T_full, T_crop)
            Z_out = max(Z_full, Z_crop)
            C_out = C_full + C_crop

            # Choose safe dtype (avoid downcasting reference)
            dtype = _choose_out_dtype(img_full, img_crop)

            ds = _init_ome_zarr(
                path_full,
                (T_out, Z_out, C_out, Hf, Wf),
                dtype,
                float(pixel_size_full_um),
                name="Aligned",
            )

            # Stream reference into channels [0:C_full]
            print(f"  > Streaming Reference ({Hf}x{Wf}) to Zarr...")
            _copy_lazy_chunked(
                img_full, ds,
                axes=axes_full_u,
                t_start=0, z_start=0, c_start=0,
                tile_yx=1024, z_chunk=1, c_chunk=1, t_chunk=1
            )

            # Compute crop paste ROI in full coordinates (clamped)
            print(f"  > Warping Crop to Zarr...")
            bb = compute_mapped_bbox_y0y1x0x1(
                A_cf, b_cf,
                src_shape_yx=(Hc, Wc),
                dst_shape_yx=(Hf, Wf),
                margin_px=int(margin_px),
            )
            y0, y1, x0, x1 = _clamp_bbox(*bb.as_y0y1x0x1(), H=Hf, W=Wf, m=0)

            # Local affine for ROI destination
            b_cf_roi = b_cf - np.array([y0, x0], float)

            # NOTE: This still materializes the full crop (your current design)
            crop_tzcyx = _as_tzcyx(img_crop, axes=axes_crop_u)

            warped = warp_dataset_with_transform(
                crop_tzcyx,
                A_px=A_cf,
                b_px=b_cf_roi,
                out_shape_yx=(y1 - y0, x1 - x0),
                axes="TZCYX",
                order=order_intensity,
                mode=mode,
                cval=cval,
            )

            warped_tzcyx = _as_tzcyx(warped, axes="TZCYX")
            Tw, Zw, Cw, _, _ = map(int, warped_tzcyx.shape)

            if C_full + Cw > ds.shape[2]:
                raise ValueError(f"Channel overflow: C_full({C_full}) + Cw({Cw}) > C_dst({ds.shape[2]})")

            ds[0:Tw, 0:Zw, C_full:C_full + Cw, y0:y1, x0:x1] = warped_tzcyx
            used_bbox = (0, Hf, 0, Wf)

        else:
            # --- TIFF STRATEGY ---
            full_tzcyx = _as_tzcyx(img_full, axes=axes_full_u)
            crop_tzcyx = _as_tzcyx(img_crop, axes=axes_crop_u)

            # Broadcast outer dims (T/Z/C) if needed
            full_tzcyx, crop_tzcyx = _broadcast_outer_dims(full_tzcyx, crop_tzcyx)

            # Use actual full canvas shape from the converted array
            Hf_true, Wf_true = map(int, full_tzcyx.shape[-2:])

            crop_warp_full = warp_dataset_with_transform(
                crop_tzcyx,
                A_px=A_cf,
                b_px=b_cf,
                out_shape_yx=(Hf_true, Wf_true),
                axes="TZCYX",
                order=order_intensity,
                mode=mode,
                cval=cval,
            )

            stack = np.concatenate([full_tzcyx, _as_tzcyx(crop_warp_full, axes="TZCYX")], axis=2)

            path_full = imagej_write_hyperstack_tzcyx(
                out_dir / "aligned_on_full_px.tif",
                stack,
                pixel_size_um=float(pixel_size_full_um),
                as_uint16_if_float=bool(as_uint16_if_float),
            )
            used_bbox = (0, Hf_true, 0, Wf_true)


    # ======================================================
    # EXPORT ROI (Lazy Safe for all)
    # ======================================================
    else:
        # Determine ROI bbox
        bb_expected = compute_mapped_bbox_y0y1x0x1(
            A_cf,
            b_cf,
            src_shape_yx=(Hc, Wc),
            dst_shape_yx=(Hf, Wf),
            margin_px=int(margin_px),
        )
        y0_exp, y1_exp, x0_exp, x1_exp = _clamp_bbox(*bb_expected.as_y0y1x0x1(), H=Hf, W=Wf, m=0)

        def _bbox_mismatch(bbox_a, bbox_b, *, size_rtol=0.25, center_rtol=0.25) -> bool:
            ya0, ya1, xa0, xa1 = map(int, bbox_a)
            yb0, yb1, xb0, xb1 = map(int, bbox_b)
            ha = max(1, ya1 - ya0)
            wa = max(1, xa1 - xa0)
            hb = max(1, yb1 - yb0)
            wb = max(1, xb1 - xb0)
            size_mismatch = (_rel_err(ha, hb) > size_rtol) or (_rel_err(wa, wb) > size_rtol)
            cya = (ya0 + ya1) * 0.5
            cxa = (xa0 + xa1) * 0.5
            cyb = (yb0 + yb1) * 0.5
            cxb = (xb0 + xb1) * 0.5
            center_mismatch = (abs(cya - cyb) / max(hb, 1) > center_rtol) or (abs(cxa - cxb) / max(wb, 1) > center_rtol)
            return bool(size_mismatch or center_mismatch)

        if bbox_full_px is None:
            bbox_from_res = norm.get("bbox_full_px", None)
            if bbox_from_res is not None:
                bb_in = _normalize_bbox_y0y1x0x1(bbox_from_res)
                bb_in = _clamp_bbox(*bb_in, H=Hf, W=Wf, m=0)

                res_pix_full = res.get("pixel_size_full_um")
                res_pix_crop = res.get("pixel_size_crop_um")
                pix_mismatch = False
                if res_pix_full is not None:
                    pix_mismatch |= _rel_err(res_pix_full, pixel_size_full_um) > pixel_size_equal_rtol
                if res_pix_crop is not None:
                    pix_mismatch |= _rel_err(res_pix_crop, pixel_size_crop_um) > pixel_size_equal_rtol

                if (not pix_mismatch) and (not _bbox_mismatch(bb_in, (y0_exp, y1_exp, x0_exp, x1_exp))):
                    y0, y1, x0, x1 = bb_in
                else:
                    y0, y1, x0, x1 = y0_exp, y1_exp, x0_exp, x1_exp
            else:
                y0, y1, x0, x1 = y0_exp, y1_exp, x0_exp, x1_exp
        else:
            bb_in = _normalize_bbox_y0y1x0x1(bbox_full_px)
            # BBox from records already includes any margin used during matching.
            # Avoid applying margin twice; callers can expand the bbox themselves.
            y0, y1, x0, x1 = _clamp_bbox(*bb_in, H=Hf, W=Wf, m=0)

        used_bbox = (y0, y1, x0, x1)

        # 1. Load ROI of Reference (Lazy Slice)
        full_roi_arr = _slice_roi_lazy(img_full, y0, y1, x0, x1, axes=axes_full_u)
        full_ref_roi = _as_tzcyx(full_roi_arr, axes=axes_full_u)

        # 2. Warp Crop to ROI
        b_cf_roi = b_cf - np.array([y0, x0], float)
        
        # Check if Crop is massive before loading
        # If massive, we implement the inverse-mapping slice (Safe)
        # Otherwise, load full crop (Standard)
        if (Hc * Wc) > (10000 * 10000):  # 100MP heuristic
            # Compute minimal source rect
            A_inv, b_inv = invert_affine_px(A_cf, b_cf_roi)  # Local Inverse
            # Map 4 corners of ROI (y0..y1 etc is local 0..H_roi) -> Source
            H_roi, W_roi = y1 - y0, x1 - x0
            bb_src = compute_mapped_bbox_y0y1x0x1(A_inv, b_inv, src_shape_yx=(H_roi, W_roi), margin_px=10)
            sy0, sy1, sx0, sx1 = _clamp_bbox(*bb_src.as_y0y1x0x1(), H=Hc, W=Wc, m=0)

            # Slice Source
            crop_roi_arr = _slice_roi_lazy(img_crop, sy0, sy1, sx0, sx1, axes=axes_crop_u)
            crop_tzcyx = _as_tzcyx(crop_roi_arr, axes=axes_crop_u)

            # Update transform for sliced source
            # dst = A * src + b  => dst = A * (src_sliced + start) + b => dst = A*src_sliced + (A*start + b)
            b_cf_roi = b_cf_roi + A_cf @ np.array([sy0, sx0], float)
        else:
            crop_tzcyx = _as_tzcyx(img_crop, axes=axes_crop_u)

        full_ref_roi, crop_tzcyx = _broadcast_outer_dims(full_ref_roi, crop_tzcyx)

        crop_warp_roi = warp_dataset_with_transform(
            crop_tzcyx, A_px=A_cf, b_px=b_cf_roi,
            out_shape_yx=(y1-y0, x1-x0), axes="TZCYX",
            order=order_intensity, mode=mode, cval=cval
        )

        stack = np.concatenate([full_ref_roi, _as_tzcyx(crop_warp_roi, axes="TZCYX")], axis=2)

        if use_zarr:
            path_full = write_ome_zarr_5d(out_dir / "aligned_on_full_px.zarr", stack, pixel_size_um=float(pixel_size_full_um), name="Aligned ROI")
        else:
            path_full = imagej_write_hyperstack_tzcyx(out_dir / "aligned_on_full_px.tif", stack, pixel_size_um=float(pixel_size_full_um), as_uint16_if_float=bool(as_uint16_if_float))


    return {
        "out_dir": str(out_dir),
        "aligned_on_full_px": str(path_full) if path_full else None,
        "export_region": export_region,
        "format": "ome-zarr" if use_zarr else "tiff",
        "used_bbox_full_px_y0y1x0x1": [int(v) for v in used_bbox],
    }

export_aligned_imagej_stacks = export_aligned_dataset


def inspect_image_header(path_str: str):
    """
    Reads image metadata (shape, dtype, axes, pixel size) without loading data.
    Supports TIFF (via tifffile) and NPY.
    """
    p = _safe_path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suf = p.suffix.lower()

    info = {"path": str(p), "kind": suf, "shape": None, "dtype": None, "axes": None, "pixel_size_um": None}

    if suf in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(str(p)) as tf:
            series = tf.series[0]
            info["shape"] = tuple(series.shape)
            info["dtype"] = str(series.dtype)

            try:
                info["pixel_size_um"] = get_pixel_size_um_from_tiff(str(p), return_details=False)
            except Exception:
                info["pixel_size_um"] = None

            # Try to get axes (prefer series.axes)
            ax = getattr(series, "axes", None)
            if isinstance(ax, str):
                ax = ax.upper().strip()
                if set(ax).issubset(_ALLOWED_AX_LETTERS) and len(set(ax)) == len(ax):
                    info["axes"] = ax

            # ImageJ metadata may provide explicit axes override
            ijm = getattr(tf, "imagej_metadata", None)
            if isinstance(ijm, dict):
                axm = ijm.get("axes", None)
                if isinstance(axm, str):
                    axm = axm.upper().strip()
                    if set(axm).issubset(_ALLOWED_AX_LETTERS) and len(set(axm)) == len(axm):
                        info["axes"] = axm

    elif suf == ".npy":
        # Memory-map npy to read header only
        arr = np.load(str(p), mmap_mode="r")
        info["shape"] = tuple(arr.shape)
        info["dtype"] = str(arr.dtype)

    elif _is_zarr_store_path(p):
        if HAS_ZARR:
            try:
                z = zarr.open(str(p), mode="r")

                arr = None
                axes_str = None
                pix_um = None

                if isinstance(z, zarr.Group):
                    ms = z.attrs.get("multiscales", [])
                    if ms:
                        ms0 = ms[0]
                        axes_meta = ms0.get("axes", [])
                        # axes can be list of dicts or list of strings
                        names = []
                        for a in axes_meta:
                            if isinstance(a, dict):
                                names.append(str(a.get("name", "")).lower())
                            else:
                                names.append(str(a).lower())

                        # Map ngff axis names to your axis letters
                        name_to_letter = {"t": "T", "z": "Z", "c": "C", "y": "Y", "x": "X"}
                        axes_str = "".join([name_to_letter.get(n, "") for n in names if n in name_to_letter]) or None

                        ds0 = ms0["datasets"][0]["path"]
                        arr = z[ds0]

                        # Try to read XY pixel size from scale transform
                        ct = ms0["datasets"][0].get("coordinateTransformations", [])
                        if ct:
                            s = ct[0].get("scale", None)
                            if isinstance(s, (list, tuple)) and axes_str is not None and len(s) == len(axes_str):
                                iy = axes_str.index("Y") if "Y" in axes_str else None
                                ix = axes_str.index("X") if "X" in axes_str else None
                                if iy is not None and ix is not None:
                                    sy = float(s[iy])
                                    sx = float(s[ix])
                                    if np.isfinite(sy) and np.isfinite(sx) and sy > 0 and sx > 0:
                                        pix_um = 0.5 * (sy + sx)

                    else:
                        # Fallback: first array
                        keys = list(z.array_keys())
                        arr = z[keys[0]] if keys else None
                else:
                    arr = z

                if arr is not None:
                    info["shape"] = tuple(arr.shape)
                    info["dtype"] = str(arr.dtype)
                if axes_str is not None:
                    info["axes"] = axes_str
                if pix_um is not None:
                    info["pixel_size_um"] = float(pix_um)

            except Exception:
                pass


    else:
        # Fallback: unsupported or unknown format
        pass

    return info

def run_export_with_record(
    *,
    rec: dict,
    transform_path: str,
    ref_path: str,
    mov_path: str,
    out_dir: str,
    axes_ref: str,
    axes_mov: str,
    pixel_size_ref_um: float,
    pixel_size_mov_um: float,
    margin_px: int,
    order_intensity: int = 1,
    pixel_size_equal_rtol: float = 1e-3,
    export_fullgrid_fullXY: bool = False,
    export_fullgrid_roiXY: bool = True,
    export_cropgrid_fullXY: bool = True,
    export_cropgrid_roiXY: bool = False,
    force_cropgrid: bool = False,
    format: str = "tiff",
):
    img_full = load_image(ref_path)
    img_crop = load_image(mov_path)

    files_created = []

    common_kwargs = dict(
        res=rec,
        img_full=img_full,
        img_crop=img_crop,
        pixel_size_full_um=pixel_size_ref_um,
        pixel_size_crop_um=pixel_size_mov_um,
        axes_full=axes_ref,
        axes_crop=axes_mov,
        margin_px=margin_px,
        order_intensity=order_intensity,
        format=format,
    )
    common_kwargs.pop("out_dir", None)  

    base_out = Path(out_dir)
    pix_equal = _rel_err(pixel_size_ref_um, pixel_size_mov_um) <= pixel_size_equal_rtol
    if pix_equal and not force_cropgrid and (export_fullgrid_fullXY or export_fullgrid_roiXY):
        export_cropgrid_fullXY = False
        export_cropgrid_roiXY = False

    if export_fullgrid_fullXY:
        print(f"  > Exporting Full Grid (Full Canvas)... mode={format}")
        sub_dir = base_out / f"ref_grid_full_{format}"

        ret = export_aligned_dataset(
            out_dir=sub_dir,          
            export_region="full",
            **common_kwargs,
        )
        if ret.get("aligned_on_full_px"):
            files_created.append(ret["aligned_on_full_px"])
        elif ret.get("skipped"):
            print(f"    ! Skipped: {ret['skipped']}")

    if export_fullgrid_roiXY:
        print(f"  > Exporting Full Grid (ROI Only)... mode={format}")
        sub_dir = base_out / f"ref_grid_roi_{format}"

        ret = export_aligned_dataset(
            out_dir=sub_dir,          # pass ONLY here
            export_region="roi",
            **common_kwargs,
        )
        if ret.get("aligned_on_full_px"):
            files_created.append(ret["aligned_on_full_px"])

    if export_cropgrid_fullXY or export_cropgrid_roiXY:
        norm = _normalize_transform_dict(
            rec,
            pixel_size_full_um=pixel_size_ref_um,
            pixel_size_crop_um=pixel_size_mov_um,
        )
        A_cf = np.asarray(norm["A_px"], float).reshape(2, 2)
        b_cf = np.asarray(norm["b_px"], float).reshape(2,)
        A_fc, b_fc = invert_affine_px(A_cf, b_cf)
        res_inv = {"A_px": A_fc, "b_px": b_fc}

        crop_kwargs = dict(
            res=res_inv,
            img_full=img_crop,
            img_crop=img_full,
            pixel_size_full_um=pixel_size_mov_um,
            pixel_size_crop_um=pixel_size_ref_um,
            axes_full=axes_mov,
            axes_crop=axes_ref,
            margin_px=margin_px,
            order_intensity=order_intensity,
            format=format,
        )

        if export_cropgrid_fullXY:
            print(f"  > Exporting Crop Grid (Full Canvas)... mode={format}")
            sub_dir = base_out / f"crop_grid_full_{format}"
            ret = export_aligned_dataset(
                out_dir=sub_dir,
                export_region="full",
                **crop_kwargs,
            )
            if ret.get("aligned_on_full_px"):
                files_created.append(ret["aligned_on_full_px"])
            elif ret.get("skipped"):
                print(f"    ! Skipped: {ret['skipped']}")

        if export_cropgrid_roiXY:
            print(f"  > Exporting Crop Grid (ROI Only)... mode={format}")
            sub_dir = base_out / f"crop_grid_roi_{format}"
            ret = export_aligned_dataset(
                out_dir=sub_dir,
                export_region="roi",
                **crop_kwargs,
            )
            if ret.get("aligned_on_full_px"):
                files_created.append(ret["aligned_on_full_px"])

    return {"files": files_created}



def _export_best_everything(
    best_out: dict,
    *,
    out_dir: Path,
    jsonl_path: Path,
    matcher_name: str,
    # original images
    img_full_orig,
    img_crop_orig,
    pixel_size_full_orig_um: float,
    pixel_size_crop_orig_um: float,
    # seg-scale images
    img_full_seg,
    img_crop_seg,
    pixel_size_full_seg_um: float,
    pixel_size_crop_seg_um: float,
    margin_px: int,
    export_do_full: bool = False,
    export_do_roi: bool = True,
    export_segscale: bool = False,
    save_segscale_transform: bool = False,
    run_id: str | None = None,
):
    """
    Automated export routine for pipeline.py.
    Uses lazy/robust export for original resolution images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Transforms
    rec = save_nucleisky_transform(
        best_out,
        out_path=out_dir / "adaptive_best_transform_original.json",
        matcher_name=f"adaptive_best_{matcher_name}",
        pixel_size_full_um=float(pixel_size_full_orig_um),
        pixel_size_crop_um=float(pixel_size_crop_orig_um),
        require_success=bool(best_out.get("success", False)),
    )
    if run_id is not None:
        rec["run_id"] = str(run_id)
    append_transform_jsonl(rec, jsonl_path)

    if save_segscale_transform or export_segscale:
        rec_seg = save_nucleisky_transform(
            best_out,
            out_path=out_dir / "adaptive_best_transform_segscale.json",
            matcher_name=f"adaptive_best_{matcher_name}_segscale",
            pixel_size_full_um=float(pixel_size_full_seg_um),
            pixel_size_crop_um=float(pixel_size_crop_seg_um),
            require_success=bool(best_out.get("success", False)),
        )
        if run_id is not None:
            rec_seg["run_id"] = str(run_id)
        append_transform_jsonl(rec_seg, jsonl_path)

    # Helper wrapper for export_aligned_dataset
    def _do_export(subdir_name, region, img_f, img_c, pix_f, pix_c):
        # Auto-switch to Zarr for full exports of massive images to prevent crashes
        # Check size roughly (heuristic)
        is_massive = False
        
        try:
            shape = _get_shape_lazy(img_f)
            h, w = int(shape[-2]), int(shape[-1])
            if h * w > 400_000_000:
                is_massive = True
        except Exception:
            pass

        
        fmt = "zarr" if (region == "full" and is_massive) else "tiff"
        
        return export_aligned_dataset(
            res=best_out,
            out_dir=out_dir / subdir_name,
            img_full=img_f,
            img_crop=img_c,
            pixel_size_full_um=float(pix_f),
            pixel_size_crop_um=float(pix_c),
            axes_full="YX", # robust assumption for pipeline inputs
            axes_crop="YX",
            export_region=region,
            margin_px=int(margin_px),
            format=fmt
        )

    # 2. FULL Exports (Original + SegScale)
    if export_do_full:
        _do_export("adaptive_best_images_original_full", "full", img_full_orig, img_crop_orig, pixel_size_full_orig_um, pixel_size_crop_orig_um)
        if export_segscale:
            _do_export("adaptive_best_images_segscale_full", "full", img_full_seg, img_crop_seg, pixel_size_full_seg_um, pixel_size_crop_seg_um)

    # 3. ROI Exports (Original + SegScale)
    if export_do_roi:
        _do_export("adaptive_best_images_original_roi", "roi", img_full_orig, img_crop_orig, pixel_size_full_orig_um, pixel_size_crop_orig_um)
        if export_segscale:
            _do_export("adaptive_best_images_segscale_roi", "roi", img_full_seg, img_crop_seg, pixel_size_full_seg_um, pixel_size_crop_seg_um)
# -------------------------------------------------------------------------
#  VISUALIZATION (Separate from Export)
# -------------------------------------------------------------------------
def warp_and_save_metrics(
    img_full, crop_img_proc, ij_percentile_normalize,
    pixel_size_full_um, pixel_size_patch_um,
    best_scale, best_R, best_t,
    margin_um="auto",
    also_warp_full_to_crop=True,
    compute_warp=True,
    save_dir=None,
    save_prefix="match",
    return_plot_data=False,
):
    """
    Computes warp, calculates SSIM, and saves results.
    Lazy-loading safe.
    """
    crop_img_proc = np.asarray(crop_img_proc)
    Hc, Wc = crop_img_proc.shape[:2]

    # Lazy shape extraction
    if hasattr(img_full, "shape"):
        Hf, Wf = img_full.shape[:2]
    else:
        Hf, Wf = np.asarray(img_full).shape[:2]

    if best_scale is None or best_R is None or best_t is None:
        return None

    s = float(best_scale)
    if (not np.isfinite(s)) or s <= 0:
        return None

    R = np.asarray(best_R, float).reshape(2, 2)
    t_um = np.asarray(best_t, float).reshape(2,)
    pixF = float(pixel_size_full_um)
    pixC = float(pixel_size_patch_um)

    # ---------------------------------------------------------
    # 1. AUTO-PADDING
    # ---------------------------------------------------------
    if margin_um == "auto":
        margin_val = 3.0 * pixF 
    else:
        margin_val = float(margin_um)

    # ---------------------------------------------------------
    # 2. Exact BBox Calculation
    # ---------------------------------------------------------
    bbox = bbox_full_px_from_similarity_um(
        crop_shape_px=(Hc, Wc),
        pixel_size_full_um=pixF,
        pixel_size_crop_um=pixC,
        scale=s,
        R_yx=R,
        t_um_yx=t_um,
        margin_um=margin_val, 
        full_shape_px=(Hf, Wf),
    )
    y0, y1, x0, x1 = bbox.as_y0y1x0x1()
    Hr, Wr = (y1 - y0), (x1 - x0)
    
    if Hr <= 0 or Wr <= 0:
        return bbox

    # ... [Save Dir & Fast Path Logic] ...
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if (not compute_warp) and (not return_plot_data) and (save_dir is None):
        return bbox

    # ---------------------------------------------------------
    # 3. Lazy Load ROI
    # ---------------------------------------------------------
    try:
        full_roi = np.asarray(img_full[y0:y1, x0:x1])
    except Exception as e:
        print(f"⚠️ Error slicing full image: {e}")
        return bbox

    # ---------------------------------------------------------
    # 4. Compute Affine Warps
    # ---------------------------------------------------------
    A = (s * pixC / pixF) * R
    b = t_um / pixF

    # Offset ROI: map (0,0) in ROI -> (y0,x0) in Full -> Crop
    A_inv = np.linalg.inv(A)
    offset_roi = A_inv @ (np.array([y0, x0], float) - b)

    from scipy.ndimage import affine_transform

    crop_warp_to_full = affine_transform(
        crop_img_proc.astype(np.float32, copy=False),
        matrix=A_inv,
        offset=offset_roi,
        output_shape=(Hr, Wr),
        order=1,
        mode="constant", cval=0.0, prefilter=False,
    )

    full_warp_to_crop = None
    if also_warp_full_to_crop:
        offset_crop = b - np.array([y0, x0], float)
        full_warp_to_crop = affine_transform(
            full_roi.astype(np.float32, copy=False),
            matrix=A,
            offset=offset_crop,
            output_shape=(Hc, Wc),
            order=1,
            mode="constant", cval=0.0, prefilter=False,
        )
    # ---------------------------------------------------------
    # 5. Normalize & Compute SSIM
    # ---------------------------------------------------------
    
    full_roi_n = np.clip(ij_percentile_normalize(full_roi).astype(np.float32), 0, 1)
    crop_warp_n = np.clip(ij_percentile_normalize(crop_warp_to_full).astype(np.float32), 0, 1)
    crop_orig_n = np.clip(ij_percentile_normalize(crop_img_proc).astype(np.float32), 0, 1)

    full_warp_n = None
    if full_warp_to_crop is not None:
        full_warp_n = np.clip(ij_percentile_normalize(full_warp_to_crop).astype(np.float32), 0, 1)

    from skimage.metrics import structural_similarity as ssim
    
    def _safe_ssim(a, b):
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h < 3 or w < 3: return None, None, (h, w)
        a2, b2 = a[:h, :w], b[:h, :w]
        win = min(7, h, w)
        if win % 2 == 0: win -= 1
        if win < 3: return None, None, (h, w)
        val, smap = ssim(a2, b2, data_range=1.0, full=True, win_size=win)
        return float(val), smap, (h, w)

    ssim_val_1, ssim_map_1, (h1, w1) = _safe_ssim(crop_warp_n, full_roi_n)
    err_1 = None if ssim_map_1 is None else np.clip(1.0 - ssim_map_1, 0, 1).astype(np.float32)

    ssim_val_2, ssim_map_2, (h2, w2) = (None, None, (None, None))
    err_2 = None
    if full_warp_n is not None:
        ssim_val_2, ssim_map_2, (h2, w2) = _safe_ssim(full_warp_n, crop_orig_n)
        err_2 = None if ssim_map_2 is None else np.clip(1.0 - ssim_map_2, 0, 1).astype(np.float32)

    # ---------------------------------------------------------
    # 6. Save & Return
    # ---------------------------------------------------------
    if save_dir is not None:
        from tifffile import imwrite
        def _sv(n, d): imwrite(save_dir / n, d)
        
        _sv(f"{save_prefix}_full_roi.tif", full_roi)
        _sv(f"{save_prefix}_crop_warp_to_full_roi.tif", crop_warp_to_full)
        if full_warp_to_crop is not None:
            _sv(f"{save_prefix}_full_warp_to_crop.tif", full_warp_to_crop)

    if return_plot_data:
        plot_data = {
            "crop_orig_n": crop_orig_n,
            "full_roi_n": full_roi_n,
            "crop_warp_n": crop_warp_n,
            "full_warp_n": full_warp_n,
            "err_1": err_1,
            "err_2": err_2,
            "ssim_val_1": ssim_val_1,
            "ssim_val_2": ssim_val_2,
            "dims_1": (h1, w1),
            "dims_2": (h2, w2),
        }
        return bbox, plot_data

    return bbox
