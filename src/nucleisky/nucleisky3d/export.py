"""export.py Export utilities for 3D alignment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np
from scipy.ndimage import affine_transform
from tifffile import imwrite

def _normalize_voxel_size(voxel_size_um, name: str) -> np.ndarray:
    if isinstance(voxel_size_um, (tuple, list, np.ndarray)):
        if len(voxel_size_um) != 3:
            raise ValueError(f"{name} must be a float or length-3 tuple. Got {voxel_size_um}")
        vox = np.asarray(voxel_size_um, dtype=float).reshape(3,)
    else:
        vox = np.asarray([float(voxel_size_um)] * 3, dtype=float)

    if not np.isfinite(vox).all() or np.any(vox <= 0):
        raise ValueError(f"{name} must contain positive finite values. Got {voxel_size_um}")
    return vox


def _normalize_similarity_params(
    *,
    res: dict | None = None,
    best_scale: float | None = None,
    best_R: np.ndarray | None = None,
    best_t: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    if res is not None:
        if not isinstance(res, dict):
            raise TypeError(f"res must be a dict, got {type(res)}")
        if best_scale is None:
            best_scale = res.get("best_scale")
        if best_R is None:
            best_R = res.get("best_R")
        if best_t is None:
            best_t = res.get("best_t")

    if best_scale is None or best_R is None or best_t is None:
        raise ValueError("best_scale, best_R, best_t are required to export aligned volumes.")

    scale = float(best_scale)
    R = np.asarray(best_R, float).reshape(3, 3)
    t = np.asarray(best_t, float).reshape(3,)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"best_scale must be positive finite. Got {best_scale}")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("best_R and best_t must be finite.")

    return scale, R, t


def _normalize_bbox_zyx(
    bbox_zyx,
    *,
    full_shape_zyx: Iterable[int],
    name: str = "bbox_zyx",
) -> tuple[int, int, int, int, int, int]:
    bbox = tuple(int(v) for v in bbox_zyx)
    if len(bbox) != 6:
        raise ValueError(f"{name} must be length-6 (z0,z1,y0,y1,x0,x1). Got {bbox_zyx}")

    z0, z1, y0, y1, x0, x1 = bbox
    Z, Y, X = (int(v) for v in full_shape_zyx)
    if any(v < 0 for v in (z0, y0, x0)):
        raise ValueError(f"{name} must start inside full volume with non-negative mins. Got {bbox}")
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"{name} must satisfy z1>z0, y1>y0, x1>x0. Got {bbox}")
    if z1 > Z or y1 > Y or x1 > X:
        raise ValueError(f"{name} exceeds full volume shape={tuple(full_shape_zyx)}. Got {bbox}")
    return bbox


def _bbox_from_record_or_result(record_or_result: dict) -> tuple[int, int, int, int, int, int]:
    if not isinstance(record_or_result, dict):
        raise TypeError(f"record_or_result must be a dict, got {type(record_or_result)}")

    bbox = record_or_result.get("bbox_full_px_z0z1y0y1x0x1")
    if bbox is None:
        bbox = record_or_result.get("best_bbox")
    if bbox is None:
        raise ValueError(
            "record_or_result must include bbox_full_px_z0z1y0y1x0x1 or best_bbox."
        )
    return tuple(int(v) for v in np.asarray(bbox, dtype=int).reshape(6,))


def _similarity_from_record_or_result(record_or_result: dict) -> tuple[float, np.ndarray, np.ndarray]:
    if not isinstance(record_or_result, dict):
        raise TypeError(f"record_or_result must be a dict, got {type(record_or_result)}")

    best_scale = record_or_result.get("scale", record_or_result.get("best_scale"))
    best_R = record_or_result.get("R_zyx", record_or_result.get("best_R"))
    best_t = record_or_result.get("t_um_zyx", record_or_result.get("best_t"))
    return _normalize_similarity_params(best_scale=best_scale, best_R=best_R, best_t=best_t)


def _maybe_float_to_uint16(data, *, enabled: bool):
    data = np.asarray(data)
    if (not enabled) or (not np.issubdtype(data.dtype, np.floating)):
        return data
    if data.size == 0:
        return data.astype(np.uint16)
    mn = np.nanmin(data)
    mx = np.nanmax(data)
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        x = (data - mn) / (mx - mn)
    else:
        x = np.zeros_like(data, dtype=np.float32)
    return (np.clip(x, 0, 1) * 65535).astype(np.uint16)


def similarity_um_to_affine_px_3d(
    *,
    best_scale: float,
    best_R: np.ndarray,
    best_t: np.ndarray,
    pixel_size_full_um: Iterable[float],
    pixel_size_crop_um: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D similarity in µm (scale, R, t) to affine in pixel coordinates.

    Returns (A_px, b_px) for forward mapping:
        full_px = A_px @ crop_px + b_px
    with coordinates in (z, y, x) order as column vectors.
    """
    pix_full = _normalize_voxel_size(pixel_size_full_um, "pixel_size_full_um")
    pix_crop = _normalize_voxel_size(pixel_size_crop_um, "pixel_size_crop_um")

    scale = float(best_scale)
    R = np.asarray(best_R, float).reshape(3, 3)
    t = np.asarray(best_t, float).reshape(3,)

    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("best_scale must be positive finite.")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("best_R and best_t must be finite.")

    D_crop = np.diag(pix_crop)
    D_full_inv = np.diag(1.0 / pix_full)

    A_px = scale * (D_full_inv @ R @ D_crop)
    b_px = (t / pix_full)

    return A_px, b_px


def _warp_crop_to_full_volume(
    img_crop: np.ndarray,
    *,
    full_shape_zyx: Iterable[int],
    A_px: np.ndarray,
    b_px: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
) -> np.ndarray:
    crop = np.asarray(img_crop)
    if crop.ndim != 3:
        raise ValueError(f"img_crop must be 3D (Z,Y,X). Got shape={crop.shape}")

    full_shape = tuple(int(v) for v in full_shape_zyx)
    if len(full_shape) != 3:
        raise ValueError(f"full_shape_zyx must be length-3. Got {full_shape_zyx}")

    A = np.asarray(A_px, float).reshape(3, 3)
    b = np.asarray(b_px, float).reshape(3,)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        raise ValueError("Affine matrix is singular; cannot warp volume.")

    M = np.linalg.inv(A)
    offset = -M @ b

    dtype = np.dtype(output_dtype) if output_dtype is not None else crop.dtype
    warped = affine_transform(
        crop.astype(np.float32, copy=False),
        matrix=M,
        offset=offset,
        output_shape=full_shape,
        order=int(order),
        mode=str(mode),
        cval=float(cval),
        prefilter=(int(order) > 1),
    )
    return warped.astype(dtype, copy=False)


def warp_crop_to_full_volume(
    img_crop: np.ndarray,
    *,
    full_shape_zyx: Iterable[int],
    pixel_size_full_um,
    pixel_size_crop_um,
    res: dict | None = None,
    best_scale: float | None = None,
    best_R: np.ndarray | None = None,
    best_t: np.ndarray | None = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
) -> np.ndarray:
    """
    Warp the crop into the full volume coordinate space (ZYX) using the alignment.
    """
    scale, R, t = _normalize_similarity_params(
        res=res,
        best_scale=best_scale,
        best_R=best_R,
        best_t=best_t,
    )
    A_px, b_px = similarity_um_to_affine_px_3d(
        best_scale=scale,
        best_R=R,
        best_t=t,
        pixel_size_full_um=pixel_size_full_um,
        pixel_size_crop_um=pixel_size_crop_um,
    )

    return _warp_crop_to_full_volume(
        img_crop,
        full_shape_zyx=full_shape_zyx,
        A_px=A_px,
        b_px=b_px,
        order=order,
        mode=mode,
        cval=cval,
        output_dtype=output_dtype,
    )


def warp_crop_to_full_bbox_volume(
    img_crop: np.ndarray,
    *,
    full_shape_zyx: Iterable[int],
    bbox_zyx,
    pixel_size_full_um,
    pixel_size_crop_um,
    res: dict | None = None,
    best_scale: float | None = None,
    best_R: np.ndarray | None = None,
    best_t: np.ndarray | None = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
    """
    Warp the crop into a reference-space ROI on the full volume grid (ZYX).

    Returns
    -------
    aligned_roi : np.ndarray
        Warped crop sampled onto bbox extents in full-grid coordinates.
    bbox_zyx : tuple[int, int, int, int, int, int]
        ROI location in the full grid as (z0,z1,y0,y1,x0,x1).
    """
    full_shape = tuple(int(v) for v in full_shape_zyx)
    bbox = _normalize_bbox_zyx(bbox_zyx, full_shape_zyx=full_shape, name="bbox_zyx")
    z0, z1, y0, y1, x0, x1 = bbox

    scale, R, t = _normalize_similarity_params(
        res=res,
        best_scale=best_scale,
        best_R=best_R,
        best_t=best_t,
    )
    A_px, b_px = similarity_um_to_affine_px_3d(
        best_scale=scale,
        best_R=R,
        best_t=t,
        pixel_size_full_um=pixel_size_full_um,
        pixel_size_crop_um=pixel_size_crop_um,
    )

    roi_shape = (z1 - z0, y1 - y0, x1 - x0)
    roi_origin = np.asarray([z0, y0, x0], float)
    b_roi = np.asarray(b_px, float) - roi_origin

    aligned_roi = _warp_crop_to_full_volume(
        img_crop,
        full_shape_zyx=roi_shape,
        A_px=A_px,
        b_px=b_roi,
        order=order,
        mode=mode,
        cval=cval,
        output_dtype=output_dtype,
    )
    return aligned_roi, bbox


def export_bbox_pair_tiffs_3d(
    img_full_zyx,
    img_crop_zyx,
    *,
    record_or_result,
    voxel_full_um_zyx,
    voxel_crop_um_zyx,
    out_dir,
    margin_px_zyx=(0, 0, 0),
    prefix="",
) -> dict:
    """Export matching bbox ROIs from full and aligned-crop volumes as TIFFs."""
    # Import lazily to avoid circular import with nucleisky3d.io.
    from .io import save_tiff_zyx
    from .matching.geometry import bbox_add_margin_px_3d

    full = np.asarray(img_full_zyx)
    crop = np.asarray(img_crop_zyx)
    if full.ndim != 3 or crop.ndim != 3:
        raise ValueError(
            f"img_full_zyx and img_crop_zyx must both be 3D ZYX arrays. Got {full.shape=} and {crop.shape=}."
        )

    bbox_base = _normalize_bbox_zyx(
        _bbox_from_record_or_result(record_or_result),
        full_shape_zyx=full.shape,
        name="record_or_result bbox",
    )
    bbox = _normalize_bbox_zyx(
        bbox_add_margin_px_3d(bbox_base, margin_px=margin_px_zyx, shape_zyx=full.shape),
        full_shape_zyx=full.shape,
        name="expanded bbox",
    )
    z0, z1, y0, y1, x0, x1 = bbox

    scale, R, t = _similarity_from_record_or_result(record_or_result)
    aligned_roi, _ = warp_crop_to_full_bbox_volume(
        crop,
        full_shape_zyx=full.shape,
        bbox_zyx=bbox,
        pixel_size_full_um=voxel_full_um_zyx,
        pixel_size_crop_um=voxel_crop_um_zyx,
        best_scale=scale,
        best_R=R,
        best_t=t,
        order=1,
    )
    full_roi = full[z0:z1, y0:y1, x0:x1]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pfx = str(prefix)
    full_path = out_path / f"{pfx}full_bbox.tif"
    aligned_path = out_path / f"{pfx}aligned_crop_bbox.tif"

    save_tiff_zyx(full_path, full_roi, voxel_size_um_zyx=voxel_full_um_zyx, axes="ZYX")
    save_tiff_zyx(aligned_path, aligned_roi, voxel_size_um_zyx=voxel_full_um_zyx, axes="ZYX")

    return {
        "full_bbox_tif": str(full_path),
        "aligned_crop_bbox_tif": str(aligned_path),
        "bbox_full_px_z0z1y0y1x0x1": [int(v) for v in bbox],
    }


def export_aligned_crop_tiff(
    img_full: np.ndarray,
    img_crop: np.ndarray,
    *,
    output_path: str | Path,
    pixel_size_full_um,
    pixel_size_crop_um,
    as_uint16_if_float: bool = False,
    res: dict | None = None,
    best_scale: float | None = None,
    best_R: np.ndarray | None = None,
    best_t: np.ndarray | None = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
    export_region: str = "full",
    write_metadata_json: bool = False,
) -> Path:
    """
    Export the aligned crop as a full-size Z-stack TIFF in the reference space.

    Parameters
    ----------
    img_full : array-like (Z, Y, X)
        Reference volume to match spatial dimensions.
    img_crop : array-like (Z, Y, X)
        Crop volume to be aligned.
    output_path : str or Path
        Destination TIFF path.
    pixel_size_full_um : float or length-3 tuple
        Voxel size (µm) for the full volume in (z,y,x).
    pixel_size_crop_um : float or length-3 tuple
        Voxel size (µm) for the crop volume in (z,y,x).
    as_uint16_if_float : bool
        Convert float outputs to uint16 using min/max scaling.
    res / best_scale/best_R/best_t : alignment params
        Either pass res dict with best_scale/best_R/best_t or pass explicitly.
    """
    full = np.asarray(img_full)
    if full.ndim != 3:
        raise ValueError(f"img_full must be 3D (Z,Y,X). Got shape={full.shape}")

    pix_full = _normalize_voxel_size(pixel_size_full_um, "pixel_size_full_um")
    z_um, y_um, x_um = (float(v) for v in pix_full)

    export_region_u = str(export_region).strip().lower()
    metadata_extra: dict = {"export_region": export_region_u}

    if export_region_u == "full":
        aligned = warp_crop_to_full_volume(
            img_crop,
            full_shape_zyx=full.shape,
            pixel_size_full_um=pixel_size_full_um,
            pixel_size_crop_um=pixel_size_crop_um,
            res=res,
            best_scale=best_scale,
            best_R=best_R,
            best_t=best_t,
            order=order,
            mode=mode,
            cval=cval,
            output_dtype=output_dtype,
        )
    elif export_region_u in {"bbox", "roi"}:
        bbox = None
        if isinstance(res, dict):
            bbox = res.get("best_bbox")
        if bbox is None:
            raise ValueError("export_region='bbox' requires res['best_bbox'].")

        aligned, bbox = warp_crop_to_full_bbox_volume(
            img_crop,
            full_shape_zyx=full.shape,
            bbox_zyx=bbox,
            pixel_size_full_um=pixel_size_full_um,
            pixel_size_crop_um=pixel_size_crop_um,
            res=res,
            best_scale=best_scale,
            best_R=best_R,
            best_t=best_t,
            order=order,
            mode=mode,
            cval=cval,
            output_dtype=output_dtype,
        )
        z0, z1, y0, y1, x0, x1 = bbox
        metadata_extra.update(
            {
                "bbox_full_px_z0z1y0y1x0x1": [z0, z1, y0, y1, x0, x1],
                "bbox_origin_full_px_zyx": [z0, y0, x0],
                "bbox_shape_zyx": [z1 - z0, y1 - y0, x1 - x0],
            }
        )
    else:
        raise ValueError(f"export_region must be 'full' or 'bbox'. Got {export_region!r}")

    aligned = _maybe_float_to_uint16(aligned, enabled=bool(as_uint16_if_float))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(
        str(path),
        aligned,
        imagej=True,
        metadata={"axes": "ZYX", "unit": "um", "spacing": z_um, **metadata_extra},
        resolution=(1.0 / x_um, 1.0 / y_um),
        bigtiff=bool(aligned.nbytes >= (4 * 1024**3)),
    )

    if write_metadata_json or export_region_u in {"bbox", "roi"}:
        sidecar = path.with_suffix(path.suffix + ".json")
        payload = {
            "tiff_path": str(path),
            "shape_zyx": [int(v) for v in aligned.shape],
            "dtype": str(aligned.dtype),
            "pixel_size_um_zyx": [z_um, y_um, x_um],
            **metadata_extra,
        }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_export_with_record_3d(
    *,
    rec: dict,
    ref_path: str | Path,
    mov_path: str | Path,
    out_dir: str | Path,
    output_name: str | None = None,
    channel_axis_ref: int | None = None,
    channel_axis_mov: int | None = None,
    pixel_size_full_um_zyx=None,
    pixel_size_crop_um_zyx=None,
    as_uint16_if_float: bool = False,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
    export_region: str = "full",
    write_metadata_json: bool = False,
) -> dict:
    """
    Convenience export using a 3D transform record and input volumes.

    Exports the aligned crop as either a full-grid volume or a bbox ROI TIFF.
    """
    # Import lazily to avoid circular import with nucleisky3d.io,
    # which imports similarity_um_to_affine_px_3d from this module.
    from .io import load_volume

    img_full = load_volume(str(ref_path), channel_axis=channel_axis_ref)
    img_crop = load_volume(str(mov_path), channel_axis=channel_axis_mov)

    pixel_size_full = pixel_size_full_um_zyx or rec.get("pixel_size_full_um_zyx")
    pixel_size_crop = pixel_size_crop_um_zyx or rec.get("pixel_size_crop_um_zyx")
    if pixel_size_full is None or pixel_size_crop is None:
        raise ValueError("pixel_size_full_um_zyx and pixel_size_crop_um_zyx must be provided or present in rec.")

    best_scale = rec.get("best_scale", rec.get("scale"))
    best_R = rec.get("best_R", rec.get("R_zyx"))
    best_t = rec.get("best_t", rec.get("t_um_zyx"))
    if best_scale is None or best_R is None or best_t is None:
        raise ValueError("rec must include best_scale/best_R/best_t or scale/R_zyx/t_um_zyx.")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        matcher = rec.get("matcher")
        suffix = f"_{matcher}" if matcher else ""
        output_name = f"aligned_crop{suffix}.tif"

    output_path = output_dir / output_name
    path = export_aligned_crop_tiff(
        img_full=img_full,
        img_crop=img_crop,
        output_path=output_path,
        pixel_size_full_um=pixel_size_full,
        pixel_size_crop_um=pixel_size_crop,
        as_uint16_if_float=as_uint16_if_float,
        res=rec,
        best_scale=best_scale,
        best_R=best_R,
        best_t=best_t,
        order=order,
        mode=mode,
        cval=cval,
        output_dtype=output_dtype,
        export_region=export_region,
        write_metadata_json=write_metadata_json,
    )
    return {"files": [str(path)]}
