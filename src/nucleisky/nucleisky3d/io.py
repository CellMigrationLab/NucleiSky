"""io.py I/O helpers: 3D TIFF metadata, JSON, and volume loading."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from skimage import io as skio
from tifffile import TiffFile, imread, imwrite

from .export import similarity_um_to_affine_px_3d
from .types import VoxelSizeDetails

_UM_PER_UNIT = {
    "m": 1e6,
    "meter": 1e6,
    "metre": 1e6,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimetre": 1e3,
    "cm": 1e4,
    "centimeter": 1e4,
    "centimetre": 1e4,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometre": 1e-3,
    "in": 25400.0,
    "inch": 25400.0,
    "inches": 25400.0,
}

_ZARR_MARKER_FILES = (".zgroup", ".zarray", ".zattrs", ".zmetadata", "zarr.json")
_LOG = logging.getLogger(__name__)


def _is_zarr_store_path(p: Path) -> bool:
    """Return True when the path points to a likely Zarr store/group."""
    if p.suffix.lower() == ".zarr":
        return True
    if not p.is_dir():
        return False
    return any((p / marker).exists() for marker in _ZARR_MARKER_FILES)

def _norm_unit(u: Any) -> Optional[str]:
    if u is None:
        return None
    s = str(u).strip()
    if not s:
        return None
    s = s.replace("μ", "µ")
    s = s.lower()
    s = s.replace("micrometers", "micrometer").replace("micrometres", "micrometre")
    s = s.replace("microns", "micron")
    if s == "um":
        s = "µm"
    return s


def _to_um(value: Any, unit: Any) -> Optional[float]:
    if value is None:
        return None
    u = _norm_unit(unit)
    if u is None:
        return None
    factor = _UM_PER_UNIT.get(u)
    if factor is None:
        return None
    try:
        return float(value) * float(factor)
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _ratio_to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, tuple) and len(v) == 2:
        num, den = v
        try:
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        except Exception:
            return None
    if hasattr(v, "numerator") and hasattr(v, "denominator"):
        try:
            den = float(v.denominator)
            if den == 0:
                return None
            return float(v.numerator) / den
        except Exception:
            return None
    if isinstance(v, str) and "/" in v:
        try:
            a, b = v.split("/", 1)
            b = float(b)
            if b == 0:
                return None
            return float(a) / b
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _decode_tag_value(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return None
    return str(x)


def _parse_image_description_kv(desc: Optional[str]) -> dict:
    if not desc:
        return {}
    lines = [ln.strip() for ln in str(desc).replace("\r", "\n").split("\n") if ln.strip()]
    out = {}
    for ln in lines:
        if "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip().lower()] = v.strip()
        elif ":" in ln and ln.lower().startswith(("unit", "spacing", "xresolution", "yresolution")):
            k, v = ln.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _parse_ome_physical_sizes_um(ome_xml: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not ome_xml:
        return (None, None, None)
    
    # Strip non-XML prefixes (e.g., ImageJ headers)
    ome_upper = ome_xml.upper()
    if "<OME" in ome_upper:
        ome_xml = ome_xml[ome_upper.find("<OME"):]
        
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return (None, None, None)

    if root.tag.startswith("{") and "}" in root.tag:
        ome_ns = root.tag.split("}")[0].strip("{")
        ns = {"ome": ome_ns}
        pix = root.find(".//ome:Pixels", ns)
    else:
        pix = root.find(".//Pixels")

    if pix is None:
        return (None, None, None)

    x = _safe_float(pix.get("PhysicalSizeX"))
    y = _safe_float(pix.get("PhysicalSizeY"))
    z = _safe_float(pix.get("PhysicalSizeZ"))

    x_um = _to_um(x, pix.get("PhysicalSizeXUnit"))
    y_um = _to_um(y, pix.get("PhysicalSizeYUnit"))
    z_um = _to_um(z, pix.get("PhysicalSizeZUnit"))

    return (x_um, y_um, z_um)


def _get_page_and_tags(tif: TiffFile, page_index: int = 0):
    try:
        page = tif.pages[page_index]
        return page, page.tags
    except Exception:
        return None, None


def _parse_xy_from_resolution_tags(
    tags,
    *,
    resolution_unit_code: int,
    imagej_unit_hint: Optional[str] = None,
    allow_guess_unit_when_missing: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if tags is None:
        return (None, None, "No tags")
    if "XResolution" not in tags or "YResolution" not in tags:
        return (None, None, "Missing XResolution/YResolution tags")

    xres = _ratio_to_float(tags["XResolution"].value)
    yres = _ratio_to_float(tags["YResolution"].value)
    if xres is None or yres is None or xres <= 0 or yres <= 0:
        return (None, None, "Invalid XResolution/YResolution values")

    unit_um = None
    note = None

    if int(resolution_unit_code) == 2:
        unit_um = 25400.0
    elif int(resolution_unit_code) == 3:
        unit_um = 10000.0
    else:
        u = _norm_unit(imagej_unit_hint)
        if u is not None and u in _UM_PER_UNIT:
            unit_um = float(_UM_PER_UNIT[u])
            note = f"ResolutionUnit=1; used ImageJ unit='{u}' to interpret X/YResolution."
        elif allow_guess_unit_when_missing:
            unit_um = 25400.0
            note = "ResolutionUnit missing/None; heuristically assuming inch (opt-in)."
        else:
            return (None, None, "ResolutionUnit missing/None and no ImageJ unit hint; not guessing.")

    x_um = float(unit_um) / float(xres)
    y_um = float(unit_um) / float(yres)
    return (x_um, y_um, note)


def _parse_tiff_resolution_tags_um(
    tif: TiffFile,
    *,
    page_index: int = 0,
    allow_guess_unit_when_missing: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    page, tags = _get_page_and_tags(tif, page_index=page_index)
    if page is None:
        return (None, None, "Invalid page_index")

    ij = getattr(tif, "imagej_metadata", None) or {}
    ij_unit = ij.get("unit", None) if isinstance(ij, dict) else None

    desc = None
    try:
        if tags is not None and "ImageDescription" in tags:
            desc = _decode_tag_value(tags["ImageDescription"].value)
    except Exception:
        desc = None

    kv = _parse_image_description_kv(desc)
    desc_unit = kv.get("unit", None)

    unit_hint = ij_unit or desc_unit

    ru = 1
    try:
        if tags is not None and "ResolutionUnit" in tags:
            ru = int(tags["ResolutionUnit"].value)
    except Exception:
        ru = 1

    return _parse_xy_from_resolution_tags(
        tags,
        resolution_unit_code=ru,
        imagej_unit_hint=unit_hint,
        allow_guess_unit_when_missing=allow_guess_unit_when_missing,
    )


def get_voxel_size_um_from_tiff(
    file_path: str,
    *,
    return_details: bool = False,
    return_zyx: bool = True,
    page_index: int = 0,
    allow_guess_unit_when_missing: bool = False,
    anisotropy_warn_threshold: float = 0.01,
    z_anisotropy_warn_threshold: float = 3.0,  
):
    """
    Extract voxel size from TIFF in µm/px.

    Priority:
      1) OME-XML PhysicalSizeX/Y/Z (+ units)
      2) TIFF XResolution/YResolution (ResolutionUnit / ImageJ unit hints)
      3) ImageJ spacing (+ unit) for Z

    Returns:
      - if return_zyx=False: mean_xy_um_per_px (float) or None
      - if return_zyx=True: (z_um, y_um, x_um) tuple (z may be None) or None
      - optionally details dict
    """
    details = VoxelSizeDetails()

    with TiffFile(file_path) as tif:
        page, tags = _get_page_and_tags(tif, page_index=page_index)
        desc = None
        if tags is not None and "ImageDescription" in tags:
            try:
                desc = _decode_tag_value(tags["ImageDescription"].value)
            except Exception:
                desc = None

        ome = getattr(tif, "ome_metadata", None)
        if ome:
            x_um, y_um, z_um = _parse_ome_physical_sizes_um(ome)
            if x_um is not None and y_um is not None:
                details.source = "OME-XML"
                details.x_um, details.y_um, details.z_um = x_um, y_um, z_um

        if details.source is None and desc and ("<ome" in desc.lower() or "<ome:ome" in desc.lower()):
            x_um, y_um, z_um = _parse_ome_physical_sizes_um(desc)
            if x_um is not None and y_um is not None:
                details.source = "OME-XML (ImageDescription)"
                details.x_um, details.y_um, details.z_um = x_um, y_um, z_um

        if details.source is None:
            x_um, y_um, note = _parse_tiff_resolution_tags_um(
                tif,
                page_index=page_index,
                allow_guess_unit_when_missing=allow_guess_unit_when_missing,
            )
            if x_um is not None and y_um is not None:
                details.source = "Resolution tags"
                details.x_um, details.y_um = x_um, y_um
                if note:
                    details.note = note

        # Safely extract ImageJ metadata and fallback to direct ImageDescription parsing
        if details.z_um is None:
            ij = getattr(tif, "imagej_metadata", None)
            if isinstance(ij, dict):
                z = ij.get("spacing", None)
                unit = ij.get("unit", None)
                z_um = _to_um(z, unit)
                if z_um is not None:
                    details.z_um = z_um
                    details.source = "ImageJ (z-only)"

            # Fallback: check the generic ImageDescription key-values for spacing
            if details.z_um is None and desc:
                kv = _parse_image_description_kv(desc)
                if "spacing" in kv:
                    z_um = _to_um(kv["spacing"], kv.get("unit"))
                    if z_um is not None:
                        details.z_um = z_um
                        details.source = "ImageDescription (spacing)"

    if details.x_um is None or details.y_um is None:
        if return_details:
            if details.note is None:
                details.note = "Could not infer XY pixel size from OME-XML or resolution tags (incl. ImageJ unit hints)."
            return (None, details.to_dict())
        return None

    x_um = float(details.x_um)
    y_um = float(details.y_um)

    mean_xy = float(np.mean([x_um, y_um]))
    rel = abs(x_um - y_um) / max(mean_xy, 1e-12)
    if rel > float(anisotropy_warn_threshold):
        msg = f"Anisotropic XY pixel size: x={x_um:.6g} µm, y={y_um:.6g} µm (rel={rel:.3g})."
        details.note = (details.note + " " if details.note else "") + msg

    if details.z_um is not None:
        z_um = float(details.z_um)
        rel_z = abs(z_um - mean_xy) / max(mean_xy, 1e-12)
        if rel_z > float(z_anisotropy_warn_threshold):
            msg = (
                f"Anisotropic Z spacing: z={z_um:.6g} µm vs mean_xy={mean_xy:.6g} µm "
                f"(rel={rel_z:.3g})."
            )
            details.note = (details.note + " " if details.note else "") + msg

    value = (float(details.z_um) if details.z_um is not None else None, y_um, x_um) if return_zyx else mean_xy

    if return_details:
        return (value, details.to_dict())
    return value


def _validate_voxel_size_um_zyx(
    voxel_size_um_zyx: Tuple[float, float, float],
    *,
    name: str,
) -> Tuple[float, float, float]:
    try:
        values = tuple(float(v) for v in voxel_size_um_zyx)
    except Exception as exc:
        raise ValueError(f"{name} must contain exactly 3 numeric values in ZYX order (µm/px).") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values in ZYX order (µm/px).")
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"{name} must contain positive finite values in ZYX order (µm/px).")
    return values


def require_voxel_size_um_zyx(
    path: str | Path,
    fallback: Optional[Tuple[float, float, float]] = None,
    allow_missing_z: bool = False,
) -> Tuple[float, float, float]:
    """
    Return strict voxel size in ZYX order (µm/px) for matching/export workflows.

    If TIFF metadata is missing/incomplete and no fallback is provided, a ValueError is raised
    with instructions because matching/export thresholds are specified in physical units (µm).

    Notes
    -----
    ``allow_missing_z`` is retained for backward compatibility but no longer fabricates ``z_um``
    from in-plane spacing. Z spacing must come from metadata or an explicit fallback.
    """
    path_obj = Path(path)
    voxel, details = get_voxel_size_um_from_tiff(str(path_obj), return_zyx=True, return_details=True)

    if voxel is not None:
        z_um, y_um, x_um = voxel
        if y_um is not None and x_um is not None:
            if z_um is None:
                if allow_missing_z:
                    _LOG.warning(
                        "allow_missing_z=True was requested for '%s', but missing Z spacing is no longer auto-filled from XY spacing.",
                        path_obj,
                    )
            else:
                return _validate_voxel_size_um_zyx((z_um, y_um, x_um), name="metadata voxel size")

    if fallback is not None:
        return _validate_voxel_size_um_zyx(fallback, name="fallback")

    note = details.get("note") if isinstance(details, dict) else None
    note_txt = f" Details: {note}" if note else ""
    raise ValueError(
        "Voxel size metadata is missing or incomplete for "
        f"'{path_obj}'. Provide fallback=(z_um, y_um, x_um) in µm/px or fix TIFF metadata. "
        "NucleiSky3D matching/export uses µm thresholds, so voxel sizes are required."
        f"{note_txt}"
    )


def make_result_dir(big_image_path=None, root_dir=None, tag="NucleiSky3D"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if root_dir is not None:
        root = Path(root_dir)
    else:
        if big_image_path:
            try:
                root = Path(big_image_path).expanduser().resolve().parent
            except Exception:
                root = Path.cwd()
        else:
            root = Path.cwd()

    out = root / f"{tag}_results_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_tiff(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(path), np.asarray(arr))


def save_tiff_zyx(
    path,
    arr,
    voxel_size_um_zyx=None,
    axes="ZYX",
    unit="um",
    bigtiff="auto",
    compress=0,
):
    """Write a 3D TIFF from a pre-ordered ZYX volume.

    This function expects ``arr`` to already be ordered as ``(Z, Y, X)`` and does
    not transpose or reorder data. When voxel spacing is provided as
    ``(z_um, y_um, x_um)`` in µm/px, ImageJ-compatible metadata is embedded
    (using ``imagej=True`` when dtype permits) with:

    - ``metadata['spacing']`` for Z spacing,
    - ``metadata['unit']`` for physical unit, and
    - TIFF ``resolution=(1/x_um, 1/y_um)`` for in-plane pixel spacing.

    Parameters
    ----------
    path : str | pathlib.Path
        Destination TIFF path.
    arr : array-like
        3D array in ZYX order.
    voxel_size_um_zyx : tuple[float, float, float] | None
        Physical voxel size ``(z_um, y_um, x_um)`` in µm/px.
    axes : str
        Axes metadata string, default ``"ZYX"``.
    unit : str
        Physical unit label stored in ImageJ metadata.
    bigtiff : bool | "auto"
        If ``"auto"``, BigTIFF is enabled when data size is >= 4 GiB.
    compress : int | str
        Compression setting forwarded to ``tifffile.imwrite`` as
        ``compression`` when non-zero.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(arr)
    if data.ndim != 3:
        raise ValueError(f"save_tiff_zyx expects a 3D ZYX array, got shape {data.shape}.")

    if np.issubdtype(data.dtype, np.integer):
        if data.dtype not in (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ):
            raise TypeError(f"Unsupported integer dtype for TIFF labels: {data.dtype}.")
    elif not np.issubdtype(data.dtype, np.floating):
        raise TypeError(f"save_tiff_zyx supports integer labels or float images, got {data.dtype}.")

    if bigtiff == "auto":
        bigtiff_flag = bool(data.nbytes >= (4 * 1024**3))
    else:
        bigtiff_flag = bool(bigtiff)

    metadata = {"axes": str(axes)}
    imagej_supported_dtype = data.dtype in (np.uint8, np.uint16, np.int16, np.float32)
    kwargs = {
        "imagej": bool(imagej_supported_dtype),
        "metadata": metadata,
        "bigtiff": bigtiff_flag,
        "photometric": "minisblack",
    }

    if compress not in (None, 0, "0", False):
        kwargs["compression"] = compress

    if voxel_size_um_zyx is not None:
        z_um, y_um, x_um = _validate_voxel_size_um_zyx(voxel_size_um_zyx, name="voxel_size_um_zyx")
        if y_um <= 0 or x_um <= 0:
            raise ValueError("voxel_size_um_zyx requires positive y_um and x_um.")
        metadata.update({"unit": unit, "spacing": float(z_um)})
        kwargs["resolution"] = (1.0 / float(x_um), 1.0 / float(y_um))

    try:
        imwrite(str(path), data, **kwargs)
    except ValueError as exc:
        if "ImageJ format does not support data type" not in str(exc):
            raise
        kwargs["imagej"] = False
        imwrite(str(path), data, **kwargs)


def _safe_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _parse_text_value(s: str):
    s = (s or "").strip()
    if s.lower() in ("none", "null", ""):
        return None
    s_norm = s.replace("_", "")
    try:
        if "." not in s_norm and "e" not in s_norm.lower():
            return int(s_norm)
    except Exception:
        pass
    try:
        return float(s_norm)
    except Exception:
        return s


def _ensure_zyx(arr: Any, *, channel_axis: Optional[int] = None, channel_index: int = 0) -> Any:
    if arr.ndim == 3:
        return arr
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D or 4D volume, got shape {arr.shape}.")

    axis = channel_axis
    if axis is None:
        if arr.shape[0] <= 4:
            axis = 0
        elif arr.shape[-1] <= 4:
            axis = -1

    if axis is None:
        raise ValueError(
            f"Ambiguous 4D volume shape {arr.shape}. Provide channel_axis to select a channel."
        )

    axis = int(axis)
    if axis < -arr.ndim or axis >= arr.ndim:
        raise ValueError(f"channel_axis={axis} is out of bounds for shape {arr.shape}.")
    axis = axis % arr.ndim

    idx = int(channel_index)
    if idx < 0 or idx >= int(arr.shape[axis]):
        raise ValueError(
            f"channel_index={idx} is out of bounds for channel axis {axis} with size {arr.shape[axis]}."
        )

    # Use slicing instead of np.take() to preserve lazy loading (Zarr/Dask arrays)
    slices = [slice(None)] * arr.ndim
    slices[axis] = idx
    vol = arr[tuple(slices)]
    
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume after channel selection, got shape {vol.shape}.")
    return vol


def load_image(path_str: str):
    """
    Load image from path using appropriate backend.
    Supports TIFF (tifffile), Numpy (npy), OME-Zarr (zarr),
    and standard formats (scikit-image fallback).

    Returns:
        numpy.ndarray, zarr.Array, or zarr.Group
    """
    p = _safe_path(path_str)

    if not p.exists():
        raise FileNotFoundError(f"Image path does not exist: {p}")

    suf = p.suffix.lower()

    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(p))

    if suf == ".npy":
        return np.load(str(p))

    if _is_zarr_store_path(p) or p.is_dir():
        try:
            import zarr
        except ImportError:
            if _is_zarr_store_path(p):
                raise ImportError("Loading .zarr files requires the 'zarr' library. Please `pip install zarr`.")
        else:
            try:
                store = zarr.open(str(p), mode="r")
            except Exception:
                if _is_zarr_store_path(p):
                    raise
            else:
                if isinstance(store, zarr.Group):
                    if "0" in store:
                        return store["0"]
                    if "s0" in store:
                        return store["s0"]
                    return store
                return store

    try:
        return skio.imread(str(p))
    except Exception:
        pass

    raise ValueError(f"Unsupported file extension '{suf}' and scikit-image fallback failed.")


def load_volume(path_str: str, *, channel_axis: Optional[int] = None, channel_index: int = 0) -> Any:
    """
    Load a 3D volume and coerce to Z, Y, X ordering.
    Returns a lazy zarr.Array if supported, or np.ndarray.
    """
    arr = load_image(path_str)
    return _ensure_zyx(arr, channel_axis=channel_axis, channel_index=channel_index)


def inspect_volume_header(path_str: str) -> dict:
    """
    Inspect 3D volume metadata (shape, dtype, axes, voxel size) without loading data.

    Supports TIFF (via tifffile), NPY, and OME-Zarr when available.
    """
    p = _safe_path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suf = p.suffix.lower()
    info = {
        "path": str(p),
        "kind": suf,
        "shape": None,
        "dtype": None,
        "axes": None,
        "inferred_zyx": None,
        "voxel_size_um": None,
        "voxel_size_um_zyx": None,
        "ome_physical_size_um": None,
        "voxel_details": None,
    }

    def _infer_zyx_from_axes(axes: Optional[str], shape: Optional[Tuple[int, ...]]):
        if not axes or not shape:
            return None
        axes_u = axes.upper().strip()
        if len(axes_u) != len(shape):
            return None
        mapping = {ax: dim for ax, dim in zip(axes_u, shape)}
        if not all(k in mapping for k in ("Z", "Y", "X")):
            return None
        return {
            "axes": "ZYX",
            "shape": (mapping["Z"], mapping["Y"], mapping["X"]),
            "source": "axes",
        }

    if suf in (".tif", ".tiff"):
        with TiffFile(str(p)) as tif:
            series = tif.series[0]
            info["shape"] = tuple(series.shape)
            info["dtype"] = str(series.dtype)

            axes = getattr(series, "axes", None)
            if isinstance(axes, str):
                info["axes"] = axes.upper().strip()

            ij = getattr(tif, "imagej_metadata", None)
            if isinstance(ij, dict):
                ij_axes = ij.get("axes", None)
                if isinstance(ij_axes, str):
                    info["axes"] = ij_axes.upper().strip()

            ome = getattr(tif, "ome_metadata", None)
            ome_xml = ome
            if not ome_xml:
                try:
                    page, tags = _get_page_and_tags(tif, page_index=0)
                    if tags is not None and "ImageDescription" in tags:
                        desc = _decode_tag_value(tags["ImageDescription"].value)
                        if desc and ("<ome" in desc.lower() or "<ome:ome" in desc.lower()):
                            ome_xml = desc
                except Exception:
                    ome_xml = None

            if ome_xml:
                x_um, y_um, z_um = _parse_ome_physical_sizes_um(ome_xml)
                if x_um is not None or y_um is not None or z_um is not None:
                    info["ome_physical_size_um"] = {"x": x_um, "y": y_um, "z": z_um}

            voxel, details = get_voxel_size_um_from_tiff(
                str(p),
                return_details=True,
                return_zyx=True,
            )
            info["voxel_size_um_zyx"] = voxel
            if voxel is not None:
                info["voxel_size_um"] = float(np.mean([v for v in voxel if v is not None]))
            info["voxel_details"] = details

    elif suf == ".npy":
        arr = np.load(str(p), mmap_mode="r")
        info["shape"] = tuple(arr.shape)
        info["dtype"] = str(arr.dtype)
        if arr.ndim == 3:
            info["axes"] = "ZYX"
            info["inferred_zyx"] = {"axes": "ZYX", "shape": tuple(arr.shape), "source": "ndim=3"}

    elif _is_zarr_store_path(p):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("Inspecting .zarr files requires the 'zarr' library. Please `pip install zarr`.") from exc

        z = zarr.open(str(p), mode="r")
        arr = None
        axes_str = None
        voxel_zyx = None

        if isinstance(z, zarr.Group):
            ms = z.attrs.get("multiscales", [])
            if ms:
                ms0 = ms[0]
                axes_meta = ms0.get("axes", [])
                names = []
                for a in axes_meta:
                    if isinstance(a, dict):
                        names.append(str(a.get("name", "")).lower())
                    else:
                        names.append(str(a).lower())
                name_to_letter = {"t": "T", "z": "Z", "c": "C", "y": "Y", "x": "X"}
                axes_str = "".join([name_to_letter.get(n, "") for n in names if n in name_to_letter]) or None
                ds0 = ms0["datasets"][0]["path"]
                arr = z[ds0]

                ct = ms0["datasets"][0].get("coordinateTransformations", [])
                if ct and axes_str:
                    scale = ct[0].get("scale", None)
                    if isinstance(scale, (list, tuple)) and len(scale) == len(axes_str):
                        axis_scale = {ax: float(sc) for ax, sc in zip(axes_str, scale)}
                        if all(ax in axis_scale for ax in ("Z", "Y", "X")):
                            voxel_zyx = (axis_scale["Z"], axis_scale["Y"], axis_scale["X"])
            else:
                keys = list(z.array_keys())
                arr = z[keys[0]] if keys else None
        else:
            arr = z

        if arr is not None:
            info["shape"] = tuple(arr.shape)
            info["dtype"] = str(arr.dtype)
        if axes_str is not None:
            info["axes"] = axes_str
        if voxel_zyx is not None:
            info["voxel_size_um_zyx"] = voxel_zyx
            info["voxel_size_um"] = float(np.mean([v for v in voxel_zyx if v is not None]))

    info["inferred_zyx"] = _infer_zyx_from_axes(info.get("axes"), info.get("shape")) or info.get("inferred_zyx")
    return info


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_sanitize(obj):
    """Recursively convert NumPy/Path objects to JSON-serializable Python types.

    ``json.dump(..., default=_json_default)`` handles conversion on write, but callers
    (notebooks/CLI wrappers) often also serialize the returned record directly to JSONL.
    Returning JSON-safe dicts avoids ``TypeError: Object of type ndarray...``.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def append_transform_jsonl(record: dict, out_jsonl: str | Path):
    """Append a single transform record as one line in a JSONL file."""
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")


def save_nucleisky_transform_3d(
    res: dict,
    out_path: str | Path | None,
    *,
    pixel_size_full_um_zyx,
    pixel_size_crop_um_zyx,
    matcher_name: str = "unknown",
    require_success: bool = True,
) -> dict:
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    success = bool(res.get("success", False))
    mq = res.get("match_quality", None)
    if isinstance(mq, dict) and "success" in mq:
        success = bool(mq["success"])

    if require_success and not success:
        raise ValueError(
            f"Match '{matcher_name}' is not successful (success={success}). "
            f"Set require_success=False if you still want to export it."
        )

    scale = float(res["best_scale"])
    R_zyx = np.asarray(res["best_R"], float).reshape(3, 3)
    t_um_zyx = np.asarray(res["best_t"], float).reshape(3,)

    A_px, b_px = similarity_um_to_affine_px_3d(
        best_scale=scale,
        best_R=R_zyx,
        best_t=t_um_zyx,
        pixel_size_full_um=pixel_size_full_um_zyx,
        pixel_size_crop_um=pixel_size_crop_um_zyx,
    )

    bbox = res.get("best_bbox", None)
    if bbox is not None:
        bbox = tuple(int(v) for v in bbox)
        if len(bbox) != 6:
            raise ValueError("res['best_bbox'] must be length-6 (z0,z1,y0,y1,x0,x1) or None.")

    record = {
        "matcher": matcher_name,
        "success": bool(success),
        "scale": float(scale),
        "R_zyx": R_zyx.tolist(),
        "t_um_zyx": t_um_zyx.tolist(),
        "pixel_size_crop_um_zyx": np.asarray(pixel_size_crop_um_zyx, float).reshape(3,).tolist(),
        "pixel_size_full_um_zyx": np.asarray(pixel_size_full_um_zyx, float).reshape(3,).tolist(),
        "A_px": np.asarray(A_px, float).tolist(),
        "b_px": np.asarray(b_px, float).tolist(),
        "bbox_full_px_z0z1y0y1x0x1": list(bbox) if bbox is not None else None,
        "match_quality": res.get("match_quality", None),
    }

    # Make returned record fully JSON-safe (match_quality can contain NumPy arrays).
    record = _json_sanitize(record)

    if out_path is not None:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=_json_default)

    return record


def _parse_positive_vec3(name: str, v):
    try:
        arr = np.asarray(v, float).reshape(3,)
    except Exception as e:
        raise ValueError(f"{name} must be a length-3 positive finite vector.") from e
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{name} must be a length-3 positive finite vector.")
    return arr


def _validate_transform_record_3d(rec: dict) -> None:
    for k in ("scale", "R_zyx", "t_um_zyx", "A_px", "b_px", "pixel_size_full_um_zyx", "pixel_size_crop_um_zyx"):
        if rec.get(k) is None:
            raise ValueError(f"Missing required 3D transform field '{k}'.")

    try:
        sc = float(rec["scale"])
    except Exception as e:
        raise ValueError("scale must be numeric.") from e
    if not np.isfinite(sc) or sc <= 0:
        raise ValueError("scale must be positive and finite.")

    try:
        R = np.asarray(rec["R_zyx"], float).reshape(3, 3)
    except Exception as e:
        raise ValueError("R_zyx must be shape (3,3).") from e
    if not np.all(np.isfinite(R)):
        raise ValueError("R_zyx contains non-finite values.")

    try:
        t = np.asarray(rec["t_um_zyx"], float).reshape(3,)
    except Exception as e:
        raise ValueError("t_um_zyx must be length 3.") from e
    if not np.all(np.isfinite(t)):
        raise ValueError("t_um_zyx contains non-finite values.")

    try:
        A = np.asarray(rec["A_px"], float).reshape(3, 3)
    except Exception as e:
        raise ValueError("A_px must be shape (3,3).") from e
    if not np.all(np.isfinite(A)):
        raise ValueError("A_px contains non-finite values.")

    try:
        b = np.asarray(rec["b_px"], float).reshape(3,)
    except Exception as e:
        raise ValueError("b_px must be length 3.") from e
    if not np.all(np.isfinite(b)):
        raise ValueError("b_px contains non-finite values.")

    _parse_positive_vec3("pixel_size_full_um_zyx", rec["pixel_size_full_um_zyx"])
    _parse_positive_vec3("pixel_size_crop_um_zyx", rec["pixel_size_crop_um_zyx"])

    if rec.get("bbox_full_px_z0z1y0y1x0x1") is not None:
        try:
            bb = np.asarray(rec["bbox_full_px_z0z1y0y1x0x1"], float).reshape(6,)
        except Exception as e:
            raise ValueError("bbox_full_px_z0z1y0y1x0x1 must be length 6.") from e
        if not np.all(np.isfinite(bb)):
            raise ValueError("bbox_full_px_z0z1y0y1x0x1 contains non-finite values.")
        z0, z1, y0, y1, x0, x1 = bb.tolist()
        if z1 < z0 or y1 < y0 or x1 < x0:
            raise ValueError("bbox_full_px_z0z1y0y1x0x1 has invalid ordering.")


def load_nucleisky_transform_3d(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        rec = json.load(f)
    for k in ("A_px", "b_px", "pixel_size_full_um_zyx", "pixel_size_crop_um_zyx"):
        if k not in rec:
            raise ValueError(f"Transform JSON missing key '{k}'")
    if rec.get("scale") is None and rec.get("best_scale") is not None:
        rec["scale"] = rec["best_scale"]
    if rec.get("R_zyx") is None and rec.get("best_R") is not None:
        rec["R_zyx"] = np.asarray(rec["best_R"], float).reshape(3, 3).tolist()
    if rec.get("t_um_zyx") is None and rec.get("best_t") is not None:
        rec["t_um_zyx"] = np.asarray(rec["best_t"], float).reshape(3,).tolist()
    if rec.get("bbox_full_px_z0z1y0y1x0x1") is None and rec.get("best_bbox") is not None:
        rec["bbox_full_px_z0z1y0y1x0x1"] = [int(v) for v in rec["best_bbox"]]
    _validate_transform_record_3d(rec)
    return rec


def load_transforms_any_3d(path_str: str) -> list[dict]:
    """
    Returns list of records. Accepts:
      - single JSON file
      - JSONL (one JSON object per line)
    """
    p = _safe_path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Transform file not found: {p}")

    suf = p.suffix.lower()
    recs = []

    def _normalize_transform_record_3d(rec: dict) -> dict:
        out = dict(rec)

        if out.get("scale") is None and out.get("best_scale") is not None:
            out["scale"] = out["best_scale"]
        if out.get("R_zyx") is None and out.get("best_R") is not None:
            out["R_zyx"] = np.asarray(out["best_R"], float).reshape(3, 3).tolist()
        if out.get("t_um_zyx") is None and out.get("best_t") is not None:
            out["t_um_zyx"] = np.asarray(out["best_t"], float).reshape(3,).tolist()

        if out.get("pixel_size_full_um_zyx") is None:
            for alias in ("pixel_size_full_orig_um_zyx", "voxel_size_full_um_zyx"):
                if out.get(alias) is not None:
                    out["pixel_size_full_um_zyx"] = np.asarray(out[alias], float).reshape(3,).tolist()
                    break
        if out.get("pixel_size_crop_um_zyx") is None:
            for alias in ("pixel_size_crop_orig_um_zyx", "pixel_size_patch_um_zyx", "voxel_size_crop_um_zyx"):
                if out.get(alias) is not None:
                    out["pixel_size_crop_um_zyx"] = np.asarray(out[alias], float).reshape(3,).tolist()
                    break

        if out.get("bbox_full_px_z0z1y0y1x0x1") is None and out.get("best_bbox") is not None:
            out["bbox_full_px_z0z1y0y1x0x1"] = [int(v) for v in out["best_bbox"]]

        has_core = all(
            out.get(k) is not None for k in (
                "scale", "R_zyx", "t_um_zyx", "pixel_size_full_um_zyx", "pixel_size_crop_um_zyx"
            )
        )
        if has_core and (out.get("A_px") is None or out.get("b_px") is None):
            A_px, b_px = similarity_um_to_affine_px_3d(
                best_scale=float(out["scale"]),
                best_R=np.asarray(out["R_zyx"], float).reshape(3, 3),
                best_t=np.asarray(out["t_um_zyx"], float).reshape(3,),
                pixel_size_full_um=np.asarray(out["pixel_size_full_um_zyx"], float).reshape(3,),
                pixel_size_crop_um=np.asarray(out["pixel_size_crop_um_zyx"], float).reshape(3,),
            )
            out["A_px"] = np.asarray(A_px, float).tolist()
            out["b_px"] = np.asarray(b_px, float).tolist()

        return out

    if suf == ".json":
        rec = _normalize_transform_record_3d(load_nucleisky_transform_3d(p))
        _validate_transform_record_3d(rec)
        rec = dict(rec)
        rec["_source_path"] = str(p)
        rec["_source_kind"] = "json"
        rec["_line"] = 0
        recs.append(rec)
        return recs

    if suf == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                if not isinstance(rec, dict):
                    continue
                rec = _normalize_transform_record_3d(rec)
                _validate_transform_record_3d(rec)
                rec["_source_path"] = str(p)
                rec["_source_kind"] = "jsonl"
                rec["_line"] = i + 1
                recs.append(rec)
        return recs

    raise ValueError("Transform must be .json or .jsonl")
