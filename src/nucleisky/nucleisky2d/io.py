
""" io.py I/O helpers: TIFF metadata, JSON, and transform persistence."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
from skimage import io as skio
from tifffile import TiffFile, imread, imwrite

from .types import PixelSizeDetails, BBox
from .matching.geometry import similarity_um_to_affine_px, rotation_deg_from_R


_UM_PER_UNIT = {
    # metric
    "m": 1e6, "meter": 1e6, "metre": 1e6,
    "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3,
    "cm": 1e4, "centimeter": 1e4, "centimetre": 1e4,
    "um": 1.0, "µm": 1.0, "μm": 1.0,
    "micron": 1.0, "microns": 1.0,
    "micrometer": 1.0, "micrometre": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3,
    # imperial
    "in": 25400.0, "inch": 25400.0, "inches": 25400.0,
}

_ZARR_MARKER_FILES = (".zgroup", ".zarray", ".zattrs", ".zmetadata", "zarr.json")


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
    s = s.replace("μ", "µ")  # greek mu -> micro sign
    s = s.lower()
    s = s.replace("micrometers", "micrometer").replace("micrometres", "micrometre")
    s = s.replace("microns", "micron")
    # common ImageJ writes "um"
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
    """
    tifffile may return rationals as:
      - (num, den)
      - fractions.Fraction
      - numpy scalars
      - objects with numerator/denominator
    """
    if v is None:
        return None

    # tuple rational
    if isinstance(v, tuple) and len(v) == 2:
        num, den = v
        try:
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        except Exception:
            return None

    # Fraction-like
    if hasattr(v, "numerator") and hasattr(v, "denominator"):
        try:
            den = float(v.denominator)
            if den == 0:
                return None
            return float(v.numerator) / den
        except Exception:
            return None

    # string like "300/1"
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
    """
    Parse ImageDescription-style key=value lines (ImageJ and similar).
    Returns lowercase keys.
    """
    if not desc:
        return {}

    # ImageJ descriptions are typically lines separated by \n
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
    """
    Returns (x_um, y_um, z_um) from OME-XML PhysicalSizeX/Y/Z (+ units) if available.
    """
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
    """
    Compute µm/px from XResolution/YResolution and a unit source.

    - If resolution_unit_code in {2,3} use inch/cm.
    - If resolution_unit_code == 1 (None), try imagej_unit_hint (e.g. micron) if provided.
    - If still unknown and allow_guess_unit_when_missing=True, guess inch.
    """
    if tags is None:
        return (None, None, "No tags")

    if "XResolution" not in tags or "YResolution" not in tags:
        return (None, None, "Missing XResolution/YResolution tags")

    xres = _ratio_to_float(tags["XResolution"].value)  # pixels per unit
    yres = _ratio_to_float(tags["YResolution"].value)
    if xres is None or yres is None or xres <= 0 or yres <= 0:
        return (None, None, "Invalid XResolution/YResolution values")

    unit_um = None
    note = None

    # Standard TIFF units
    if int(resolution_unit_code) == 2:
        unit_um = 25400.0
    elif int(resolution_unit_code) == 3:
        unit_um = 10000.0
    else:
        # ImageJ often stores true unit separately (e.g. "micron") while ResolutionUnit==1
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
    """
    Robust TIFF XY pixel size reader:
      - If ResolutionUnit=inch/cm -> use TIFF spec
      - If ResolutionUnit=1/None -> attempt ImageJ unit from imagej_metadata or ImageDescription
    """
    page, tags = _get_page_and_tags(tif, page_index=page_index)
    if page is None:
        return (None, None, "Invalid page_index")

    # Pull unit hint from ImageJ metadata and/or ImageDescription
    ij = getattr(tif, "imagej_metadata", None) or {}
    ij_unit = ij.get("unit", None)

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


def get_pixel_size_um_from_tiff(
    file_path: str,
    *,
    return_details: bool = False,
    return_xy: bool = False,
    page_index: int = 0,
    allow_guess_unit_when_missing: bool = False,
    anisotropy_warn_threshold: float = 0.01,
):
    """
    Extracts pixel size from TIFF in µm/px.

    Priority:
      1) OME-XML PhysicalSizeX/Y/Z (+ units) from tif.ome_metadata
      1b) OME-XML inside ImageDescription (common when ome_metadata is not exposed)
      2) TIFF XResolution/YResolution interpreted using:
           - ResolutionUnit (inch/cm), OR
           - ImageJ unit (unit=micron etc) when ResolutionUnit==1
      3) ImageJ spacing (+ unit) for Z only

    Returns:
      - if return_xy=False: mean_xy_um_per_px (float) or None
      - if return_xy=True: (x_um, y_um) tuple or None
      - optionally details dict
    """
    details = PixelSizeDetails()

    with TiffFile(file_path) as tif:
        # --- Fetch ImageDescription early (needed for OME-in-description and ImageJ unit hints)
        page, tags = _get_page_and_tags(tif, page_index=page_index)
        desc = None
        if tags is not None and "ImageDescription" in tags:
            try:
                desc = _decode_tag_value(tags["ImageDescription"].value)
            except Exception:
                desc = None

        # 1) OME metadata via tifffile property
        ome = getattr(tif, "ome_metadata", None)
        if ome:
            x_um, y_um, z_um = _parse_ome_physical_sizes_um(ome)
            if x_um is not None and y_um is not None:
                details.source = "OME-XML"
                details.x_um, details.y_um, details.z_um = x_um, y_um, z_um

        # 1b) OME-XML embedded in ImageDescription (tifffile doesn't always expose ome_metadata)
        if details.source is None and desc and ("<ome" in desc.lower() or "<ome:ome" in desc.lower()):
            x_um, y_um, z_um = _parse_ome_physical_sizes_um(desc)
            if x_um is not None and y_um is not None:
                details.source = "OME-XML (ImageDescription)"
                details.x_um, details.y_um, details.z_um = x_um, y_um, z_um

        # 2) TIFF resolution tags (with ImageJ unit hint support when ResolutionUnit==1)
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

        # 3) ImageJ z-spacing fallback (often z only)
        if details.z_um is None and getattr(tif, "imagej_metadata", None):
            ij = tif.imagej_metadata or {}
            z = ij.get("spacing", None)
            unit = ij.get("unit", None)
            z_um = _to_um(z, unit)
            if z_um is not None:
                details.z_um = z_um
                if details.source is None:
                    details.source = "ImageJ (z-only)"

    # If no XY available
    if details.x_um is None or details.y_um is None:
        if return_details:
            if details.note is None:
                details.note = "Could not infer XY pixel size from OME-XML or resolution tags (incl. ImageJ unit hints)."
            return (None, details.to_dict())
        return None

    x_um = float(details.x_um)
    y_um = float(details.y_um)

    # Anisotropy check
    mean_xy = float(np.mean([x_um, y_um]))
    rel = abs(x_um - y_um) / max(mean_xy, 1e-12)
    if rel > float(anisotropy_warn_threshold):
        msg = f"Anisotropic XY pixel size: x={x_um:.6g} µm, y={y_um:.6g} µm (rel={rel:.3g})."
        details.note = (details.note + " " if details.note else "") + msg

    value = (x_um, y_um) if return_xy else mean_xy

    if return_details:
        return (value, details.to_dict())
    return value


def _resolve_pixel_size_um_per_px(
    *,
    manual_flag: bool,
    manual_value: Optional[float],
    meta_value: Optional[float],
    fallback_value: float,
    label: str,
) -> Tuple[float, str, Optional[float]]:
    """
    Decide pixel size µm/px with provenance.

    Returns:
      (pix_um, source_str, meta_pix_um_or_none)
    """
    if manual_flag:
        if manual_value is None:
            raise ValueError(f"{label}: manual_flag=True but manual_value is None.")
        pix = float(manual_value)
        if pix <= 0:
            raise ValueError(f"{label}: manual pixel size must be > 0 (µm/px).")
        return pix, "manual", (float(meta_value) if meta_value is not None else None)

    if meta_value is not None:
        mv = float(meta_value)
        if mv > 0 and np.isfinite(mv):
            return mv, "metadata", mv

    pix = float(fallback_value)
    if not np.isfinite(pix) or pix <= 0:
        pix = 1.0
    return pix, "fallback", None


def make_result_dir(big_image_path=None, root_dir=None, tag="NucleiSky"):
    """
    Create a timestamped results folder.
    If root_dir is None:
        - if big_image_path exists: place folder next to big image
        - else: use current working directory
    """
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


def _safe_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _parse_text_value(s: str):
    s = (s or "").strip()
    if s.lower() in ("none", "null", ""):
        return None

    # allow underscore formatting like 100_000
    s_norm = s.replace("_", "")

    # try int
    try:
        if "." not in s_norm and "e" not in s_norm.lower():
            return int(s_norm)
    except Exception:
        pass

    # try float
    try:
        return float(s_norm)
    except Exception:
        return s

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

    # 1. TIFF / TIF
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(p))
    
    # 2. Numpy
    if suf == ".npy":
        return np.load(str(p))

    # 3. OME-Zarr / Zarr
    # Support both *.zarr paths and marker-based directories.
    # Also probe generic directories, since some valid stores may be folder-based
    # without explicit top-level marker files.
    if _is_zarr_store_path(p) or p.is_dir():
        try:
            import zarr
        except ImportError:
            if _is_zarr_store_path(p):
                raise ImportError("Loading .zarr files requires the 'zarr' library. Please `pip install zarr`.")
        else:
            try:
                # Open in read-only mode for safety
                store = zarr.open(str(p), mode='r')
            except Exception:
                # Not a readable zarr store: fall through to skimage fallback.
                # For known zarr-like paths, surface the original failure.
                if _is_zarr_store_path(p):
                    raise
            else:
                # If it's a Group (OME-NGFF), attempt to return the high-res array
                if isinstance(store, zarr.Group):
                    # OME-NGFF v0.4+ stores scales as "0", "1", "2"...
                    if "0" in store:
                        return store["0"]
                    # Older conventions might use "s0"
                    if "s0" in store:
                        return store["s0"]

                    # If we can't auto-resolve, return the group so user can pick
                    return store

                # If it's already an Array (plain zarr)
                return store

    # 4. Fallback (scikit-image)
    try:
        return skio.imread(str(p))
    except Exception:
        pass

    raise ValueError(f"Unsupported file extension '{suf}' and scikit-image fallback failed.")


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
    sometimes also serialize the returned record (e.g. to JSONL). Returning a fully
    JSON-safe dict avoids surprises.
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


def save_nucleisky_transform(
    res: dict,
    out_path: str | Path,
    *,
    matcher_name: str = "unknown",
    pixel_size_full_um: float,
    pixel_size_crop_um: float,
    require_success: bool = True,
):
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
    R_yx = np.asarray(res["best_R"], float).reshape(2, 2)
    t_um_yx = np.asarray(res["best_t"], float).reshape(2,)

    A_px, b_px = similarity_um_to_affine_px(
        scale, R_yx, t_um_yx,
        pixel_size_src_um=float(pixel_size_crop_um),
        pixel_size_dst_um=float(pixel_size_full_um),
    )

    bbox = res.get("bbox_full_px", None)
    if bbox is not None and not isinstance(bbox, BBox):
        raise TypeError("res['bbox_full_px'] must be a BBox (canonical) or None.")

    record = {
        "matcher": matcher_name,
        "success": bool(success),
        "scale": float(scale),
        "rotation_deg": rotation_deg_from_R(R_yx),
        "R_yx": R_yx.tolist(),
        "t_um_yx": t_um_yx.tolist(),
        "pixel_size_crop_um": float(pixel_size_crop_um),
        "pixel_size_full_um": float(pixel_size_full_um),
        "A_px": np.asarray(A_px, float).tolist(),
        "b_px": np.asarray(b_px, float).tolist(),
        "bbox_full_px_y0y1x0x1": (list(bbox.as_y0y1x0x1()) if bbox is not None else None),
        "match_quality": res.get("match_quality", None),
    }

    # Make returned record fully JSON-safe (match_quality can contain NumPy arrays).
    record = _json_sanitize(record)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=_json_default)

    return record


def append_transform_jsonl(record: dict, out_jsonl: str | Path):
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def load_nucleisky_transform(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        rec = json.load(f)
    for k in ("A_px", "b_px", "pixel_size_full_um", "pixel_size_crop_um"):
        if k not in rec:
            raise ValueError(f"Transform JSON missing key '{k}'")
    ok, problems = validate_transform_record(rec)
    if not ok:
        raise ValueError("Invalid 2D transform record: " + "; ".join(problems))
    return rec


def load_transforms_any(path_str: str, *, strict: bool = False):
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

    if suf == ".json":
        rec = load_nucleisky_transform(p)
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
                try:
                    rec = json.loads(s)
                except Exception as e:
                    if strict:
                        raise ValueError(f"Invalid JSONL at {p} line {i+1}: {e}") from e
                    raise
                if not isinstance(rec, dict):
                    continue
                if strict:
                    ok, problems = validate_transform_record(rec)
                    if not ok:
                        raise ValueError(
                            f"Invalid 2D transform record at {p} line {i+1}: " + "; ".join(problems)
                        )
                rec["_source_path"] = str(p)
                rec["_source_kind"] = "jsonl"
                rec["_line"] = i + 1
                recs.append(rec)
        return recs

    raise ValueError("Transform must be .json or .jsonl")


def validate_transform_record(rec: dict):
    """
    Hard validation: return (ok, problems[])
    Accept either:
      - similarity params: scale, R_yx, t_um_yx
      - or direct affine: A_px, b_px
    """
    problems = []
    if not isinstance(rec, dict):
        return False, ["Record is not a dict."]

    if "success" in rec and rec["success"] is False:
        problems.append("success=false")

    has_sim = all(k in rec for k in ("scale","R_yx","t_um_yx"))
    has_aff = all(k in rec for k in ("A_px","b_px"))
    if not (has_sim or has_aff):
        problems.append("Missing similarity params (scale/R_yx/t_um_yx) AND missing affine (A_px/b_px).")

    if has_aff:
        try:
            A = np.asarray(rec["A_px"], float).reshape(2, 2)
            b = np.asarray(rec["b_px"], float).reshape(2,)
            if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
                problems.append("A_px or b_px contains non-finite values.")
        except Exception:
            problems.append("Could not parse affine params A_px/b_px as (2,2) and (2,).")

    if has_sim:
        try:
            sc = float(rec["scale"])
            R = np.asarray(rec["R_yx"], float).reshape(2, 2)
            t = np.asarray(rec["t_um_yx"], float).reshape(2,)
            if not np.isfinite(sc) or sc <= 0:
                problems.append("scale is not a positive finite number.")
            if not np.all(np.isfinite(R)):
                problems.append("R_yx contains non-finite values.")
            if not np.all(np.isfinite(t)):
                problems.append("t_um_yx contains non-finite values.")
        except Exception:
            problems.append("Could not parse similarity params as scale float, R_yx (2,2), t_um_yx (2,).")

    for k in ("pixel_size_full_um","pixel_size_crop_um"):
        if k in rec and rec[k] is not None:
            try:
                v = float(rec[k])
                if not np.isfinite(v) or v <= 0:
                    problems.append(f"{k} is not a positive finite number.")
            except Exception:
                problems.append(f"{k} could not be parsed as float.")

    if "bbox_full_px_y0y1x0x1" in rec and rec["bbox_full_px_y0y1x0x1"] is not None:
        try:
            bb = np.asarray(rec["bbox_full_px_y0y1x0x1"], float).reshape(4,)
            if not np.all(np.isfinite(bb)):
                problems.append("bbox_full_px_y0y1x0x1 contains non-finite values.")
            y0, y1, x0, x1 = bb.tolist()
            if y1 < y0 or x1 < x0:
                problems.append("bbox_full_px_y0y1x0x1 has invalid ordering (expected y1>=y0 and x1>=x0).")
        except Exception:
            problems.append("bbox_full_px_y0y1x0x1 must be length-4 finite numeric values.")

    return (len(problems) == 0), problems
