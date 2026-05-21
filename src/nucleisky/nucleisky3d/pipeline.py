"""pipeline.py High-level orchestration for 3D feature extraction and matching."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import time
import traceback
from datetime import datetime, timezone
import json

import numpy as np
from scipy.spatial import cKDTree

from .config import (
    DEFAULT_MATCHER_CONFIG,
    _norm_matcher_name,
    deep_merge_dict,
    effective_flat_config,
    normalize_runtime_overrides,
)
from .export import export_aligned_crop_tiff
from .features import extract_nuclear_features_3d
from .matching.hashing3d import run_geometric_hashing_matching_3d_um
from .matching.geometry import bbox_full_px_from_similarity_um_3d
from .matching.geometry import sanitize_points_zyx_um
from .matching.pyramid import run_pyramid_based_matching_um
from .segmentation import segment_nuclei_2p5d
from .utils import _stable_u32, compute_min_inliers_stable
from .io import append_transform_jsonl


SUPPORTED_MATCHERS = ("pyramid", "hashing")


def _safe_n_points_3d(x) -> int:
    if x is None:
        return 0
    if hasattr(x, "shape") and getattr(x, "shape", None) is not None:
        try:
            return int(x.shape[0])
        except Exception:
            pass
    try:
        return int(len(x))
    except Exception:
        return 0


def _extract_quality_3d(out: Dict[str, Any]) -> Tuple[float, float]:
    q = out.get("match_quality") or {}
    frac = q.get("frac_inliers", 0.0)
    err = q.get("mean_error_um", math.inf)

    try:
        frac = float(frac) if frac is not None and math.isfinite(float(frac)) else 0.0
    except Exception:
        frac = 0.0

    try:
        err = float(err) if err is not None and math.isfinite(float(err)) else math.inf
    except Exception:
        err = math.inf

    return frac, err


def _quality_tuple_3d(rec: dict):
    mq = rec.get("match_quality", {}) if isinstance(rec.get("match_quality", {}), dict) else {}
    frac = mq.get("frac_inliers", None)
    err = mq.get("mean_error_um", None)
    frac = float(frac) if frac is not None and math.isfinite(float(frac)) else -1.0
    err = float(err) if err is not None and math.isfinite(float(err)) else float("inf")
    return (-frac, err)


def _rel_err(a: float, b: float) -> float:
    a = float(a)
    b = float(b)
    den = max(abs(b), 1e-12)
    return abs(a - b) / den

def _format_attempt_line_3d(
    *,
    attempt_idx: int,
    matcher: str,
    success: bool,
    had_transform: bool,
    frac_inliers: float,
    mean_error_um: float,
    duration_s: float,
    score: float,
    seed: int | None,
    n_crop: int,
    min_inliers: int | None = None,
    note: str | None = None,
) -> str:
    seed_str = "-" if seed is None else str(int(seed))
    mi_str = "-" if min_inliers is None else str(int(min_inliers))
    note_str = "" if not note else f"  note={note}"
    return (
        f"[{attempt_idx:02d}] matcher={matcher:<7} success={int(bool(success))} "
        f"hadT={int(bool(had_transform))} "
        f"frac={frac_inliers:0.3f} err={mean_error_um:0.3g}µm "
        f"dt={duration_s:0.2f}s score={score:0.2f} "
        f"seed={seed_str} n_crop={int(n_crop)} min_inliers={mi_str}"
        f"{note_str}"
    )


def _score_record_3d(rec: dict, target_full_um_zyx, target_crop_um_zyx, pix_rtol=0.02):
    psf = rec.get("pixel_size_full_um_zyx", None)
    psc = rec.get("pixel_size_crop_um_zyx", None)

    pix_penalty = 0.0
    pix_known = (
        target_full_um_zyx is not None
        and target_crop_um_zyx is not None
        and psf is not None
        and psc is not None
    )
    if pix_known:
        psf = np.asarray(psf, float).reshape(3,)
        psc = np.asarray(psc, float).reshape(3,)
        tf = np.asarray(target_full_um_zyx, float).reshape(3,)
        tc = np.asarray(target_crop_um_zyx, float).reshape(3,)
        ef = [_rel_err(a, b) for a, b in zip(psf, tf)]
        ec = [_rel_err(a, b) for a, b in zip(psc, tc)]
        pix_penalty = float(sum(ef) + sum(ec))
        if any(v > float(pix_rtol) for v in ef + ec):
            pix_penalty += 10.0

    q = _quality_tuple_3d(rec)
    return (pix_penalty, q[0], q[1])


def pick_best_transform_3d(records, target_full_um_zyx=None, target_crop_um_zyx=None, pix_rtol=0.02):
    """Select best successful 3D transform record from JSON/JSONL loaded records."""
    valid = []
    rejected = []
    for r in records:
        try:
            ok = bool(r.get("success", False)) and (r.get("scale") is not None or r.get("best_scale") is not None)
            has_R = r.get("R_zyx") is not None or r.get("best_R") is not None
            has_t = r.get("t_um_zyx") is not None or r.get("best_t") is not None
            ok = ok and has_R and has_t
        except Exception:
            ok = False

        if ok:
            valid.append(r)
        else:
            rejected.append(r)

    if not valid:
        msg = ["No valid/successful 3D transforms found."]
        if rejected:
            msg.append(f"Rejected {len(rejected)} record(s).")
        raise ValueError(" ".join(msg))

    scored = [(_score_record_3d(r, target_full_um_zyx, target_crop_um_zyx, pix_rtol=pix_rtol), r) for r in valid]
    scored.sort(key=lambda x: x[0])
    best = scored[0][1]

    shortlist = []
    if target_full_um_zyx is not None and target_crop_um_zyx is not None:
        shortlist = [r for s, r in scored if s[0] < 10.0]
    else:
        shortlist = [r for _, r in scored]

    return best, shortlist


def _save_best_summary_json_3d(best_out: dict, path: Path) -> None:
    """Save lightweight 3D summary JSON without large arrays (e.g., NN distance vectors)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    q = best_out.get("match_quality") or {}
    summary = {
        "success": bool(best_out.get("success", False)),
        "matcher": best_out.get("matcher", None),
        "best_scale": best_out.get("best_scale", None),
        "best_t_um_zyx": (
            np.asarray(best_out["best_t"], float).reshape(3,).tolist()
            if best_out.get("best_t", None) is not None else None
        ),
        "best_R_zyx": (
            np.asarray(best_out["best_R"], float).reshape(3, 3).tolist()
            if best_out.get("best_R", None) is not None else None
        ),
        "match_quality": {
            "frac_inliers": q.get("frac_inliers", None),
            "mean_error_um": q.get("mean_error_um", None),
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def score_attempt_3d(out: Dict[str, Any], duration_s: Optional[float] = None) -> float:
    """Scoring policy to pick the best fallback if all matchers fail."""
    has_t = out.get("best_t", None) is not None
    frac, err = _extract_quality_3d(out)

    if not has_t:
        return -1e12 + 100.0 * frac

    s = 1000.0 * frac - 1.0 * err
    if duration_s is not None and math.isfinite(duration_s):
        s -= 0.05 * float(duration_s)
    return float(s)


def _choose_matcher_order_from_n_3d(n_crop: int) -> Tuple[List[str], str]:
    """3D Policy: Pyramid handles small clouds well, Hashing excels at large dense clouds."""
    n_crop = int(n_crop)
    if n_crop >= 1000:
        return ["hashing", "pyramid"], "n_crop >= 1000"
    return ["pyramid", "hashing"], "n_crop < 1000"


def _ensure_3d_array(arr, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 3:
        raise ValueError(f"{name} must be a 3D array. Got shape={out.shape}")
    return out


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


def _extract_features_and_centroids(label_img_3d, voxel_size_um: np.ndarray):
    df = extract_nuclear_features_3d(
        label_img_3d, pixel_size_um=tuple(float(v) for v in voxel_size_um)
    )

    if df.empty:
        return df, np.empty((0, 3), dtype=np.float32)

    centroids_px = df[["centroid_z_px", "centroid_y_px", "centroid_x_px"]].to_numpy(
        dtype=np.float32, copy=False
    )
    centroids_um = centroids_px * voxel_size_um[None, :]
    return df, centroids_um


def centroids_from_df_3d(df, voxel_size_um_zyx=None, name: str = "") -> np.ndarray:
    """Extract centroids in microns from a precomputed 3D dataframe.

    Accepts either ``centroid_{z,y,x}_um`` columns directly or pixel-space
    ``centroid_{z,y,x}_px`` columns, which are converted to microns using
    ``voxel_size_um_zyx``.
    """
    if df is None:
        raise ValueError(f"{name or 'df'} must be a pandas DataFrame, got None.")
    if len(df) == 0:
        return np.empty((0, 3), dtype=np.float32)

    um_cols = ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
    px_cols = ["centroid_z_px", "centroid_y_px", "centroid_x_px"]

    if all(col in df.columns for col in um_cols):
        centroids_um = df[um_cols].to_numpy(dtype=np.float32, copy=False)
    elif all(col in df.columns for col in px_cols):
        if voxel_size_um_zyx is None:
            raise ValueError(
                f"{name or 'df'} has centroid_*_px columns; voxel_size_um_zyx is required to convert px to µm."
            )

        voxel = _normalize_voxel_size(voxel_size_um_zyx, "voxel_size_um_zyx")
        centroids_px = df[px_cols].to_numpy(dtype=np.float32, copy=False)
        centroids_um = centroids_px * voxel[None, :]
    else:
        raise ValueError(
            f"{name or 'df'} must contain either centroid columns {um_cols} or {px_cols}."
        )

    if centroids_um.ndim != 2 or centroids_um.shape[1] != 3:
        raise ValueError(
            f"{name or 'df'} centroids must have shape (N, 3). Got {centroids_um.shape}."
        )
    if not np.isfinite(centroids_um).all():
        raise ValueError(f"{name or 'df'} centroids contain non-finite values.")

    return centroids_um.astype(np.float32, copy=False)


def evaluate_match_quality_3d(
    centroids_crop_um: np.ndarray,
    centroids_full_um: np.ndarray,
    best_scale: float | None,
    best_R: np.ndarray | None,
    best_t: np.ndarray | None,
    *,
    inlier_radius_um: float = 2.0,
    frac_inliers_thresh: float = 0.45,
    return_dists: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate alignment quality using NN distances in full cloud.

    IMPORTANT:
    - return_dists defaults to False to avoid huge arrays + JSON serialization issues.
    - Set return_dists=True only for debugging.
    """
    if best_scale is None or best_R is None or best_t is None:
        return dict(
            success=False,
            frac_inliers=0.0,
            mean_error_um=None,
            dists=None,
        )

    crop_um = np.asarray(centroids_crop_um, float)
    full_um = np.asarray(centroids_full_um, float)

    crop_aligned_um = float(best_scale) * (crop_um @ np.asarray(best_R, float).T) + np.asarray(best_t, float)
    tree_full = cKDTree(full_um)
    dists, _ = tree_full.query(crop_aligned_um)

    inlier_mask = dists <= float(inlier_radius_um)
    n_total = int(len(dists))
    n_inliers = int(inlier_mask.sum())
    frac_inliers = n_inliers / max(1, n_total)
    mean_error_um = float(dists[inlier_mask].mean()) if n_inliers > 0 else None
    success = frac_inliers >= float(frac_inliers_thresh)

    return dict(
        success=bool(success),
        frac_inliers=float(frac_inliers),
        mean_error_um=mean_error_um,
        dists=(dists if return_dists else None),
    )


def NucleiSky3D(
    centroids_crop_um,
    centroids_full_um,
    full_shape_px_zyx,
    crop_shape_px_zyx,
    pixel_size_full_um_zyx,
    pixel_size_crop_um_zyx,
    matcher: str = "pyramid",
    *,
    matcher_config: Dict[str, Any] | None = None,
    matcher_kwargs: Dict[str, Any] | None = None,
    df_full=None,
    df_crop=None,
    return_dists: bool = False,
) -> Dict[str, Any]:
    """High-level 3D matching orchestrator.

    Strictly handles point-cloud matching. Segmentation and feature extraction
    must be performed prior to calling this function.
    """
    voxel_full_um = _normalize_voxel_size(pixel_size_full_um_zyx, name="pixel_size_full_um_zyx")
    voxel_crop_um = _normalize_voxel_size(pixel_size_crop_um_zyx, name="pixel_size_crop_um_zyx")

    matcher_config = matcher_config or {}
    matcher_kwargs = matcher_kwargs or {}

    matcher_mode = _norm_matcher_name(matcher)
    if matcher_mode not in SUPPORTED_MATCHERS:
        raise ValueError(f"matcher must be one of {SUPPORTED_MATCHERS}, got {matcher!r} -> {matcher_mode!r}")

    cfg_struct = deep_merge_dict(DEFAULT_MATCHER_CONFIG, matcher_config)
    cfg_struct = deep_merge_dict(cfg_struct, normalize_runtime_overrides(matcher_kwargs, matcher_mode))
    cfg = dict(effective_flat_config(cfg_struct, matcher_mode))

    def _resolve_min_inliers_cfg(cfg_flat: dict, n_crop: int) -> tuple[int, int, float, int, float]:
        """Resolve robust min-inlier thresholds with 2D-style dynamic scaling."""
        min_inliers_val = cfg_flat.get("min_inliers", None)

        if isinstance(min_inliers_val, dict):
            min_inliers_abs = min_inliers_val.get("min_inliers_abs", min_inliers_val.get("abs", 5))
            min_inliers_frac = min_inliers_val.get("min_inliers_frac", min_inliers_val.get("frac", 0.12))
            hard_floor = min_inliers_val.get(
                "min_inliers_hard_floor",
                min_inliers_val.get("hard_floor", cfg_flat.get("min_inliers_hard_floor", 3)),
            )
            cap_frac = min_inliers_val.get(
                "min_inliers_cap_frac",
                min_inliers_val.get("cap_frac", cfg_flat.get("min_inliers_cap_frac", 0.80)),
            )
        else:
            min_inliers_abs = cfg_flat.get("min_inliers_abs", min_inliers_val if min_inliers_val is not None else 5)
            min_inliers_frac = cfg_flat.get("min_inliers_frac", 0.12)
            hard_floor = cfg_flat.get("min_inliers_hard_floor", 3)
            cap_frac = cfg_flat.get("min_inliers_cap_frac", 0.80)

        min_inliers_abs = int(min_inliers_abs)
        min_inliers_frac = float(min_inliers_frac)
        hard_floor = int(hard_floor)
        cap_frac = float(cap_frac)

        min_inliers_eff = compute_min_inliers_stable(
            int(n_crop),
            min_inliers_abs=min_inliers_abs,
            min_inliers_frac=min_inliers_frac,
            hard_floor=hard_floor,
            cap_frac=cap_frac,
        )
        return int(min_inliers_eff), min_inliers_abs, min_inliers_frac, hard_floor, cap_frac

    # Safe config extraction
    inlier_radius_um = float(cfg.get("inlier_radius_um", 2.0))
    frac_inliers_thresh = float(cfg.get("frac_inliers_thresh", 0.45))
    dedup_radius_um = float(cfg.get("sanitize_dedup_radius_um", 0.0))
    drop_nonfinite = bool(cfg.get("sanitize_drop_nonfinite", True))
    nn_outlier_percentile = cfg.get("sanitize_nn_outlier_percentile", None)
    min_points = int(cfg.get("sanitize_min_points", 4))

    def _filtered_kwargs(fn, cfg_flat: dict, explicit: set[str]):
        sig = inspect.signature(fn)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        return {k: v for k, v in cfg_flat.items() if k not in explicit and (has_var_kw or k in sig.parameters)}

    # Preflight: sanitize points
    preflight: Dict[str, Any] = {}
    try:
        centroids_full_um, stats_full = sanitize_points_zyx_um(
            centroids_full_um,
            dedup_radius_um=dedup_radius_um,
            drop_nonfinite=drop_nonfinite,
            nn_outlier_percentile=nn_outlier_percentile,
            min_points=min_points,
            name="centroids_full_um",
        )
        centroids_crop_um, stats_crop = sanitize_points_zyx_um(
            centroids_crop_um,
            dedup_radius_um=dedup_radius_um,
            drop_nonfinite=drop_nonfinite,
            nn_outlier_percentile=nn_outlier_percentile,
            min_points=min_points,
            name="centroids_crop_um",
        )
        preflight = {
            "sanitize": {
                "full": stats_full,
                "crop": stats_crop,
                "dedup_radius_um": float(dedup_radius_um),
                "drop_nonfinite": bool(drop_nonfinite),
                "nn_outlier_percentile": nn_outlier_percentile,
                "min_points": int(min_points),
            }
        }
    except ValueError as exc:
        return dict(
            success=False,
            matcher=matcher_mode,
            best_scale=None,
            best_R=None,
            best_t=None,
            best_bbox=None,
            match_quality=dict(success=False, frac_inliers=0.0, mean_error_um=None, dists=None),
            error=f"Point-cloud sanitization failed: {exc}",
            preflight=preflight,
        )

    if len(centroids_full_um) < min_points or len(centroids_crop_um) < min_points:
        return dict(
            success=False,
            matcher=matcher_mode,
            best_scale=None,
            best_R=None,
            best_t=None,
            best_bbox=None,
            match_quality=dict(success=False, frac_inliers=0.0, mean_error_um=None, dists=None),
            error="Insufficient nuclei for 3D matching.",
            preflight=preflight,
        )

    n_crop = int(len(centroids_crop_um))
    min_inliers_eff, min_inliers_abs, min_inliers_frac, min_inliers_hard_floor, min_inliers_cap_frac = _resolve_min_inliers_cfg(
        cfg, n_crop
    )
    preflight["min_inliers"] = {
        "n_crop": int(n_crop),
        "min_inliers": int(min_inliers_eff),
        "min_inliers_abs": int(min_inliers_abs),
        "min_inliers_frac": float(min_inliers_frac),
        "min_inliers_hard_floor": int(min_inliers_hard_floor),
        "min_inliers_cap_frac": float(min_inliers_cap_frac),
    }

    # Execute matcher
    if matcher_mode == "pyramid":
        pyramid_kwargs = _filtered_kwargs(
            run_pyramid_based_matching_um,
            cfg,
            {
                "centroids_crop_um",
                "centroids_full_um",
                "df_full",
                "df_crop",
                "voxel_size_full_um_zyx",
                "voxel_size_crop_um_zyx",
                "full_shape_px_zyx",
                "crop_shape_px_zyx",
                "min_inliers",
                "min_inliers_abs",
                "min_inliers_frac",
                "min_inliers_hard_floor",
                "min_inliers_cap_frac",
            },
        )

        best_scale, best_R, best_t, best_bbox = run_pyramid_based_matching_um(
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            df_full=df_full,
            df_crop=df_crop,
            voxel_size_full_um_zyx=voxel_full_um,
            voxel_size_crop_um_zyx=voxel_crop_um,
            full_shape_px_zyx=full_shape_px_zyx,
            crop_shape_px_zyx=crop_shape_px_zyx,
            min_inliers=int(min_inliers_eff),
            **pyramid_kwargs,
        )

    else:
        # Build a robust call that supports either `min_inliers` OR the expanded params
        sig = inspect.signature(run_geometric_hashing_matching_3d_um)
        params = set(sig.parameters.keys())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        hashing_kwargs = _filtered_kwargs(
            run_geometric_hashing_matching_3d_um,
            cfg,
            {
                "centroids_crop_um",
                "centroids_full_um",
                "full_shape_px",
                "patch_shape_px",
                "pixel_size_full_um_zyx",
                "pixel_size_patch_um_zyx",
                "df_full",
                "df_crop",
                "min_inliers",
                "min_inliers_abs",
                "min_inliers_frac",
                "min_inliers_hard_floor",
                "min_inliers_cap_frac",
            },
        )

        call_kwargs = dict(
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            full_shape_px=full_shape_px_zyx,
            patch_shape_px=crop_shape_px_zyx,
            pixel_size_full_um_zyx=voxel_full_um,
            pixel_size_patch_um_zyx=voxel_crop_um,
            df_full=df_full,
            df_crop=df_crop,
            **hashing_kwargs,
        )

        # Preferred unified API: min_inliers
        if "min_inliers" in params:
            call_kwargs["min_inliers"] = int(min_inliers_eff)
        else:
            # Back-compat API: expanded threshold specification
            if ("min_inliers_abs" in params) or has_var_kw:
                # Keep behavior consistent with unified `min_inliers` API:
                # back-compat absolute threshold receives the resolved effective value.
                call_kwargs["min_inliers_abs"] = int(min_inliers_eff)
            if ("min_inliers_frac" in params) or has_var_kw:
                call_kwargs["min_inliers_frac"] = float(min_inliers_frac)
            if ("min_inliers_hard_floor" in params) or has_var_kw:
                call_kwargs["min_inliers_hard_floor"] = int(min_inliers_hard_floor)
            if ("min_inliers_cap_frac" in params) or has_var_kw:
                call_kwargs["min_inliers_cap_frac"] = float(min_inliers_cap_frac)

        best_scale, best_R, best_t, best_bbox = run_geometric_hashing_matching_3d_um(**call_kwargs)

    match_quality = evaluate_match_quality_3d(
        centroids_crop_um,
        centroids_full_um,
        best_scale,
        best_R,
        best_t,
        inlier_radius_um=inlier_radius_um,
        frac_inliers_thresh=frac_inliers_thresh,
        return_dists=bool(return_dists),
    )

    return dict(
        success=bool(match_quality["success"]),
        matcher=matcher_mode,
        best_scale=None if best_scale is None else float(best_scale),
        best_R=None if best_R is None else np.asarray(best_R, float),
        best_t=None if best_t is None else np.asarray(best_t, float),
        best_bbox=None if best_bbox is None else tuple(int(v) for v in best_bbox),
        match_quality=match_quality,
        preflight=preflight,
    )


def run_adaptive_nucleisky_3d(
    *,
    matcher_order: Optional[List[str]] = None,
    base_seed: int = 0,
    matcher_config: Optional[Dict[str, Any]] = None,
    stop_on_success: bool = True,
    store_full_out: bool = False,
    max_total_time_s: Optional[float] = None,
    verbose: bool = True,
    print_fn=print,
    return_dists: bool = False,
    **nucleisky_inputs: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Adaptive orchestration for 3D, with uniform logging for all matchers."""
    t_start = time.perf_counter()

    def _time_exceeded() -> bool:
        if max_total_time_s is None:
            return False
        return (time.perf_counter() - t_start) >= float(max_total_time_s)

    # 1) Determine n_crop safely
    df_crop = nucleisky_inputs.get("df_crop", None)
    n_crop = _safe_n_points_3d(df_crop)
    if n_crop == 0:
        centroids_crop = nucleisky_inputs.get("centroids_crop_um", None)
        n_crop = _safe_n_points_3d(centroids_crop)

    # 2) Determine matcher order
    if matcher_order is None:
        matcher_order_used, order_label = _choose_matcher_order_from_n_3d(n_crop)
        order_source = "auto_by_nuclei_count"
    else:
        matcher_order_used = [_norm_matcher_name(m) for m in list(matcher_order)]
        order_label = "explicit"
        order_source = "caller_provided"

    history: List[Dict[str, Any]] = []
    best_out: Optional[Dict[str, Any]] = None
    best_score: float = -math.inf

    for attempt_idx, matcher in enumerate(matcher_order_used):
        if _time_exceeded():
            rec = {
                "attempt": int(attempt_idx),
                "matcher": str(matcher),
                "seed": None,
                "success": False,
                "had_transform": False,
                "frac_inliers": 0.0,
                "mean_error_um": math.inf,
                "duration_s": 0.0,
                "score": -1e18,
                "n_crop": int(n_crop),
                "matcher_order_used": list(matcher_order_used),
                "order_source": str(order_source),
                "order_label": str(order_label),
                "note": "time_budget_exceeded",
            }
            history.append(rec)
            if verbose:
                print_fn(_format_attempt_line_3d(
                    attempt_idx=rec["attempt"],
                    matcher=str(rec["matcher"]),
                    success=rec["success"],
                    had_transform=rec["had_transform"],
                    frac_inliers=rec["frac_inliers"],
                    mean_error_um=rec["mean_error_um"],
                    duration_s=rec["duration_s"],
                    score=rec["score"],
                    seed=rec["seed"],
                    n_crop=rec["n_crop"],
                    min_inliers=None,
                    note=rec.get("note"),
                ))
            break

        m = _norm_matcher_name(matcher)

        seed = _stable_u32(base_seed, m, attempt_idx, n_crop)
        matcher_kwargs = {"_common": {"random_state": int(seed)}, m: {}}

        t0 = time.perf_counter()
        try:
            out = NucleiSky3D(
                matcher=m,
                matcher_config=matcher_config,
                matcher_kwargs=matcher_kwargs,
                return_dists=bool(return_dists),
                **nucleisky_inputs,
            )
        except Exception:
            traceback.print_exc()
            out = dict(
                success=False,
                matcher=m,
                best_scale=None,
                best_R=None,
                best_t=None,
                match_quality=None,
                note="NucleiSky3D call crashed; see traceback above.",
            )
        dt = float(time.perf_counter() - t0)

        frac, err = _extract_quality_3d(out)
        had_transform = (
            out.get("best_scale") is not None
            and out.get("best_R") is not None
            and out.get("best_t") is not None
        )
        s = score_attempt_3d(out, duration_s=dt)

        # pull resolved min_inliers from preflight when present (uniform across matchers)
        min_inliers_eff = None
        try:
            min_inliers_eff = int((out.get("preflight") or {}).get("min_inliers", {}).get("min_inliers"))
        except Exception:
            min_inliers_eff = None

        rec = {
            "attempt": int(attempt_idx),
            "matcher": str(m),
            "seed": int(seed),
            "success": bool(out.get("success", False)),
            "had_transform": bool(had_transform),
            "frac_inliers": float(frac),
            "mean_error_um": float(err),
            "duration_s": float(dt),
            "score": float(s),
            "n_crop": int(n_crop),
            "min_inliers": (int(min_inliers_eff) if min_inliers_eff is not None else None),
            "matcher_order_used": list(matcher_order_used),
            "order_source": str(order_source),
            "order_label": str(order_label),
        }
        if store_full_out:
            rec["out"] = out
        history.append(rec)

        if verbose:
            print_fn(_format_attempt_line_3d(
                attempt_idx=rec["attempt"],
                matcher=rec["matcher"],
                success=rec["success"],
                had_transform=rec["had_transform"],
                frac_inliers=rec["frac_inliers"],
                mean_error_um=rec["mean_error_um"],
                duration_s=rec["duration_s"],
                score=rec["score"],
                seed=rec["seed"],
                n_crop=rec["n_crop"],
                min_inliers=rec.get("min_inliers"),
                note=(out.get("note") or out.get("error")),
            ))

        if s > best_score:
            best_score = float(s)
            best_out = out

        if stop_on_success and bool(out.get("success", False)):
            return out, history

    if best_out is None:
        best_out = {"success": False, "matcher": None, "best_scale": None, "best_R": None, "best_t": None, "match_quality": None}
    return best_out, history


def run_adaptive_matching_and_export_3d(
    *,
    df_full,
    df_crop,
    img_full_orig=None,  # High-res raw data (optional, for final export)
    img_crop_orig=None,
    pixel_size_full_orig_um_zyx,
    pixel_size_crop_orig_um_zyx,
    result_dir: str,
    cfg_selected: Optional[dict] = None,
    base_seed: int = 0,
    store_full_out: bool = False,
    max_total_time_s: Optional[float] = None,
    img_full_seg=None,  # Downsampled data used for actual matching (optional)
    img_crop_seg=None,
    pixel_size_full_seg_um_zyx=None,
    pixel_size_crop_seg_um_zyx=None,
    labels_full=None,
    labels_crop=None,
    save_segmentation_masks: bool = True,
    verbose: bool = True,
    print_fn=print,
) -> Tuple[Dict[str, Any], list]:
    """Fully robust 3D Adaptive Matching and Export (library-first I/O)."""
    from .io import save_tiff_zyx, save_nucleisky_transform_3d, append_transform_jsonl
    from .config import save_matcher_config_json

    try:
        if len(df_full) == 0 or len(df_crop) == 0:
            raise ValueError("df_full and df_crop must not be empty for 3D matching.")

        # 1) Dual-scale logic (orig vs seg)
        vox_full_orig = _normalize_voxel_size(pixel_size_full_orig_um_zyx, "pixel_size_full_orig_um_zyx")
        vox_crop_orig = _normalize_voxel_size(pixel_size_crop_orig_um_zyx, "pixel_size_crop_orig_um_zyx")

        vox_full_seg = vox_full_orig if pixel_size_full_seg_um_zyx is None else _normalize_voxel_size(pixel_size_full_seg_um_zyx, "pixel_size_full_seg_um_zyx")
        vox_crop_seg = vox_crop_orig if pixel_size_crop_seg_um_zyx is None else _normalize_voxel_size(pixel_size_crop_seg_um_zyx, "pixel_size_crop_seg_um_zyx")

        match_full_shape = (
            img_full_orig.shape if img_full_orig is not None
            else (labels_full.shape if labels_full is not None else (img_full_seg.shape if img_full_seg is not None else None))
        )
        match_crop_shape = (
            img_crop_orig.shape if img_crop_orig is not None
            else (labels_crop.shape if labels_crop is not None else (img_crop_seg.shape if img_crop_seg is not None else None))
        )

        # choose grid for matching
        match_vox_full = vox_full_orig if img_full_orig is not None else vox_full_seg
        match_vox_crop = vox_crop_orig if img_crop_orig is not None else vox_crop_seg
        bbox_grid = "orig" if (img_full_orig is not None and img_crop_orig is not None) else "seg"

        # 2) Pre-extract centroids (pure math afterwards)
        centroids_full_um = centroids_from_df_3d(df_full, voxel_size_um_zyx=vox_full_seg, name="df_full")
        centroids_crop_um = centroids_from_df_3d(df_crop, voxel_size_um_zyx=vox_crop_seg, name="df_crop")

        # 3) Output directory
        out_dir = Path(result_dir) / "matching" / "adaptive_3d" / "exports_adaptive"
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_segmentation_masks and (labels_full is not None or labels_crop is not None):
            seg_dir = out_dir / "segmentation_masks"
            seg_dir.mkdir(parents=True, exist_ok=True)
            if labels_full is not None:
                save_tiff_zyx(seg_dir / "labels_full.tif", labels_full, voxel_size_um_zyx=vox_full_seg)
            if labels_crop is not None:
                save_tiff_zyx(seg_dir / "labels_crop.tif", labels_crop, voxel_size_um_zyx=vox_crop_seg)

        # 4) Run adaptive matching with uniform logging
        best_out, history = run_adaptive_nucleisky_3d(
            matcher_order=None,
            base_seed=int(base_seed),
            matcher_config=cfg_selected,
            store_full_out=bool(store_full_out),
            stop_on_success=True,
            max_total_time_s=max_total_time_s,
            verbose=bool(verbose),
            print_fn=print_fn,
            return_dists=False,  # keep outputs JSON-safe + small by default
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            df_full=df_full,
            df_crop=df_crop,
            full_shape_px_zyx=match_full_shape,
            crop_shape_px_zyx=match_crop_shape,
            pixel_size_full_um_zyx=match_vox_full,
            pixel_size_crop_um_zyx=match_vox_crop,
        )

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

        cfg_effective = deep_merge_dict(DEFAULT_MATCHER_CONFIG, cfg_selected)
        save_matcher_config_json(cfg_effective, out_dir / "matcher_config_used.json")

        _save_best_summary_json_3d(best_out, out_dir / "best_summary.json")

        # 5) Save history.jsonl (already JSON-safe primitives)
        with (out_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            for rec in history:
                safe = dict(rec)
                safe.pop("out", None)
                safe["run_id"] = run_id
                json.dump(safe, f, ensure_ascii=False)
                f.write("\n")

        # 6) Export transform + bbox + optional aligned crop TIFF
        if best_out.get("success") and best_out.get("best_scale") is not None:
            if (
                best_out.get("best_R") is not None
                and best_out.get("best_t") is not None
                and match_full_shape is not None
                and match_crop_shape is not None
            ):
                best_out = dict(best_out)
                best_out["best_bbox"] = tuple(
                    int(v) for v in bbox_full_px_from_similarity_um_3d(
                        crop_shape_px=match_crop_shape,
                        pixel_size_full_um_zyx=match_vox_full,
                        pixel_size_crop_um_zyx=match_vox_crop,
                        scale=best_out["best_scale"],
                        R_zyx=best_out["best_R"],
                        t_um_zyx=best_out["best_t"],
                        full_shape_px=match_full_shape,
                    )
                )

            # Library record (sanitized in io)
            transform_record = save_nucleisky_transform_3d(
                best_out,
                out_path=None,
                pixel_size_full_um_zyx=vox_full_orig,
                pixel_size_crop_um_zyx=vox_crop_orig,
                matcher_name=str(best_out.get("matcher", "adaptive_best")),
                require_success=True,
            )
            transform_record["bbox_grid"] = bbox_grid
            transform_record["run_id"] = run_id

            # Use library JSONL writer (no manual json.dump / numpy issues)
            append_transform_jsonl(transform_record, out_dir / "transforms.jsonl")

            # Export aligned TIFF if we have images
            img_to_export = img_full_orig if img_full_orig is not None else img_full_seg
            crop_to_export = img_crop_orig if img_crop_orig is not None else img_crop_seg
            if img_to_export is not None and crop_to_export is not None:
                try:
                    export_aligned_crop_tiff(
                        img_full=img_to_export,
                        img_crop=crop_to_export,
                        output_path=out_dir / f"aligned_crop_{best_out.get('matcher','adaptive')}.tif",
                        pixel_size_full_um=vox_full_orig,
                        pixel_size_crop_um=vox_crop_orig,
                        best_scale=best_out["best_scale"],
                        best_R=best_out["best_R"],
                        best_t=best_out["best_t"],
                    )
                    if verbose:
                        print_fn(f"✅ 3D Adaptive export -> {out_dir}")
                except Exception as e:
                    if verbose:
                        print_fn(f"⚠️ Matching succeeded, but TIFF export failed: {e}")

        return best_out, history

    except Exception:
        traceback.print_exc()
        raise
