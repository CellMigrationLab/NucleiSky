
"""pipeline.py High-level API: orchestration of segmentation, matching, scoring, and export."""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import normalize_runtime_overrides
from .export import export_aligned_imagej_stacks, warp_and_save_metrics 
from .visualization import plot_warp_overlay
from .features import extract_nuclear_features
from .io import (
    append_transform_jsonl,
    get_pixel_size_um_from_tiff,
    load_nucleisky_transform,
    make_result_dir,
    save_nucleisky_transform,
    save_tiff,
)
from .matching.graph import run_graph_based_matching_um
from .matching.hashing import run_geometric_hashing_matching_um
from .matching.quad import run_quad_based_matching_um
from .matching.triangle import run_triangle_based_matching_um
from .preprocess import (
    choose_common_target_um_per_px,
    rescale_to_target_um_per_px,
    scale_normalize_pair_for_segmentation,
    ij_percentile_normalize,
)
from .segmentation import segment_nuclei_dispatch
import inspect
import math

from scipy.spatial import cKDTree

from .config import DEFAULT_MATCHER_CONFIG, deep_merge_dict, effective_flat_config, normalize_runtime_overrides
from .io import validate_transform_record
from .preprocess import _as_array
from .utils import _rel_err, _is_finite_number, _stable_u32  
from .config import _norm_matcher_name, save_matcher_config_json  

from .export import _export_best_everything
from .features import extract_centroids_um
from .features import stack_feature_vectors
from .matching.geometry import invert_affine_px
from .matching.geometry import rotation_deg_from_R
from .preprocess import require_2d_image, require_2d_label_mask
from .preprocess import require_positive_float
from .matching.geometry import _ensure_points_yx, compute_min_inliers_stable


# ============================================================
# Hard-coded controller + export policy for adaptive mode
# ============================================================

# Count-based matcher policy (AUTO if matcher_order=None)
ADAPTIVE_N_SMALL = 20
ADAPTIVE_N_LARGE = 1000

ADAPTIVE_ORDER_LT20   = ["quad", "triangles", "graph", "hashing"]
ADAPTIVE_ORDER_GE20   = ["triangles", "graph", "quad", "hashing"]
ADAPTIVE_ORDER_GE1000 = ["triangles", "quad", "graph", "hashing"]

ADAPTIVE_BASE_SEED = 0
ADAPTIVE_MAX_TOTAL_TIME_S = None



def NucleiSky(
    centroids_crop_um,
    centroids_full_um,
    img_full,
    img_crop,
    ij_percentile_normalize,
    pixel_size_full_um,
    pixel_size_crop_um,
    matcher="graph",          # "graph", "quad", "triangles", "hashing"
    features_crop=None,       # required for graph matcher
    features_full=None,       # required for graph matcher
    df_full=None,
    df_crop=None,
    labels_full=None,
    labels_crop=None,

    # config plumbing
    matcher_config=None,      # full structured config (e.g. from UI)
    matcher_kwargs=None,      # runtime overrides (flat matcher-only OR hierarchical)

    # debugging / outputs    
    save_dir=None,
    save_prefix="match",
):
    """
    Config-driven NucleiSky wrapper.
    """

    # -------------------------
    # Small internal validators
    # -------------------------

    if matcher_kwargs is None:
        matcher_kwargs = {}
    if matcher_config is None:
        matcher_config = {}

    def _ensure_pos_float(x, name="x"):
        try:
            v = float(x)
        except Exception:
            raise ValueError(f"{name} must be a float. Got {type(x)}")
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a positive finite float. Got {v}")
        return v

    def _as_feature_matrix(F, expected_rows, name="features"):
        if F is None:
            return None
        # ... [Keep Feature Matrix Logic] ...
        if isinstance(F, (list, tuple)) and len(F) == expected_rows:
            try:
                M = np.stack([np.asarray(v, float).ravel() for v in F], axis=0)
                if M.ndim != 2 or M.shape[0] != expected_rows:
                    raise ValueError
                return M.astype(np.float32, copy=False)
            except Exception:
                pass
        try:
            import pandas as pd
            if isinstance(F, pd.Series) and len(F) == expected_rows:
                M = np.stack([np.asarray(v, float).ravel() for v in F.tolist()], axis=0)
                return M.astype(np.float32, copy=False)
        except Exception:
            pass
        M = np.asarray(F, float)
        if M.ndim == 1: M = M[:, None]
        return M.astype(np.float32, copy=False)

    def _validate_flat_cfg_keys(cfg_flat: dict, matcher_mode: str):
        allowed = set(DEFAULT_MATCHER_CONFIG["_common"].keys()) | set(DEFAULT_MATCHER_CONFIG.get(matcher_mode, {}).keys())
        extra = set(cfg_flat.keys()) - allowed
        if extra:
            raise KeyError(f"Unknown config keys for matcher '{matcher_mode}': {sorted(extra)}")

    # -------------------------
    # Normalize / select matcher
    # -------------------------
    matcher_mode = str(matcher).strip().lower()
    if matcher_mode in ("triangle", "tri", "tris"):
        matcher_mode = "triangles"

    # -------------------------
    # Build effective config 
    # -------------------------
    cfg_struct = deep_merge_dict(DEFAULT_MATCHER_CONFIG, matcher_config)
    cfg_struct = deep_merge_dict(cfg_struct, normalize_runtime_overrides(matcher_kwargs, matcher_mode))
    cfg = effective_flat_config(cfg_struct, matcher_mode)

    _validate_flat_cfg_keys(cfg, matcher_mode)

    rs = cfg.get("random_state", 0)
    random_state = None if rs is None else int(rs)

    def _filtered_kwargs(fn, cfg: dict, explicit: dict, extra: dict | None = None):
        sig = inspect.signature(fn)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        ALWAYS_SKIP = {"min_inliers_abs", "min_inliers_frac"}
        out = {}
        for k, v in cfg.items():
            if k in explicit or k in ALWAYS_SKIP: continue
            if has_var_kw or (k in sig.parameters): out[k] = v
        if extra:
            for k, v in extra.items():
                if k in explicit or k in ALWAYS_SKIP: continue
                if has_var_kw or (k in sig.parameters): out[k] = v
        return out

    note_parts = []

    centroids_crop_um = _ensure_points_yx(centroids_crop_um, name="centroids_crop_um")
    centroids_full_um = _ensure_points_yx(centroids_full_um, name="centroids_full_um")
    Nc = int(len(centroids_crop_um))

    pix_full = _ensure_pos_float(pixel_size_full_um, name="pixel_size_full_um")
    pix_crop = _ensure_pos_float(pixel_size_crop_um, name="pixel_size_crop_um")

    # common params
    inlier_radius_um = _ensure_pos_float(cfg["inlier_radius_um"], name="inlier_radius_um")
    margin_um = float(cfg.get("margin_um", 0.0))
    scale_min = float(cfg["scale_min"])
    scale_max = float(cfg["scale_max"])
    angle_max_deg = cfg.get("angle_max_deg", None)
    if angle_max_deg is not None: angle_max_deg = float(angle_max_deg)

    use_dynamic_scale = bool(cfg.get("use_dynamic_scale", False))
    dynamic_rel_tol = float(cfg.get("dynamic_rel_tol", 0.1))
    use_icp_refinement = bool(cfg.get("use_icp_refinement", True))
    frac_inliers_thresh = float(cfg.get("frac_inliers_thresh", 0.6))

    if use_dynamic_scale and (df_full is None or df_crop is None):
        use_dynamic_scale = False
        note_parts.append("use_dynamic_scale was True but df_full/df_crop not provided; dynamic scale disabled.")

    min_inliers_abs = int(cfg.get("min_inliers_abs", 10))
    min_inliers_frac = float(cfg.get("min_inliers_frac", 0.1))
    min_inliers = compute_min_inliers_stable(
        Nc, min_inliers_abs, min_inliers_frac, hard_floor=3, cap_frac=0.95
    )

    # -------------------------
    # Enforce 2D image inputs
    # -------------------------
    img_full = require_2d_image(img_full, label="img_full")
    img_crop = require_2d_image(img_crop, label="img_crop")

    full_shape_px = tuple(img_full.shape)
    patch_shape_px = tuple(img_crop.shape)

    # -------------------------
    # Run selected matcher (Passing Shapes)
    # -------------------------
    best_scale, best_R, best_t, bbox = (None, None, None, None)

    if matcher_mode == "graph":
        print(f"\n\n========== Matcher: Graph ==========")
        if features_crop is None or features_full is None:
            raise ValueError("Graph matcher requires features_crop and features_full.")

        features_full_m = _as_feature_matrix(features_full, expected_rows=len(centroids_full_um), name="features_full")
        features_crop_m = _as_feature_matrix(features_crop, expected_rows=len(centroids_crop_um), name="features_crop")

        best_scale, best_R, best_t, bbox = run_graph_based_matching_um(
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            full_shape_px=full_shape_px,
            patch_shape_px=patch_shape_px,
            features_crop=features_crop_m,
            features_full=features_full_m,
            pixel_size_full_um=pix_full,
            pixel_size_patch_um=pix_crop,
            
            # ... Pass Config ...
            k_nn_graph=int(cfg.get("k_nn_graph", 8)),
            k_ngh_feat=int(cfg.get("k_ngh_feat", 6)),
            standardize=bool(cfg.get("standardize", True)),
            w_shape=float(cfg.get("w_shape", 0.3)),
            w_graph=float(cfg.get("w_graph", 1.0)),
            w_triangles=float(cfg.get("w_triangles", 0.7)),
            n_triangles=int(cfg.get("n_triangles", 10)),
            n_feat_neighbors=int(cfg.get("n_feat_neighbors", 5)),
            n_iters=int(cfg.get("n_iters", 50_000)),
            min_inliers=int(min_inliers),
            min_triangle_area_um2=float(cfg.get("min_triangle_area_um2", 1e-4)),
            enforce_unique_full_matches=bool(cfg.get("enforce_unique_full_matches", True)),
            feat_ratio=float(cfg.get("feat_ratio", 0.85)),
            feat_max_dist=cfg.get("feat_max_dist", None),
            require_mutual=bool(cfg.get("require_mutual", True)),
            k_spatial=int(cfg.get("k_spatial", 3)),
            require_feat_consistency=bool(cfg.get("require_feat_consistency", True)),
            prosac=bool(cfg.get("prosac", True)),
            pretest_n=int(cfg.get("pretest_n", 200)),
            refit_on_inliers=bool(cfg.get("refit_on_inliers", True)),
            min_inlier_radius_frac_nn=float(cfg.get("min_inlier_radius_frac_nn", 0.12)),
            max_candidate_pairs=int(cfg.get("max_candidate_pairs", 30_000)),
            inlier_radius_um=inlier_radius_um,
            scale_min=scale_min,
            scale_max=scale_max,
            angle_max_deg=angle_max_deg,
            random_state=random_state,
            use_icp_refinement=use_icp_refinement,
            margin_um=margin_um,
            df_full=df_full,
            df_crop=df_crop,
            use_dynamic_scale=bool(use_dynamic_scale),
            dynamic_rel_tol=float(dynamic_rel_tol),
            n_candidates_per_patch=int(cfg.get("n_candidates_per_patch", 3)),
            n_candidates_per_full=int(cfg.get("n_candidates_per_full", 3)),
            pretest_relax=float(cfg.get("pretest_relax", 0.8)),
            soft_fail_return_best=bool(cfg.get("soft_fail_return_best", True)),
            min_inliers_cap_frac=float(cfg.get("min_inliers_cap_frac", 0.80)),
        )

    elif matcher_mode == "quad":
        print(f"\n\n========== Matcher: Quad ==========")
        explicit = dict(
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            full_shape_px=full_shape_px,
            patch_shape_px=patch_shape_px,
            pixel_size_full_um=pix_full,
            pixel_size_patch_um=pix_crop,
            inlier_radius_um=inlier_radius_um,
            scale_min=scale_min,
            scale_max=scale_max,
            random_state=random_state,
            margin_um=margin_um,
            df_full=df_full,
            df_crop=df_crop,
            use_dynamic_scale=bool(use_dynamic_scale),
            dynamic_rel_tol=float(dynamic_rel_tol),            
        )

        quad_kwargs = _filtered_kwargs(
            run_quad_based_matching_um,
            cfg=cfg,
            explicit=explicit,
            extra={"min_inliers": int(min_inliers)},
        )

        best_scale, best_R, best_t, bbox = run_quad_based_matching_um(
            **explicit,
            **quad_kwargs,
        )

    elif matcher_mode == "triangles":
        print(f"\n\n========== Matcher: Triangles ==========")
        explicit = dict(
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            full_shape_px=full_shape_px,
            patch_shape_px=patch_shape_px,
            pixel_size_full_um=pix_full,
            pixel_size_patch_um=pix_crop,
            inlier_radius_um=inlier_radius_um,
            scale_min=scale_min,
            scale_max=scale_max,
            random_state=random_state,
            margin_um=margin_um,
            df_full=df_full,
            df_crop=df_crop,
            use_dynamic_scale=bool(use_dynamic_scale),
            dynamic_rel_tol=float(dynamic_rel_tol),            
        )

        tri_kwargs = _filtered_kwargs(
            run_triangle_based_matching_um,
            cfg=cfg,
            explicit=explicit,
            extra={"min_inliers": int(min_inliers)},
        )

        best_scale, best_R, best_t, bbox = run_triangle_based_matching_um(
            **explicit,
            **tri_kwargs,
        )

    elif matcher_mode == "hashing":
        print(f"\n\n========== Matcher: Hashing ==========")
        explicit = dict(
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            full_shape_px=full_shape_px,
            patch_shape_px=patch_shape_px,
            pixel_size_full_um=pix_full,
            pixel_size_patch_um=pix_crop,
            inlier_radius_um=inlier_radius_um,
            scale_min=scale_min,
            scale_max=scale_max,
            random_state=random_state,
            margin_um=margin_um,
            df_full=df_full,
            df_crop=df_crop,
            use_dynamic_scale=bool(use_dynamic_scale),
            dynamic_rel_tol=float(dynamic_rel_tol),            
        )

        hash_kwargs = _filtered_kwargs(
            run_geometric_hashing_matching_um,
            cfg=cfg,
            explicit=explicit,
            extra={"min_inliers": int(min_inliers)},
        )

        best_scale, best_R, best_t, bbox = run_geometric_hashing_matching_um(
            **explicit,
            **hash_kwargs,
        )

    else:
        raise ValueError(f"Unknown matcher '{matcher}'. Use 'graph', 'quad', 'triangles', or 'hashing'.")

    # -------------------------
    # Evaluate Quality
    # -------------------------
    if best_scale is None or best_R is None or best_t is None:
        return dict(
            success=False,
            matcher=matcher_mode,
            best_scale=None,
            best_R=None,
            best_t=None,
            bbox_full_px=None,
            df_full=df_full,
            df_crop=df_crop,
            centroids_full_um=centroids_full_um,
            centroids_crop_um=centroids_crop_um,
            labels_full=labels_full,
            labels_crop=labels_crop,
            match_quality=None,
            note=(" ".join(note_parts) if note_parts else None),
        )

    quality = evaluate_match_quality(
        centroids_crop_um=centroids_crop_um,
        centroids_full_um=centroids_full_um,
        best_scale=best_scale,
        best_R=best_R,
        best_t=best_t,
        inlier_radius_um=inlier_radius_um,
        frac_inliers_thresh=frac_inliers_thresh,
    )

    print("\nMatch quality:")
    print(f"  success        = {quality['success']}")
    print(f"  frac_inliers   = {quality['frac_inliers']:.3f}")
    print(f"  mean_error_um = {quality['mean_error_um']}")


    return dict(
        success=bool(quality["success"]),
        matcher=matcher_mode,
        best_scale=float(best_scale),
        best_R=np.asarray(best_R, float),
        best_t=np.asarray(best_t, float),
        bbox_full_px=bbox,
        df_full=df_full,
        df_crop=df_crop,
        centroids_full_um=centroids_full_um,
        centroids_crop_um=centroids_crop_um,
        labels_full=labels_full,
        labels_crop=labels_crop,
        match_quality=quality,
        note=(" ".join(note_parts) if note_parts else None),
    )



def run_adaptive_nucleisky(
    *,
    matcher_order: Optional[List[str]] = None,   # if None -> auto based on n_crop
    base_seed: int = ADAPTIVE_BASE_SEED,
    matcher_config: Optional[Dict[str, Any]] = None,
    stop_on_success: bool = True,
    store_full_out: bool = False,
    max_total_time_s: Optional[float] = ADAPTIVE_MAX_TOTAL_TIME_S,
    **nucleisky_inputs: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    New adaptive strategy (as requested):

      - Choose matcher order based on number of nuclei in the patch (n_crop)
          n_crop < 20    -> ["quad", "triangles", "graph", "hashing"]
          n_crop >= 20   -> ["triangles", "graph", "quad", "hashing"]
          n_crop >= 1000 -> ["hashing", "triangles", "graph", "quad"]

      - Always use DEFAULT settings (no ultrafast, no stage overrides).
      - If a matcher fails (success != True), try the next.
      - Stop immediately when success == True.
      - If none succeeds, return best fallback by score_attempt.
    """
    t_start = time.perf_counter()

    def _time_exceeded() -> bool:
        if max_total_time_s is None:
            return False
        return (time.perf_counter() - t_start) >= float(max_total_time_s)

    # Determine n_crop (number of nuclei in patch)
    n_crop = _safe_n_points(nucleisky_inputs.get("centroids_crop_um", None))
    if n_crop <= 0 and ("df_crop" in nucleisky_inputs) and (nucleisky_inputs["df_crop"] is not None):
        try:
            n_crop = int(len(nucleisky_inputs["df_crop"]))
        except Exception:
            pass

    # Determine matcher order
    if matcher_order is None:
        matcher_order_used, order_label = _choose_matcher_order_from_n(n_crop)
        order_source = "auto_by_nuclei_count"
    else:
        matcher_order_used = [_norm_matcher_name(m) for m in list(matcher_order)]
        order_label = "explicit"
        order_source = "caller_provided"

    # Graph availability check (only used to skip graph cleanly if features missing)
    features_crop = nucleisky_inputs.get("features_crop", None)
    features_full = nucleisky_inputs.get("features_full", None)
    graph_available = (features_crop is not None) and (features_full is not None)

    history: List[Dict[str, Any]] = []
    best_out: Optional[Dict[str, Any]] = None
    best_score: float = -math.inf

    for attempt_idx, matcher in enumerate(matcher_order_used):
        if _time_exceeded():
            history.append({
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
            })
            break

        m = _norm_matcher_name(matcher)

        if m == "graph" and not graph_available:
            history.append({
                "attempt": int(attempt_idx),
                "matcher": "graph",
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
                "note": "graph skipped (features_crop/features_full missing)",
            })
            continue

        # Deterministic seed per attempt
        seed = int((_stable_u32(base_seed, m, attempt_idx, n_crop) + int(base_seed)) & 0xFFFFFFFF)

        # DEFAULT settings: only enforce random_state deterministically
        matcher_kwargs = {"_common": {"random_state": int(seed)}, m: {}}

        t0 = time.perf_counter()
        try:
            out = NucleiSky(
                matcher=m,
                matcher_config=matcher_config,
                matcher_kwargs=matcher_kwargs,
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
                note="NucleiSky call crashed; see traceback above.",
            )
        dt = float(time.perf_counter() - t0)

        had_t = out.get("best_t", None) is not None
        frac, err = _extract_quality(out)
        s = score_attempt(out, duration_s=dt)

        rec = {
            "attempt": int(attempt_idx),
            "matcher": str(m),
            "seed": int(seed),
            "success": bool(out.get("success", False)),
            "had_transform": bool(had_t),
            "frac_inliers": float(frac),
            "mean_error_um": float(err) if math.isfinite(err) else math.inf,
            "duration_s": float(dt),
            "score": float(s),
            "n_crop": int(n_crop),
            "matcher_order_used": list(matcher_order_used),
            "order_source": str(order_source),
            "order_label": str(order_label),
        }
        if store_full_out:
            rec["out"] = out
        history.append(rec)

        if s > best_score:
            best_score = float(s)
            best_out = out

        # Stop condition: ONLY when success is true (as requested)
        if stop_on_success and bool(out.get("success", False)):
            return out, history

    if best_out is None:
        best_out = {"success": False, "matcher": None, "best_t": None, "match_quality": None}
    return best_out, history


def run_adaptive_matching_and_export(
    *,
    df_full,
    df_crop,
    img_full,
    img_crop,
    pixel_size_full_um: float,
    pixel_size_crop_um: float,
    result_dir: str,
    cfg_selected: Optional[dict] = None,
    base_seed: int = 0,
    margin_px: int = 20,
    store_full_out: bool = False,
    max_total_time_s: Optional[float] = None,
    features_full=None,
    features_crop=None,
    img_full_seg=None,
    img_crop_seg=None,
    pixel_size_full_seg_um=None,
    pixel_size_crop_seg_um=None,
    labels_full=None,
    labels_crop=None,
    save_segmentation_masks: bool = True,
    ij_percentile_normalize: Optional[Any] = None,
) -> Tuple[Dict[str, Any], list]:
    """
    API-compatible version of adaptive matching.

    Args:
        df_full, df_crop: DataFrames with extracted nuclear features.
        img_full, img_crop: Numpy arrays (2D).
        pixel_size_full_um, pixel_size_crop_um: Pixel sizes in µm/px.
        result_dir: Path to base output directory.
        cfg_selected: Matcher configuration (defaults to DEFAULT_MATCHER_CONFIG).
        base_seed: Random seed for reproducibility.
        margin_px: Margin (px) when exporting overlays.
        store_full_out: Whether to keep all adaptive attempts.
        max_total_time_s: Optional global time budget for adaptive controller.
        features_full/features_crop: Optional feature arrays (for graph matcher).
        img_full_seg/img_crop_seg: Optional segmentation-scale images.
        pixel_size_full_seg_um/pixel_size_crop_seg_um: Optional µm/px at segmentation scale.
        labels_full/labels_crop: Optional segmentation masks (2D label images).
        save_segmentation_masks: Save label masks (if provided) under the export directory.
        ij_percentile_normalize: Optional normalization function or parameters.

    Returns:
        (best_result_dict, history_list)
    """
    try:
        # Validate inputs
        img_full_orig = require_2d_image(img_full, label="img_full")
        img_crop_orig = require_2d_image(img_crop, label="img_crop")

        if len(df_full) == 0 or len(df_crop) == 0:
            raise ValueError("df_full and df_crop must not be empty")

        pix_full_orig = require_positive_float(pixel_size_full_um, label="pixel_size_full_um")
        pix_crop_orig = require_positive_float(pixel_size_crop_um, label="pixel_size_crop_um")

        img_full_seg = img_full if img_full_seg is None else img_full_seg
        img_crop_seg = img_crop if img_crop_seg is None else img_crop_seg

        pix_full_seg = pix_full_orig if pixel_size_full_seg_um is None else pixel_size_full_seg_um
        pix_crop_seg = pix_crop_orig if pixel_size_crop_seg_um is None else pixel_size_crop_seg_um

        # Extract centroids
        centroids_full_um = extract_centroids_um(df_full, name="df_full")
        centroids_crop_um = extract_centroids_um(df_crop, name="df_crop")

        # Features (optional)
        if features_full is None or features_crop is None:
            try:
                features_full = stack_feature_vectors(df_full, name="df_full")
                features_crop = stack_feature_vectors(df_crop, name="df_crop")
            except Exception:
                features_full = None
                features_crop = None
                print("ℹ️ No feature vectors found — graph matcher will be skipped.")

        # Choose matcher order
        order_used, label = _choose_matcher_order_from_n(len(df_crop))
        print(f"Adaptive matching order policy: {label} ({order_used})")

        # Prepare output dirs
        base_save = Path(result_dir) / "matching" / "adaptive"
        base_save.mkdir(parents=True, exist_ok=True)
        out_dir = base_save / "exports_adaptive"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save segmentation masks (if provided)
        if save_segmentation_masks and (labels_full is not None or labels_crop is not None):
            seg_dir = out_dir / "segmentation_masks"
            seg_dir.mkdir(parents=True, exist_ok=True)

            def _save_mask(mask, *, name: str, expected_shape):
                if mask is None:
                    return None
                checked = require_2d_label_mask(mask, label=name, expected_shape=expected_shape)
                out_path = seg_dir / f"{name}.tif"
                save_tiff(out_path, checked)
                return out_path

            _save_mask(labels_full, name="labels_full", expected_shape=img_full_seg.shape[:2])
            _save_mask(labels_crop, name="labels_crop", expected_shape=img_crop_seg.shape[:2])

        # Run adaptive matching
        best_out, history = run_adaptive_nucleisky(
            matcher_order=None,
            base_seed=int(base_seed),
            matcher_config=cfg_selected,
            store_full_out=bool(store_full_out),
            stop_on_success=True,
            max_total_time_s=max_total_time_s,
            centroids_crop_um=centroids_crop_um,
            centroids_full_um=centroids_full_um,
            img_full=img_full_seg,
            img_crop=img_crop_seg,
            ij_percentile_normalize=ij_percentile_normalize,
            pixel_size_full_um=float(pix_full_seg),
            pixel_size_crop_um=float(pix_crop_seg),
            features_crop=features_crop,
            features_full=features_full,
            df_full=df_full,
            df_crop=df_crop,            
            save_dir=None,
            save_prefix="adaptive",
        )

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

        # Save outputs
        cfg_effective = deep_merge_dict(DEFAULT_MATCHER_CONFIG, cfg_selected)
        save_matcher_config_json(cfg_effective, out_dir / "matcher_config_used.json")

        # Save adaptive attempt history with run identifier
        history_with_run = [{**rec, "run_id": run_id} for rec in history]
        _save_history_jsonl(history_with_run, out_dir / "history.jsonl")

        # Export adaptive summary and transforms
        jsonl_path = out_dir / "transforms.jsonl"
        _export_best_everything(
            best_out,
            out_dir=out_dir,
            jsonl_path=jsonl_path,
            matcher_name=str(best_out.get("matcher", "adaptive_best")),
            img_full_orig=img_full_orig,
            img_crop_orig=img_crop_orig,
            pixel_size_full_orig_um=float(pix_full_orig),
            pixel_size_crop_orig_um=float(pix_crop_orig),
            img_full_seg=img_full_seg,
            img_crop_seg=img_crop_seg,
            pixel_size_full_seg_um=float(pix_full_seg),
            pixel_size_crop_seg_um=float(pix_crop_seg),
            margin_px=int(margin_px),
            export_do_full=False,
            export_do_roi=True,
            export_segscale=False,
            save_segscale_transform=False,
            run_id=run_id,
        )

        print("✅ Adaptive matching completed and exported successfully.")
        return best_out, history

    except Exception:
        traceback.print_exc()
        raise


def evaluate_match_quality(
    centroids_crop_um,
    centroids_full_um,
    best_scale,
    best_R,
    best_t,
    inlier_radius_um=2.0,
    frac_inliers_thresh=0.6,
):
    """
    Evaluate how good the match is based on centroid constellation only.

    - Transform crop centroids with the estimated similarity (scale, R, t)
    - For each transformed point, find nearest neighbour in full set
    - Count how many are within inlier_radius_um
    - Return:
        * success (bool)
        * fraction of inliers
        * mean error over inliers
        * all distances for debugging
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

    # 1) Apply estimated transform: patch → full (in µm)
    crop_aligned_um = best_scale * (crop_um @ best_R.T) + best_t  # (Nc,2)

    # 2) Nearest-neighbour distances in full constellation
    tree_full = cKDTree(full_um)
    dists, nn_idx = tree_full.query(crop_aligned_um)  # dists in µm

    # 3) Inlier mask and stats
    inlier_mask = dists <= inlier_radius_um
    n_total = len(dists)
    n_inliers = int(inlier_mask.sum())
    frac_inliers = n_inliers / max(1, n_total)

    mean_error_um = float(dists[inlier_mask].mean()) if n_inliers > 0 else None

    success = frac_inliers >= frac_inliers_thresh

    return dict(
        success=success,
        frac_inliers=frac_inliers,
        mean_error_um=mean_error_um,
        dists=dists,
    )


def score_attempt(out: Dict[str, Any], duration_s: Optional[float] = None) -> float:
    """
    Scoring policy (only used to pick a best fallback if no attempt succeeds):
      - strongly reward having a transform (best_t)
      - prefer higher inlier fraction and lower mean error
      - lightly penalize duration to break ties
    """
    has_t = out.get("best_t", None) is not None
    frac, err = _extract_quality(out)

    if not has_t:
        return -1e12 + 100.0 * frac

    s = 1000.0 * frac - 1.0 * err
    if duration_s is not None and math.isfinite(duration_s):
        s -= 0.05 * float(duration_s)
    return float(s)

def _choose_matcher_order_from_n(n_crop: int) -> Tuple[List[str], str]:
    """
    Implements your policy:
      - n < 20      -> quad, triangles, graph, hashing
      - 20..999     -> triangles, graph, quad, hashing
      - n >= 1000   -> hashing, triangles, graph, quad
    Returns (order, label).
    """
    n_crop = int(n_crop)
    if n_crop >= int(ADAPTIVE_N_LARGE):
        return list(ADAPTIVE_ORDER_GE1000), f"n_crop>= {ADAPTIVE_N_LARGE}"
    if n_crop >= int(ADAPTIVE_N_SMALL):
        return list(ADAPTIVE_ORDER_GE20), f"{ADAPTIVE_N_SMALL}<=n_crop<{ADAPTIVE_N_LARGE}"
    return list(ADAPTIVE_ORDER_LT20), f"n_crop<{ADAPTIVE_N_SMALL}"


def _safe_n_points(x) -> int:    
    if x is None:
        return 0
    if hasattr(x, "shape") and x.shape is not None:
        try:
            return int(x.shape[0])
        except Exception:
            pass
    try:
        return int(len(x))
    except Exception:
        return 0


def _extract_quality(out: Dict[str, Any]) -> Tuple[float, float]:
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


def _quality_tuple(rec: dict):
    mq = rec.get("match_quality", {}) if isinstance(rec.get("match_quality", {}), dict) else {}
    frac = mq.get("frac_inliers", None)
    err = mq.get("mean_error_um", None)
    frac = float(frac) if frac is not None and _is_finite_number(frac) else -1.0
    err = float(err) if err is not None and _is_finite_number(err) else float("inf")
    return (-frac, err)


def _score_record(rec: dict, target_full_um, target_crop_um, pix_rtol=0.02):
    psf = rec.get("pixel_size_full_um", None)
    psc = rec.get("pixel_size_crop_um", None)

    pix_penalty = 0.0
    pix_known = (target_full_um is not None and target_crop_um is not None and psf is not None and psc is not None)
    if pix_known:
        psf = float(psf); psc = float(psc)
        ef = _rel_err(psf, float(target_full_um))
        ec = _rel_err(psc, float(target_crop_um))
        pix_penalty = ef + ec
        if ef > pix_rtol or ec > pix_rtol:
            pix_penalty += 10.0

    q = _quality_tuple(rec)  # (-frac, err)
    return (pix_penalty, q[0], q[1])


def pick_best_transform(records, target_full_um, target_crop_um, pix_rtol=0.02):
    valid = []
    rejected = []
    for r in records:
        ok, problems = validate_transform_record(r)
        if ok and (r.get("success", True) is True):
            valid.append(r)
        else:
            rejected.append((r, problems))

    if not valid:
        msg = ["No valid/successful transforms found."]
        if rejected:
            msg.append(f"Rejected {len(rejected)} record(s). First rejection: {rejected[0][1]}")
        raise ValueError(" ".join(msg))

    scored = [(_score_record(r, target_full_um, target_crop_um, pix_rtol=pix_rtol), r) for r in valid]
    scored.sort(key=lambda x: x[0])
    best = scored[0][1]

    shortlist = []
    if target_full_um is not None and target_crop_um is not None:
        for s, r in scored:
            if s[0] < 10.0:
                shortlist.append(r)
    else:
        shortlist = [r for _, r in scored]

    return best, shortlist


def _save_history_jsonl(history: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in history:
            safe = dict(rec)
            safe.pop("out", None)
            json.dump(safe, f, ensure_ascii=False)
            f.write("\n")

def _save_best_summary_json(best_out: dict, path: Path) -> None:
    """Save a small JSON summary (arrays stripped) for quick inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    q = best_out.get("match_quality") or {}
    summary = {
        "success": bool(best_out.get("success", False)),
        "matcher": best_out.get("matcher", None),
        "best_scale": best_out.get("best_scale", None),
        "best_t_um": (np.asarray(best_out["best_t"], float).ravel().tolist()
                     if best_out.get("best_t", None) is not None else None),
        "best_R": (np.asarray(best_out["best_R"], float).reshape(2, 2).tolist()
                   if best_out.get("best_R", None) is not None else None),
        "match_quality": {
            "frac_inliers": q.get("frac_inliers", None),
            "mean_error_um": q.get("mean_error_um", None),
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
