
"""config.py Configuration helpers (matcher configs, overrides, merges)."""

import copy
import json
from pathlib import Path

MATCHER_CONFIG = {
    "_common": {
        "inlier_radius_um": 2.0,
        "scale_min": 0.5,
        "scale_max": 2.0,
        "angle_max_deg": None,
        "random_state": 42,
        "use_dynamic_scale": True,
        "dynamic_rel_tol": 0.2,
        "use_icp_refinement": True,
        "margin_um": 5.0,
        "frac_inliers_thresh": 0.6,
    },

    "graph": {
        "k_nn_graph": 8,
        "k_ngh_feat": 5,
        "standardize": True,
        "w_shape": 0.4,
        "w_graph": 0.8,
        "w_triangles": 0.3,
        "n_triangles": 10,
        "n_feat_neighbors": 20,
        "n_iters": 50_000,
        "min_inliers_abs": 5,
        "min_inliers_frac": 0.12,
        "min_triangle_area_um2": 1e-6,
        "enforce_unique_full_matches": True,
        "feat_ratio": 0.85,
        "feat_max_dist": None,
        "require_mutual": False,
        "k_spatial": 4,
        "require_feat_consistency": False,
        "prosac": True,
        "pretest_n": 20,
        "refit_on_inliers": True,
        "min_inlier_radius_frac_nn": 0.2,
        "max_candidate_pairs": 200_000,
        "n_candidates_per_patch": 10,
        "n_candidates_per_full": 10,
        "pretest_relax": 0.6,
        "soft_fail_return_best": True,
        "min_inliers_cap_frac": 0.80,
    },

    "quad": {
        "k_nn_quad": 40,
        "n_desc_neighbors": 7,
        "n_iters": 50_000,
        "min_inliers_abs": 5,
        "min_inliers_frac": 0.12,
        "angle_max_deg": None,
        "k_candidates": 8,
        "n_quads_per_center": 14,
        "min_area2": 1e-6,
        "max_candidate_pairs": 30_000,
        "use_triplet_hypotheses": True,
        "early_stop_frac": 1.0,
        "early_stop_inliers": None,
    },

    "triangles": {
        "n_triangles": 5,
        "n_iters": 50_000,
        "min_inliers_abs": 5,
        "min_inliers_frac": 0.12,
        "angle_max_deg": None,
        "k_nn_tri": 8,
        "n_feat_neighbors": 1,
        "max_candidate_pairs": None,
        "early_stop_frac": 1.0,
        "early_stop_inliers": None,
        "min_triangle_area_um2": 1e-6,
        "use_scale_aware_area_floor": True,
        "area_floor_alpha": 0.02,
    },

    "hashing": {
        # main hashing params
        "base_distance_um": 10.0,
        "bin_size_r": 0.1,
        "angle_bin_deg": 10,
        "vote_thresh": 3,
        "n_iters": 50_000,
        "min_inliers_abs": 5,
        "min_inliers_frac": 0.12,
        "angle_max_deg": None,
        "max_neighbors_full": 40,
        "max_pairs_per_anchor": 30,
        "max_k_per_pair": 20,
        "max_candidates_per_bin": 200,
        "max_neighbors_patch": 40,
        "max_pairs_per_anchor_patch": 30,
        "max_k_per_pair_patch": 20,
        "neighbor_bin_radius": 1,
        "max_candidates_test": 120,
        "randomize_candidates": False,
        "pretest_n": 80,
        "early_stop_frac": 1.0,
    },
}


DEFAULT_MATCHER_CONFIG = copy.deepcopy(MATCHER_CONFIG)


def load_matcher_config_json(path: str | Path, default: dict = None) -> dict:
    """Load JSON and deep-merge onto default config."""
    if default is None:
        default = DEFAULT_MATCHER_CONFIG
    path = Path(path)
    with path.open("r") as f:
        user_cfg = json.load(f)
    return deep_merge_dict(default, user_cfg)

def save_matcher_config_json(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

def deep_merge_dict(base: dict, override: dict | None) -> dict:
    """Recursively merge override into base (without mutating base)."""
    if override is None:
        return copy.deepcopy(base)
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def effective_flat_config(cfg_struct: dict, matcher_mode: str) -> dict:
    """Return a flat dict = _common overlaid with matcher section."""
    if "_common" not in cfg_struct:
        raise ValueError("Config must have a '_common' section.")
    out = dict(cfg_struct["_common"])
    out.update(cfg_struct.get(matcher_mode, {}))
    return out


def normalize_runtime_overrides(runtime_overrides: dict | None, matcher_mode: str) -> dict | None:
    """
    Accept either:
      - hierarchical: {"_common": {...}, "graph": {...}}
      - flat (assume matcher-specific): {"n_iters": 20000, ...}
    """
    if not runtime_overrides:
        return None
    if "_common" in runtime_overrides or matcher_mode in runtime_overrides:
        return runtime_overrides
    return {matcher_mode: runtime_overrides}


def _norm_matcher_name(m: str) -> str:
    mm = str(m).strip().lower()
    if mm in ("triangle", "tri", "tris"):
        return "triangles"
    return mm    

