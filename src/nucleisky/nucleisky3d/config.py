"""config.py Configuration helpers (matcher configs, overrides, merges) for 3D."""

import copy
import json
from pathlib import Path

MATCHER_CONFIG = {
    "_common": {
        "inlier_radius_um": 3.0,
        "scale_min": 0.8,
        "scale_max": 1.2,
        "angle_max_deg": None,
        "random_state": 42,
        "use_dynamic_scale": False,
        "dynamic_rel_tol": 0.2,
        "use_icp_refinement": True,
        "margin_um": 5.0,
        "frac_inliers_thresh": 0.45,
        "sanitize_dedup_radius_um": 0.0,
        "sanitize_drop_nonfinite": True,
        "sanitize_nn_outlier_percentile": None,
        "sanitize_min_points": 4,
    },
    "pyramid": {
        "use_dynamic_scale": False,
        "dynamic_rel_tol": 0.1,
        "n_tetrahedra": 15,
        "n_iters": 150_000,
        "min_inliers": 5,
        "k_nn_tetra": 20,
        "n_feat_neighbors": 3,
        "max_candidate_pairs": None,
        "early_stop_inliers": None,
        "early_stop_frac": None,
        "use_icp_refinement": True,
        "icp_iters": 10,
    },
    "hashing3d": {
        "use_dynamic_scale": False,
        "dynamic_rel_tol": 0.1,
        "base_distance_um": 10.0,
        "bin_size_xyz": 0.15,
        "vote_thresh": 3,
        "n_iters": 50_000,
        "min_inliers_abs": 20,
        "min_inliers_frac": 0.12,
        "min_inliers_hard_floor": 3,
        "min_inliers_cap_frac": 0.80,
        "max_neighbors_full": 40,
        "max_pairs_per_anchor": 20,
        "max_k_per_pair": 16,
        "max_l_per_base": 20,
        "max_candidates_per_bin": 300,
        "max_neighbors_patch": 40,
        "max_pairs_per_anchor_patch": 20,
        "max_k_per_pair_patch": 16,
        "max_l_per_base_patch": 20,
        "neighbor_bin_radius": 1,
        "max_candidates_test": 120,
        "randomize_candidates": False,
        "pretest_n": 80,
        "early_stop_frac": 1.0,
        "use_icp_refinement": True,
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
    section = cfg_struct.get(matcher_mode, {})
    if not section and matcher_mode == "hashing":
        section = cfg_struct.get("hashing3d", {})
    out.update(section)
    return out


def normalize_runtime_overrides(runtime_overrides: dict | None, matcher_mode: str) -> dict | None:
    """
    Accept either:
      - hierarchical: {"_common": {...}, "pyramid": {...}}
      - flat (assume matcher-specific): {"n_iters": 20000, ...}
    """
    if not runtime_overrides:
        return None
    if matcher_mode == "hashing" and "hashing3d" in runtime_overrides and "hashing" not in runtime_overrides:
        runtime_overrides = dict(runtime_overrides)
        runtime_overrides["hashing"] = runtime_overrides["hashing3d"]
    if "_common" in runtime_overrides or matcher_mode in runtime_overrides:
        return runtime_overrides
    return {matcher_mode: runtime_overrides}


def _norm_matcher_name(m: str) -> str:
    mm = str(m).strip().lower()
    if mm in ("hashing3d", "hashing", "hash"):
        return "hashing"
    if mm in ("pyramid3d", "tetra", "tetrahedron"):
        return "pyramid"
    return mm
