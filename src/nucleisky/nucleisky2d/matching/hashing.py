
"""hashing.py Geometric hashing matcher."""

import zlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from collections import defaultdict


@njit(fastmath=True)
def _build_local_frame_2d_core(x1, y1, x2, y2, eps):
    """
    Pure scalar math implementation to avoid Numba contiguous array 
    warnings and np.empty/np.eye memory initialization bugs.
    """
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx * dx + dy * dy)
    
    # Pre-allocate with zeros (safer than np.empty in Numba)
    B = np.zeros((2, 2), dtype=np.float64)
    
    if not np.isfinite(d) or d <= eps:
        # Dummy Identity Matrix for failure
        B[0, 0] = 1.0
        B[1, 1] = 1.0
        return B, 0.0, False

    # Unit X vector
    ex0 = dx / d
    ex1 = dy / d
    
    # Orthogonal Unit Y vector (-90 deg rotation)
    ey0 = -ex1
    ey1 = ex0
    
    # Explicit matrix population
    B[0, 0] = ex0
    B[0, 1] = ex1
    B[1, 0] = ey0
    B[1, 1] = ey1
    
    return B, d, True

def _build_local_frame_2d(p1, p2, eps=1e-8):
    """Wrapper that unfolds the 1D points into scalars for the Numba core."""
    B, d, is_valid = _build_local_frame_2d_core(
        float(p1[0]), float(p1[1]), 
        float(p2[0]), float(p2[1]), 
        float(eps)
    )
    return B, d, is_valid

from .geometry import (
    estimate_similarity,
    estimate_dynamic_scale_bounds,
    icp_similarity,
    _wrap_angle_rad,
    _circ_diff_rad, bbox_full_px_from_similarity_um,
)


def geometric_hashing_match_similarity(
    patch_pts_um,
    full_pts_um,
    base_distance_um=10.0,
    bin_size_r=0.10,
    angle_bin_deg=5,
    vote_thresh=3,
    inlier_radius_um=2.0,
    scale_min=0.5,
    scale_max=2.0,
    angle_max_deg=None,
    n_iters=100000,
    min_inliers=5,
    random_state=None,
    do_plots=False,
    # FULL hash build caps
    max_neighbors_full=40,
    max_pairs_per_anchor=30,
    max_k_per_pair=20,
    max_candidates_per_bin=400,
    # PATCH sampling caps (kNN-driven sampling)
    max_neighbors_patch=40,
    max_pairs_per_anchor_patch=30,
    max_k_per_pair_patch=20,
    # Robustness additions
    neighbor_bin_radius=1,          # search ± this many bins for (r, theta)
    max_candidates_test=120,        # how many candidates to test per iteration (after ranking)
    randomize_candidates=False,     # if True, sample candidates randomly instead of ranking by closeness

    # Speed / stability
    pretest_n=80,                   # evaluate inliers on subset first (0 disables)
    early_stop_frac=1.0,            # stop if inliers >= early_stop_frac * Np (e.g., 0.95)
):
    """
    Geometric hashing with stability fixes for larger Np:
      - patch sampling uses kNN neighborhoods (fewer wasted iterations)
      - stores (r_norm, theta) in buckets and ranks candidates by descriptor closeness
      - searches neighboring bins to reduce quantization sensitivity
      - optional pretest for cheap early rejection
      - early stop once score is "good enough"
    """
    rng = np.random.default_rng(random_state)

    patch_pts_um = np.asarray(patch_pts_um, float)
    full_pts_um  = np.asarray(full_pts_um, float)

    Np = len(patch_pts_um)
    Nf = len(full_pts_um)
    if Np < 3 or Nf < 3:
        return None, None, None, []

    base = float(base_distance_um)
    if base <= 0:
        raise ValueError("base_distance_um must be > 0.")
    bin_size_r = float(bin_size_r)
    if bin_size_r <= 0:
        raise ValueError("bin_size_r/bin_size_r must be > 0.")
    ang_bin = float(angle_bin_deg)
    if ang_bin <= 0:
        raise ValueError("angle_bin_deg must be > 0.")

    theta_bin_width = np.deg2rad(ang_bin)
    n_theta_bins = int(np.ceil((2.0 * np.pi) / theta_bin_width))

    # Distance gate that is consistent with unknown scale:
    # We want pairs where d_full ~ base, but in patch space d_patch = d_full / scale.
    # So accept d_patch if there exists scale in [scale_min, scale_max] such that
    # base*0.5 <= d_patch*scale <= base*1.5  =>  base*0.5/scale_max <= d_patch <= base*1.5/scale_min
    d_patch_lo = (base * 0.5) / float(scale_max)
    d_patch_hi = (base * 1.5) / float(scale_min)

    # --- Build hash table on FULL points using kNN neighborhoods ---
    tree_full = cKDTree(full_pts_um)
    kN_full = min(int(max_neighbors_full) + 1, Nf)
    _, nn_full = tree_full.query(full_pts_um, k=kN_full)
    nn_full = np.atleast_2d(nn_full)[:, 1:]  # drop self

    # store: (i, j, k, r_norm, theta)
    hash_table = defaultdict(list)

    for i in range(Nf):
        p1 = full_pts_um[i]
        js = nn_full[i][: int(max_pairs_per_anchor)]

        for j in js:
            j = int(j)
            if j == i:
                continue
            p2 = full_pts_um[j]
            B, d, is_valid = _build_local_frame_2d(p1, p2)
            if not is_valid:
                continue
            # full distance gate around base
            if not (base * 0.5 < d < base * 1.5):
                continue

            ks = nn_full[i][: int(max_k_per_pair)]
            for k in ks:
                k = int(k)
                if k in (i, j):
                    continue

                rel = B @ (full_pts_um[k] - p1)
                r_norm = float(np.linalg.norm(rel) / d)  # dimensionless
                theta = float(_wrap_angle_rad(np.arctan2(rel[1], rel[0])))

                r_bin = int(np.floor(r_norm / bin_size_r))
                theta_bin = int(np.floor(theta / theta_bin_width)) % n_theta_bins
                key = (r_bin, theta_bin)

                bucket = hash_table[key]
                if len(bucket) < int(max_candidates_per_bin):
                    bucket.append((i, j, k, r_norm, theta))

    # --- Patch kNN for efficient hypothesis sampling ---
    tree_patch = cKDTree(patch_pts_um)
    kN_patch = min(int(max_neighbors_patch) + 1, Np)
    _, nn_patch = tree_patch.query(patch_pts_um, k=kN_patch)
    nn_patch = np.atleast_2d(nn_patch)[:, 1:]  # drop self

    # scoring uses full tree
    best_score, best_scale, best_R, best_t, best_inliers = 0, None, None, None, []
    early_stop_inliers = int(np.ceil(float(early_stop_frac) * Np))
    early_stop_inliers = max(int(min_inliers), min(early_stop_inliers, Np))

    # pretest subset indices (fixed for stability)
    if int(pretest_n) > 0 and Np > int(pretest_n):
        pre_idx = rng.choice(Np, size=int(pretest_n), replace=False)
    else:
        pre_idx = None

    def _score_inliers(pts_trans):
        dists, _ = tree_full.query(pts_trans, distance_upper_bound=float(inlier_radius_um))
        return np.where(np.isfinite(dists))[0]

    for _ in range(int(n_iters)):
        # --- sample a good patch anchor i, then j,k from its neighbors ---
        i = int(rng.integers(Np))
        neighs_i = nn_patch[i][: int(max_pairs_per_anchor_patch)]
        if len(neighs_i) < 2:
            continue

        # choose j with acceptable distance (scale-aware gate)
        j = None
        for _try in range(6):
            jj = int(neighs_i[int(rng.integers(len(neighs_i)))])
            if jj == i:
                continue
            d_patch = float(np.linalg.norm(patch_pts_um[jj] - patch_pts_um[i]))
            if np.isfinite(d_patch) and (d_patch_lo <= d_patch <= d_patch_hi):
                j = jj
                break
        if j is None:
            continue

        # choose k distinct from i,j
        neighs_k = nn_patch[i][: int(max_k_per_pair_patch)]
        if len(neighs_k) < 2:
            continue
        k = None
        for _try in range(6):
            kk = int(neighs_k[int(rng.integers(len(neighs_k)))])
            if kk != i and kk != j:
                k = kk
                break
        if k is None:
            continue

        cp1, cp2, cp3 = patch_pts_um[i], patch_pts_um[j], patch_pts_um[k]
        B, d, is_valid = _build_local_frame_2d(cp1, cp2)
        if not is_valid:
            continue

        # define local frame from (i -> j)
        rel = B @ (cp3 - cp1)

        r_norm_p = float(np.linalg.norm(rel) / d)
        theta_p  = float(_wrap_angle_rad(np.arctan2(rel[1], rel[0])))

        r_bin0 = int(np.floor(r_norm_p / bin_size_r))
        t_bin0 = int(np.floor(theta_p / theta_bin_width)) % n_theta_bins

        # --- gather candidates from neighboring bins ---
        cand = []
        rb = int(neighbor_bin_radius)
        for dr in range(-rb, rb + 1):
            r_bin = r_bin0 + dr
            for dt in range(-rb, rb + 1):
                t_bin = (t_bin0 + dt) % n_theta_bins
                key = (r_bin, t_bin)
                if key in hash_table:
                    cand.extend(hash_table[key])

        if not cand:
            continue

        # optional gate: avoid extremely tiny bins if vote_thresh > 1
        if int(vote_thresh) > 1 and len(cand) < int(vote_thresh):
            continue

        # --- choose which candidates to actually test ---
        if randomize_candidates and len(cand) > int(max_candidates_test):
            idx = rng.choice(len(cand), size=int(max_candidates_test), replace=False)
            cand_sel = [cand[t] for t in idx]
        else:
            # rank by closeness in (r_norm, theta) (cheap and effective)
            # smaller is better
            scores = np.empty(len(cand), dtype=float)
            for u, (_, _, _, r_f, th_f) in enumerate(cand):
                drn = abs(float(r_f) - r_norm_p)
                dth = _circ_diff_rad(float(th_f), theta_p) / theta_bin_width
                scores[u] = drn / max(bin_size_r, 1e-8) + dth
            # take best K
            K = min(int(max_candidates_test), len(cand))
            if K < len(cand):
                keep = np.argpartition(scores, K - 1)[:K]
                cand_sel = [cand[t] for t in keep]
            else:
                cand_sel = cand

        # --- evaluate selected candidates ---
        src = np.stack([cp1, cp2, cp3], axis=0)

        for fi, fj, fk, _, _ in cand_sel:
            dst = np.stack([full_pts_um[int(fi)], full_pts_um[int(fj)], full_pts_um[int(fk)]], axis=0)

            try:
                scale, R, t = estimate_similarity(src, dst)
            except Exception:
                continue

            scale = float(scale)
            if not (float(scale_min) <= scale <= float(scale_max)):
                continue

            if angle_max_deg is not None:
                angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                if abs(angle) > float(angle_max_deg):
                    continue

            # Pretest (cheap): score on a subset first
            if pre_idx is not None:
                pts_trans_pre = scale * (patch_pts_um[pre_idx] @ R.T) + t
                inl_pre = _score_inliers(pts_trans_pre)
                # optimistic upper bound if all remaining points were inliers
                ub = int(len(inl_pre)) + (Np - int(pretest_n))
                if ub <= best_score:
                    continue

            # Full score
            pts_trans = scale * (patch_pts_um @ R.T) + t
            inliers = _score_inliers(pts_trans)
            score = int(len(inliers))

            if score > best_score:
                best_score = score
                best_scale, best_R, best_t = scale, np.asarray(R, float), np.asarray(t, float)
                best_inliers = inliers

                # Early stop (quad-style)
                if best_score >= early_stop_inliers:
                    break

        if best_score >= early_stop_inliers:
            break

    if best_score < int(min_inliers):
        return None, None, None, []

    return best_scale, best_R, best_t, best_inliers


def run_geometric_hashing_matching_um(
    centroids_crop_um,
    centroids_full_um,
    full_shape_px: tuple[int, int],
    patch_shape_px: tuple[int, int],    
    pixel_size_full_um,
    pixel_size_patch_um,
    inlier_radius_um=2.0,
    scale_min=0.5,
    scale_max=2.0,
    random_state=42,
    margin_um=5.0,
    df_full=None,
    df_crop=None,
    use_dynamic_scale=True,
    dynamic_rel_tol=0.1,
    **hash_kwargs,
):
    centroids_crop_um = np.asarray(centroids_crop_um, float)
    centroids_full_um = np.asarray(centroids_full_um, float)
    Nc = len(centroids_crop_um)

    scale_min_eff, scale_max_eff = float(scale_min), float(scale_max)
    if use_dynamic_scale and df_full is not None and df_crop is not None:
        scale_prior, scale_min_eff, scale_max_eff = estimate_dynamic_scale_bounds(
            df_full=df_full,
            df_patch=df_crop,
            pixel_size_full_um=float(pixel_size_full_um),
            pixel_size_patch_um=float(pixel_size_patch_um),
            full_shape_px=full_shape_px,
            patch_shape_px=patch_shape_px,
            coarse_scale_min=float(scale_min),
            coarse_scale_max=float(scale_max),
            rel_tol=float(dynamic_rel_tol),
        )
        print(
            f"[hashing / scale prior] s ≈ {scale_prior:.3f}, "
            f"effective range = [{scale_min_eff:.3f}, {scale_max_eff:.3f}]"
        )

    scale, R, t, inliers = geometric_hashing_match_similarity(
        patch_pts_um=centroids_crop_um,
        full_pts_um=centroids_full_um,
        base_distance_um=float(hash_kwargs.get("base_distance_um", 10.0)),
        bin_size_r=float(hash_kwargs.get("bin_size_r", 0.1)),
        angle_bin_deg=float(hash_kwargs.get("angle_bin_deg", 10)),
        vote_thresh=int(hash_kwargs.get("vote_thresh", 3)),
        inlier_radius_um=float(inlier_radius_um),
        scale_min=float(scale_min_eff),
        scale_max=float(scale_max_eff),
        angle_max_deg=hash_kwargs.get("angle_max_deg", None),
        n_iters=int(hash_kwargs.get("n_iters", 50_000)),
        min_inliers=int(hash_kwargs.get("min_inliers", max(20, int(0.12 * Nc)))),
        random_state=random_state,

        # full caps
        max_neighbors_full=int(hash_kwargs.get("max_neighbors_full", 40)),
        max_pairs_per_anchor=int(hash_kwargs.get("max_pairs_per_anchor", 30)),
        max_k_per_pair=int(hash_kwargs.get("max_k_per_pair", 20)),
        max_candidates_per_bin=int(hash_kwargs.get("max_candidates_per_bin", 200)),

        # patch caps
        max_neighbors_patch=int(hash_kwargs.get("max_neighbors_patch", 40)),
        max_pairs_per_anchor_patch=int(hash_kwargs.get("max_pairs_per_anchor_patch", 30)),
        max_k_per_pair_patch=int(hash_kwargs.get("max_k_per_pair_patch", 20)),

        # robustness
        neighbor_bin_radius=int(hash_kwargs.get("neighbor_bin_radius", 1)),
        max_candidates_test=int(hash_kwargs.get("max_candidates_test", 120)),
        randomize_candidates=bool(hash_kwargs.get("randomize_candidates", False)),

        # speed
        pretest_n=int(hash_kwargs.get("pretest_n", 80)),
        early_stop_frac=float(hash_kwargs.get("early_stop_frac", 1.0)),
    )


    if scale is None:
        print("Geometric hashing matcher failed.")
        return None, None, None, None

    print("HASHING RANSAC (µm): scale =", scale)
    print("HASHING RANSAC (µm): R =\n", R)
    print("HASHING RANSAC (µm): t (dy,dx) [µm] =", t)
    print("HASHING RANSAC: # inliers =", len(inliers))

    if bool(hash_kwargs.get("use_icp_refinement", True)):
        ref_scale, ref_R, ref_t = icp_similarity(
            centroids_crop_um,
            centroids_full_um,
            scale,
            R,
            t,
            n_iters=10,
            inlier_radius_um=float(inlier_radius_um),
        )
        scale, R, t = ref_scale, ref_R, ref_t
        print("\nHASHING ICP refined (µm): scale =", scale)
        print("HASHING ICP refined (µm): R =\n", R)
        print("HASHING ICP refined (µm): t (dy,dx) [µm] =", t)
    
    Hf, Wf = full_shape_px[:2]
    Hc, Wc = patch_shape_px[:2]

    bbox = bbox_full_px_from_similarity_um(
        crop_shape_px=(Hc, Wc),
        pixel_size_full_um=float(pixel_size_full_um),
        pixel_size_crop_um=float(pixel_size_patch_um),
        scale=float(scale),
        R_yx=np.asarray(R),
        t_um_yx=np.asarray(t),
        margin_um=float(margin_um),
        full_shape_px=(Hf, Wf),
    )

    return scale, R, t, bbox
