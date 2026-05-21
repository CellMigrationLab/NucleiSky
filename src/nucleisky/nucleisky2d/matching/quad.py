"""quad.py Quad-based matcher."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial import cKDTree
import itertools

import numpy as np
from numba import njit

from .geometry import estimate_similarity
from .geometry import estimate_dynamic_scale_bounds, icp_similarity, bbox_full_px_from_similarity_um

@njit(fastmath=True)
def _robust_quad_descriptor_core(pts4, eps, min_area2):
    c = pts4[0]
    neigh = pts4[1:] - c

    dists = np.empty(3, dtype=np.float64)
    for i in range(3):
        d = np.linalg.norm(neigh[i])
        if not np.isfinite(d):
            return np.zeros(10, dtype=np.float32), False
        dists[i] = d

    sort_idx = np.argsort(dists)
    neigh_sorted = neigh[sort_idx]
    dists_sorted = dists[sort_idx]

    max_dist = dists_sorted[2]
    if max_dist < eps:
        max_dist = eps
    if (not np.isfinite(max_dist)) or max_dist <= eps:
        return np.zeros(10, dtype=np.float32), False

    neigh_norm = neigh_sorted / max_dist
    anchor = neigh_norm[2]
    if np.hypot(anchor[0], anchor[1]) <= eps:
        return np.zeros(10, dtype=np.float32), False

    v1 = neigh_norm[1] - neigh_norm[0]
    v2 = neigh_norm[2] - neigh_norm[0]
    area2 = v1[0] * v2[1] - v1[1] * v2[0]
    if area2 < 0.0:
        area2 = -area2
    if area2 < min_area2:
        return np.zeros(10, dtype=np.float32), False

    angle0 = np.arctan2(anchor[0], anchor[1])
    c_ang = np.cos(angle0)
    s_ang = np.sin(angle0)
    R = np.array([[c_ang, -s_ang], [s_ang, c_ang]], dtype=np.float64)
    neigh_aligned = (R @ neigh_norm.T).T

    if neigh_aligned[2, 1] < 0.0:
        neigh_aligned *= -1.0
    if neigh_aligned[1, 0] < 0.0:
        neigh_aligned[:, 0] *= -1.0

    angles_internal = np.empty(3, dtype=np.float64)
    pairs = ((0,1),(1,2),(2,0))
    for idx in range(3):
        a,b = pairs[idx]
        u = neigh_aligned[a]
        v = neigh_aligned[b]
        denom = np.linalg.norm(u) * np.linalg.norm(v) + eps        
        
        val = (u[0] * v[0] + u[1] * v[1]) / denom
        
        if val < -1.0:
            val = -1.0
        elif val > 1.0:
            val = 1.0
        angles_internal[idx] = np.arccos(val)

    desc = np.empty(10, dtype=np.float32)
    desc[0:3] = (dists_sorted / max_dist).astype(np.float32)
    desc[3:7] = neigh_aligned[0:2].ravel().astype(np.float32)
    desc[7:10] = angles_internal.astype(np.float32)
    return desc, True


def _robust_quad_descriptor(pts4_um, eps=1e-8, min_area2=1e-6):
    pts4 = np.asarray(pts4_um, dtype=float)
    if pts4.shape != (4, 2) or (not np.all(np.isfinite(pts4))):
        return None

    desc, is_valid = _robust_quad_descriptor_core(pts4, float(eps), float(min_area2))
    if not bool(is_valid) or desc.shape != (10,) or (not np.all(np.isfinite(desc))):
        return None
    return desc


def build_local_quads(
    points_um,
    k_nn_quad=20,
    k_candidates=8,
    n_quads_per_center=10,
    random_state=None,
    min_area2=1e-6,
):
    pts = np.asarray(points_um, float)
    N = int(len(pts))
    if N < 4:
        return np.zeros((0, 4), dtype=int), np.zeros((0, 10), dtype=np.float32)

    rng = np.random.default_rng(random_state)

    tree = cKDTree(pts)
    k_eff = min(int(k_nn_quad) + 1, N)
    _, idxs = tree.query(pts, k=k_eff)

    quads, descs = [], []

    k_candidates = int(max(3, min(int(k_candidates), k_eff - 1)))
    n_quads_per_center = int(max(1, n_quads_per_center))

    for i in range(N):
        neighs = idxs[i][1:1 + k_candidates]
        if len(neighs) < 3:
            continue

        # all triplets among these neighbors
        triplets = list(itertools.combinations(neighs.tolist(), 3))
        if not triplets:
            continue

        # sample / cap per center (like n_triangles)
        if len(triplets) > n_quads_per_center:
            sel = rng.choice(len(triplets), size=n_quads_per_center, replace=False)
            triplets = [triplets[j] for j in sel]

        for (a, b, c) in triplets:
            quad_idx = np.array([i, a, b, c], dtype=int)
            desc = _robust_quad_descriptor(pts[quad_idx], min_area2=float(min_area2))
            if desc is None:
                continue
            quads.append(quad_idx)
            descs.append(desc)

    if not quads:
        return np.zeros((0, 4), dtype=int), np.zeros((0, 10), dtype=np.float32)

    return np.stack(quads), np.stack(descs).astype(np.float32, copy=False)


def quad_match_similarity(
    patch_pts_um,
    full_pts_um,
    k_nn_quad=20,
    n_desc_neighbors=6,
    inlier_radius_um=2.0,
    n_iters=100000,
    scale_min=0.5,
    scale_max=2.0,
    angle_max_deg=None,
    min_inliers=10,
    random_state=None,
    k_candidates=8,
    n_quads_per_center=14,
    min_area2=1e-6,
    max_candidate_pairs=30000,
    use_triplet_hypotheses=True,   # tolerate 1 bad neighbor inside quad
    early_stop_inliers=None,       # like quad/triangle: stop when "good enough"
):
    rng = np.random.default_rng(random_state)
    patch_pts_um = np.asarray(patch_pts_um, float)
    full_pts_um  = np.asarray(full_pts_um, float)

    Np = int(len(patch_pts_um))
    Nf = int(len(full_pts_um))
    if Np < 4 or Nf < 4:
        return None, None, None, []

    if early_stop_inliers is None:
        early_stop_inliers = Np
    early_stop_inliers = int(max(1, min(int(early_stop_inliers), Np)))

    # Build many quads/descs
    patch_idx, patch_desc = build_local_quads(
        patch_pts_um,
        k_nn_quad=int(k_nn_quad),
        k_candidates=int(k_candidates),
        n_quads_per_center=int(n_quads_per_center),
        random_state=int(rng.integers(1_000_000_000)),
        min_area2=float(min_area2),
    )
    full_idx, full_desc = build_local_quads(
        full_pts_um,
        k_nn_quad=int(k_nn_quad),
        k_candidates=int(k_candidates),
        n_quads_per_center=int(n_quads_per_center),
        random_state=int(rng.integers(1_000_000_000)),
        min_area2=float(min_area2),
    )

    if patch_idx.shape[0] == 0 or full_idx.shape[0] == 0:
        return None, None, None, []

    # Descriptor tree
    tree_desc = cKDTree(full_desc)
    k_desc = int(max(1, min(int(n_desc_neighbors), full_desc.shape[0])))

    # Candidate quad pairs
    pairs = []
    for qi in range(patch_desc.shape[0]):
        dists, idxs = tree_desc.query(patch_desc[qi], k=k_desc)
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)
        for d, j in zip(dists, idxs):
            if not np.isfinite(d):
                continue
            j = int(j)
            if 0 <= j < full_desc.shape[0]:
                pairs.append((qi, j, float(d)))

    if len(pairs) == 0:
        return None, None, None, []

    # Cap candidate pairs
    max_candidate_pairs = int(max_candidate_pairs) if max_candidate_pairs is not None else None
    if max_candidate_pairs is not None and len(pairs) > max_candidate_pairs:
        pairs.sort(key=lambda x: x[2])
        pairs = pairs[:max_candidate_pairs]

    tree_full = cKDTree(full_pts_um)

    best_score = 0
    best_scale = best_R = best_t = None
    best_inliers = np.array([], dtype=int)

    triplet_subsets = (
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        if bool(use_triplet_hypotheses) else None
    )

    for _ in range(int(n_iters)):
        qi_patch, qi_full, _ = pairs[int(rng.integers(len(pairs)))]
        idx_patch = patch_idx[qi_patch]
        idx_full  = full_idx[qi_full]

        if len(set(idx_patch)) < 4 or len(set(idx_full)) < 4:
            continue

        src4 = patch_pts_um[idx_patch]
        dst4 = full_pts_um[idx_full]

        # Hypothesis: either 4-point or best-of-3-of-4
        hyp_list = [(src4, dst4)]
        if triplet_subsets is not None:
            hyp_list = [(src4[list(s)], dst4[list(s)]) for s in triplet_subsets]

        best_local = None
        for src, dst in hyp_list:
            try:
                scale, R, t = estimate_similarity(src, dst)
            except Exception:
                continue

            if not (float(scale_min) <= float(scale) <= float(scale_max)):
                continue

            if angle_max_deg is not None:
                ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                if abs(ang) > float(angle_max_deg):
                    continue

            pts_trans = float(scale) * (patch_pts_um @ R.T) + t
            dists, _ = tree_full.query(pts_trans, distance_upper_bound=float(inlier_radius_um))
            inliers = np.where(np.isfinite(dists))[0]
            score = int(len(inliers))

            if best_local is None or score > best_local[0]:
                best_local = (score, float(scale), np.asarray(R, float), np.asarray(t, float), inliers)

        if best_local is None:
            continue

        score, scale, R, t, inliers = best_local
        if score > best_score:
            best_score = score
            best_scale, best_R, best_t = scale, R, t
            best_inliers = inliers

            if best_score >= early_stop_inliers:
                break

    if best_score < int(min_inliers):
        return None, None, None, []

    return best_scale, best_R, best_t, best_inliers


def run_quad_based_matching_um(
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
    **quad_kwargs,
):
    centroids_crop_um = np.asarray(centroids_crop_um, float)
    centroids_full_um = np.asarray(centroids_full_um, float)
    Nc = len(centroids_crop_um)

    scale_min_eff, scale_max_eff = float(scale_min), float(scale_max)
    
    # Use shapes for dynamic scale estimation
    if use_dynamic_scale and (df_full is not None) and (df_crop is not None):
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
            f"[quad / scale prior] s ≈ {scale_prior:.3f}, "
            f"effective range = [{scale_min_eff:.3f}, {scale_max_eff:.3f}]"
        )

    # Optional: accept early_stop_frac (0..1) and convert to an inlier count
    early_stop_inliers = quad_kwargs.get("early_stop_inliers", None)
    early_stop_frac = quad_kwargs.get("early_stop_frac", None)
    if early_stop_inliers is None and early_stop_frac is not None:
        early_stop_inliers = int(np.ceil(float(early_stop_frac) * max(1, Nc)))

    best_scale, best_R, best_t, inliers = quad_match_similarity(
        centroids_crop_um,
        centroids_full_um,
        k_nn_quad=int(quad_kwargs.get("k_nn_quad", 20)),
        n_desc_neighbors=int(quad_kwargs.get("n_desc_neighbors", 6)),
        inlier_radius_um=float(inlier_radius_um),
        n_iters=int(quad_kwargs.get("n_iters", 50_000)),
        scale_min=float(scale_min_eff),
        scale_max=float(scale_max_eff),
        angle_max_deg=quad_kwargs.get("angle_max_deg", None),
        min_inliers=int(quad_kwargs.get("min_inliers", max(30, int(0.18 * Nc)))),
        random_state=random_state,

        k_candidates=int(quad_kwargs.get("k_candidates", 8)),
        n_quads_per_center=int(quad_kwargs.get("n_quads_per_center", 14)),
        min_area2=float(quad_kwargs.get("min_area2", 1e-6)),
        max_candidate_pairs=int(quad_kwargs.get("max_candidate_pairs", 30_000)),
        use_triplet_hypotheses=bool(quad_kwargs.get("use_triplet_hypotheses", True)),
        early_stop_inliers=early_stop_inliers,
    )

    if best_scale is None:
        print("Quad-based matching failed.")
        return None, None, None, None

    print("QUAD RANSAC (µm): scale =", best_scale)
    print("QUAD RANSAC (µm): R =\n", best_R)
    print("QUAD RANSAC (µm): t (dy,dx) [µm] =", best_t)
    print("QUAD RANSAC: # inliers =", len(inliers))

    if bool(quad_kwargs.get("use_icp_refinement", True)):
        ref_scale, ref_R, ref_t = icp_similarity(
            centroids_crop_um,
            centroids_full_um,
            best_scale,
            best_R,
            best_t,
            n_iters=10,
            inlier_radius_um=float(inlier_radius_um),
        )
        best_scale, best_R, best_t = ref_scale, ref_R, ref_t
        print("\nQUAD ICP refined (µm): scale =", best_scale)
        print("QUAD ICP refined (µm): R =\n", best_R)
        print("QUAD ICP refined (µm): t (dy,dx) [µm] =", best_t)
    
    # Use passed shapes
    Hf, Wf = full_shape_px[:2]
    Hc, Wc = patch_shape_px[:2]

    bbox = bbox_full_px_from_similarity_um(
        crop_shape_px=(Hc, Wc),
        pixel_size_full_um=float(pixel_size_full_um),
        pixel_size_crop_um=float(pixel_size_patch_um),
        scale=float(best_scale),
        R_yx=np.asarray(best_R),
        t_um_yx=np.asarray(best_t),
        margin_um=float(margin_um),
        full_shape_px=(Hf, Wf),
    )

    return best_scale, best_R, best_t, bbox
