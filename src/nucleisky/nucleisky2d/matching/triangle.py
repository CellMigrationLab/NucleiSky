"""triangle.py Triangle-based matcher."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from .geometry import build_geometric_knn_graph, build_triangle_node_features
from .geometry import estimate_similarity
from .geometry import estimate_dynamic_scale_bounds, icp_similarity, bbox_full_px_from_similarity_um
from ..features import _zscore_with_ref


@njit(fastmath=True)
def _triangle_area2_core(p1, p2, p3):
    v10 = p2[0] - p1[0]
    v11 = p2[1] - p1[1]
    v20 = p3[0] - p1[0]
    v21 = p3[1] - p1[1]
    cross = v10 * v21 - v11 * v20
    if cross < 0.0:
        cross = -cross
    return cross


@njit(fastmath=True)
def _apply_similarity_2d_core(pts, scale, R, t):
    return scale * (pts @ R.T) + t



def triangle_match_similarity(
    patch_pts_um,
    full_pts_um,
    n_triangles=5,
    inlier_radius_um=2.0,
    n_iters=50000,
    scale_min=0.5,
    scale_max=2.0,
    angle_max_deg=None,
    min_inliers=5,
    random_state=None,
    k_nn_tri=8,
    n_feat_neighbors=1,
    max_candidate_pairs=None,
    early_stop_inliers=None,
    early_stop_frac=None,
    min_triangle_area_um2=1e-6,

    # ---- OPTIONAL stability: make area floor scale-aware ----
    use_scale_aware_area_floor=True,
    area_floor_alpha=0.02,
):
    rng = np.random.default_rng(random_state)

    patch_pts_um = np.asarray(patch_pts_um, float)
    full_pts_um  = np.asarray(full_pts_um,  float)

    Np = int(len(patch_pts_um))
    Nf = int(len(full_pts_um))
    if Np < 3 or Nf < 3:
        return None, None, None, []

    # Early stop target
    if early_stop_inliers is None and early_stop_frac is not None:
        early_stop_inliers = int(np.ceil(float(early_stop_frac) * max(1, Np)))
    if early_stop_inliers is None:
        early_stop_inliers = Np
    early_stop_inliers = int(max(1, min(int(early_stop_inliers), Np)))

    # ---- Build triangle features (k is exposed) ----
    k_nn_tri = int(max(3, k_nn_tri))
    G_full  = build_geometric_knn_graph(full_pts_um,  k=k_nn_tri)
    G_patch = build_geometric_knn_graph(patch_pts_um, k=k_nn_tri)

    tri_full  = build_triangle_node_features(
        full_pts_um,  G_full,
        n_triangles=int(n_triangles),
        min_triangle_area_um2=float(min_triangle_area_um2),
    )
    tri_patch = build_triangle_node_features(
        patch_pts_um, G_patch,
        n_triangles=int(n_triangles),
        min_triangle_area_um2=float(min_triangle_area_um2),
    )

    tri_full_z, tri_mu, tri_sigma = _zscore_with_ref(tri_full,  ref_mu=None,  ref_sigma=None)
    tri_patch_z, _, _             = _zscore_with_ref(tri_patch, ref_mu=tri_mu, ref_sigma=tri_sigma)

    tree_feat = cKDTree(tri_full_z)

    k_feat = int(max(1, min(int(n_feat_neighbors), Nf)))
    pairs = []
    for i in range(Np):
        dists, idxs = tree_feat.query(tri_patch_z[i], k=k_feat)
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)
        for d, j in zip(dists, idxs):
            if np.isfinite(d):
                pairs.append((i, int(j), float(d)))

    if len(pairs) < 3:
        return None, None, None, []

    if max_candidate_pairs is not None:
        max_candidate_pairs = int(max_candidate_pairs)
        if len(pairs) > max_candidate_pairs:
            pairs.sort(key=lambda x: x[2])
            pairs = pairs[:max_candidate_pairs]

    tree_full_spatial = cKDTree(full_pts_um)

    area_floor_um2 = float(min_triangle_area_um2)


    if bool(use_scale_aware_area_floor):

        def _median_nn(pts):
            pts = np.asarray(pts, float)
            if len(pts) < 3:
                return 1.0
            tree = cKDTree(pts)
            d, _ = tree.query(pts, k=2)
            d2 = np.asarray(d)[:, 1]
            d2 = d2[np.isfinite(d2) & (d2 > 0)]
            return float(np.median(d2)) if d2.size else 1.0

        med_nn_full  = _median_nn(full_pts_um)
        med_nn_patch = _median_nn(patch_pts_um)

        alpha = float(area_floor_alpha)
        if np.isfinite(med_nn_full) and med_nn_full > 0:
            area_floor_um2 = max(area_floor_um2, (alpha * med_nn_full) ** 2)
        if np.isfinite(med_nn_patch) and med_nn_patch > 0:
            area_floor_um2 = max(area_floor_um2, (alpha * med_nn_patch) ** 2)

    area2_floor = 2.0 * float(area_floor_um2)

    best_scale = best_R = best_t = None
    best_inliers = np.array([], dtype=int)
    best_score = 0

    # RANSAC loop
    for _ in range(int(n_iters)):
        idxs_h = rng.choice(len(pairs), size=3, replace=False)
        i_list = [pairs[h][0] for h in idxs_h]
        j_list = [pairs[h][1] for h in idxs_h]

        if len(set(i_list)) < 3 or len(set(j_list)) < 3:
            continue

        src = patch_pts_um[i_list]
        dst = full_pts_um[j_list]

        if float(_triangle_area2_core(src[0], src[1], src[2])) < area2_floor:
            continue
        if float(_triangle_area2_core(dst[0], dst[1], dst[2])) < area2_floor:
            continue

        try:
            scale, R, t = estimate_similarity(src, dst)
        except Exception:
            continue

        if not (float(scale_min) <= float(scale) <= float(scale_max)):
            continue

        if angle_max_deg is not None:
            angle_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            if abs(angle_deg) > float(angle_max_deg):
                continue

        pts_trans = _apply_similarity_2d_core(patch_pts_um, float(scale), np.asarray(R, dtype=float), np.asarray(t, dtype=float))
        dists, _ = tree_full_spatial.query(pts_trans, distance_upper_bound=float(inlier_radius_um))
        inliers = np.where(np.isfinite(dists))[0]
        score = int(len(inliers))

        if score > best_score:
            best_score = score
            best_scale = float(scale)
            best_R = np.asarray(R, float)
            best_t = np.asarray(t, float)
            best_inliers = inliers

            if best_score >= early_stop_inliers:
                break

    if best_score < int(min_inliers):
        return None, None, None, []

    return best_scale, best_R, best_t, best_inliers


def run_triangle_based_matching_um(
    centroids_crop_um,
    centroids_full_um,
    # CHANGED: Accept shapes, NOT image arrays
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
    **tri_kwargs,
):
    centroids_crop_um = np.asarray(centroids_crop_um, float)
    centroids_full_um = np.asarray(centroids_full_um, float)
    Nc = int(len(centroids_crop_um))

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
            f"[triangle / scale prior] s ≈ {scale_prior:.3f}, "
            f"effective range = [{scale_min_eff:.3f}, {scale_max_eff:.3f}]"
        )

    # early stop: accept either inliers or frac
    early_stop_inliers = tri_kwargs.get("early_stop_inliers", None)
    early_stop_frac    = tri_kwargs.get("early_stop_frac", None)
    min_triangle_area_um2 = float(tri_kwargs.get("min_triangle_area_um2", 1e-6))
    use_scale_aware_area_floor = bool(tri_kwargs.get("use_scale_aware_area_floor", True))
    area_floor_alpha = float(tri_kwargs.get("area_floor_alpha", 0.02))

    best_scale, best_R, best_t, inliers = triangle_match_similarity(
        centroids_crop_um,
        centroids_full_um,
        n_triangles=int(tri_kwargs.get("n_triangles", 5)),
        inlier_radius_um=float(inlier_radius_um),
        n_iters=int(tri_kwargs.get("n_iters", 50000)),
        scale_min=float(scale_min_eff),
        scale_max=float(scale_max_eff),
        angle_max_deg=tri_kwargs.get("angle_max_deg", None),
        min_inliers=int(tri_kwargs.get("min_inliers", max(20, int(0.12 * Nc)))),
        random_state=random_state,
        k_nn_tri=int(tri_kwargs.get("k_nn_tri", 8)),
        n_feat_neighbors=int(tri_kwargs.get("n_feat_neighbors", 1)),
        max_candidate_pairs=tri_kwargs.get("max_candidate_pairs", None),
        early_stop_inliers=early_stop_inliers,
        early_stop_frac=early_stop_frac,
        min_triangle_area_um2=min_triangle_area_um2,
        use_scale_aware_area_floor=use_scale_aware_area_floor,
        area_floor_alpha=area_floor_alpha,
    )

    if best_scale is None:
        print("Triangle-based matching failed.")
        return None, None, None, None

    print("TRIANGLE RANSAC (µm): scale =", best_scale)
    print("TRIANGLE RANSAC (µm): R =\n", best_R)
    print("TRIANGLE RANSAC (µm): t (dy,dx) [µm] =", best_t)
    print("TRIANGLE RANSAC: # inliers =", len(inliers))

    if bool(tri_kwargs.get("use_icp_refinement", True)):
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
        print("\nTRIANGLE ICP refined (µm): scale =", best_scale)
        print("TRIANGLE ICP refined (µm): R =\n", best_R)
        print("TRIANGLE ICP refined (µm): t (dy,dx) [µm] =", best_t)
    
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
