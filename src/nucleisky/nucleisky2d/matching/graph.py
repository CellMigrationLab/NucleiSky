"""graph.py Graph-based matcher."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from .geometry import (
    _ensure_points_yx,
    _wrap_pi,
    _triangle_area2,
    _knn_candidates,
    build_triangle_node_features,
    compute_min_inliers_stable,
    estimate_dynamic_scale_bounds,
    icp_similarity,
    build_geometric_knn_graph, estimate_similarity, bbox_full_px_from_similarity_um,
)


from ..features import _robust_median, _robust_mad, _sanitize_features, _zscore_with_ref


_EPS = 1e-8


def _estimate_full_median_nn(full_pts_um, sample_max=2000):
    """
    Robust typical spacing in the FULL constellation, used to prevent too-small inlier radius.
    Returns median of 2nd-nearest neighbor distances (k=2).
    """
    pts = np.asarray(full_pts_um, float)
    N = len(pts)
    if N < 3:
        return 1.0
    if N > int(sample_max):
        rng = np.random.default_rng(0)
        sel = rng.choice(N, size=int(sample_max), replace=False)
        ptsq = pts[sel]
    else:
        ptsq = pts
    tree = cKDTree(pts)
    dists, _ = tree.query(ptsq, k=2)
    d2 = np.asarray(dists)[:, 1]
    d2 = d2[np.isfinite(d2) & (d2 > 0)]
    return float(np.median(d2)) if d2.size else 1.0


def build_graph_constellation_features(G, points_um, k_ngh_feat=6):
    """
    Rotation-robust node descriptor.

    Output dim = 3*k_ngh_feat + 3
      [d_norm_1..k, cosΔ_1..k, sinΔ_1..k, d_center_norm, local/global, degree_norm]
    """
    pts = _ensure_points_yx(points_um, name="points_um")
    N = int(len(pts))
    k = int(k_ngh_feat)
    F = 3 * k + 3
    feats = np.zeros((N, F), dtype=np.float32)
    if N == 0:
        return feats

    center = np.mean(pts, axis=0)
    d_center = np.linalg.norm(pts - center, axis=1)
    rad = float(np.max(d_center)) if np.isfinite(np.max(d_center)) and np.max(d_center) > 0 else 1.0

    edge_dists = [d.get("dist", np.nan) for _, _, d in G.edges(data=True)]
    global_med = _robust_median(edge_dists, fallback=1.0, positive_only=True)

    for i in G.nodes:
        neighs = G.nodes[i].get("neighbors", [])
        if len(neighs) == 0:
            feats[i, -3] = 0.0
            feats[i, -2] = 1.0
            feats[i, -1] = 0.0
            continue

        dists = np.array([G[i][j].get("dist", np.nan) for j in neighs], float)
        angs  = np.array([G[i][j].get("angle", np.nan) for j in neighs], float)

        order = np.argsort(np.nan_to_num(dists, nan=np.inf))
        dists = dists[order]
        angs  = angs[order]

        local_med = _robust_median(dists, fallback=global_med, positive_only=True)
        d_norm = dists / (local_med + _EPS)

        anchor = 0.0
        finite_ang = angs[np.isfinite(angs)]
        if finite_ang.size:
            anchor = float(finite_ang[0])

        dtheta = _wrap_pi(angs - anchor)
        cosv = np.cos(dtheta)
        sinv = np.sin(dtheta)

        m = min(k, len(d_norm))
        feats[i, 0:m]           = np.nan_to_num(d_norm[:m], nan=0.0, posinf=0.0, neginf=0.0)
        feats[i, k:k+m]         = np.nan_to_num(cosv[:m],   nan=0.0, posinf=0.0, neginf=0.0)
        feats[i, 2*k:2*k+m]     = np.nan_to_num(sinv[:m],   nan=0.0, posinf=0.0, neginf=0.0)

        feats[i, -3] = 0.0
        feats[i, -2] = float(local_med) / (global_med + _EPS)
        feats[i, -1] = float(len(neighs)) / (float(k) + _EPS)

    feats[~np.isfinite(feats)] = 0.0
    return feats.astype(np.float32)


def build_combined_node_features(
    centroids_um,
    shape_features,
    k_nn_graph=10,
    k_ngh_feat=6,
    standardize=True,
    w_shape=0.3,
    w_graph=1.0,
    w_triangles=0.7,
    n_triangles=10,
    min_triangle_area_um2=1e-6,
    ref_stats=None,
    return_stats=False,
):
    pts_um = _ensure_points_yx(centroids_um, name="centroids_um")
    Xshape = np.asarray(shape_features, float)
    if Xshape.ndim == 1:
        Xshape = Xshape[:, None]

    N = int(len(pts_um))
    shape_dim = int(Xshape.shape[1])
    graph_dim = 3 * int(k_ngh_feat) + 3
    tri_dim = 2
    F_total = shape_dim + graph_dim + tri_dim

    if N == 0:
        empty = np.zeros((0, F_total), dtype=np.float32)
        if return_stats:
            stats = dict(shape_mu=None, shape_sigma=None, graph_mu=None, graph_sigma=None, tri_mu=None, tri_sigma=None)
            return empty, stats
        return empty

    if Xshape.shape[0] != N:
        raise ValueError(f"Shape features and centroids must align. Got {Xshape.shape[0]} vs {N}")

    G = build_geometric_knn_graph(pts_um, k=k_nn_graph)
    graph_feats = build_graph_constellation_features(G, pts_um, k_ngh_feat=k_ngh_feat)
    tri_feats = build_triangle_node_features(pts_um, G, n_triangles=n_triangles, min_triangle_area_um2=min_triangle_area_um2)

    if ref_stats is None:
        shape_mu = shape_sigma = None
        graph_mu = graph_sigma = None
        tri_mu = tri_sigma = None
    else:
        shape_mu = ref_stats.get("shape_mu", None)
        shape_sigma = ref_stats.get("shape_sigma", None)
        graph_mu = ref_stats.get("graph_mu", None)
        graph_sigma = ref_stats.get("graph_sigma", None)
        tri_mu = ref_stats.get("tri_mu", None)
        tri_sigma = ref_stats.get("tri_sigma", None)

    if standardize:
        shape_z, shape_mu, shape_sigma = _zscore_with_ref(Xshape, ref_mu=shape_mu, ref_sigma=shape_sigma)
        graph_z, graph_mu, graph_sigma = _zscore_with_ref(graph_feats, ref_mu=graph_mu, ref_sigma=graph_sigma)
        tri_z, tri_mu, tri_sigma = _zscore_with_ref(tri_feats, ref_mu=tri_mu, ref_sigma=tri_sigma)
    else:
        shape_z = np.nan_to_num(Xshape, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        graph_z = np.nan_to_num(graph_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        tri_z   = np.nan_to_num(tri_feats,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    feats = np.concatenate([w_shape*shape_z, w_graph*graph_z, w_triangles*tri_z], axis=1).astype(np.float32)
    feats[~np.isfinite(feats)] = 0.0

    if return_stats:
        return feats, dict(shape_mu=shape_mu, shape_sigma=shape_sigma,
                           graph_mu=graph_mu, graph_sigma=graph_sigma,
                           tri_mu=tri_mu, tri_sigma=tri_sigma)
    return feats


def graph_match_similarity(
    patch_pts_um,
    full_pts_um,
    feats_patch,
    feats_full,
    n_feat_neighbors=10,

    # keep multiple candidates per node when features are ambiguous
    n_candidates_per_patch=3,
    n_candidates_per_full=3,

    inlier_radius_um=2.0,
    n_iters=50000,
    scale_min=0.5,
    scale_max=2.0,
    angle_max_deg=None,
    min_inliers=5,
    random_state=None,
    min_triangle_area_um2=1e-6,
    enforce_unique_full_matches=True,
    feat_ratio=0.95,
    feat_max_dist=None,
    require_mutual=True,
    k_spatial=5,
    require_feat_consistency=True,
    prosac=True,
    pretest_n=60,
    pretest_relax=0.8,
    refit_on_inliers=True,
    min_inlier_radius_frac_nn=0.12,
    max_candidate_pairs=60000,    
    soft_fail_return_best=True,           
    min_inliers_cap_frac=0.80,
    early_stop_inliers=None,
):
    rng = np.random.default_rng(random_state)

    patch_pts_um = _ensure_points_yx(patch_pts_um, name="patch_pts_um")
    full_pts_um  = _ensure_points_yx(full_pts_um,  name="full_pts_um")

    feats_patch = np.asarray(feats_patch, float)
    feats_full  = np.asarray(feats_full,  float)

    Np = int(len(patch_pts_um))
    Nf = int(len(full_pts_um))
    if Np < 3 or Nf < 3:
        return None, None, None, []

    # Handle early stop target
    if early_stop_inliers is None:
        early_stop_inliers = Np
    early_stop_inliers = int(max(min_inliers, min(early_stop_inliers, Np)))

    feats_full_s  = _sanitize_features(feats_full)
    feats_patch_s = _sanitize_features(feats_patch, ref_feat=feats_full_s)

    if feats_full_s.shape[0] != Nf or feats_patch_s.shape[0] != Np:
        raise ValueError("Feature row counts must match point counts (full and patch).")

    tree_full_feat  = cKDTree(feats_full_s)
    tree_patch_feat = cKDTree(feats_patch_s)

    keep_k_patch = max(1, min(int(n_candidates_per_patch), Nf))
    keep_k_full  = max(1, min(int(n_candidates_per_full), Np))

    # Ask for enough k to allow ambiguity + ratio logic
    k_full  = max(2, min(int(n_feat_neighbors), Nf, max(2, keep_k_patch + 1)))
    k_patch = max(2, min(int(n_feat_neighbors), Np, max(2, keep_k_full + 1)))

    # --- forward candidates: patch i -> multiple full j ---
    forward = {}  # (i,j) -> dist
    for i in range(Np):
        d, js = _knn_candidates(tree_full_feat, feats_patch_s[i], k=k_full, keep_k=keep_k_patch, ratio=feat_ratio)
        for dd, j in zip(np.atleast_1d(d), np.atleast_1d(js)):
            j = int(j)
            if 0 <= j < Nf and np.isfinite(dd):
                forward[(int(i), j)] = float(dd)

    # --- reverse candidates: full j -> multiple patch i (for mutual) ---
    reverse = {}
    for j in range(Nf):
        d, is_ = _knn_candidates(tree_patch_feat, feats_full_s[j], k=k_patch, keep_k=keep_k_full, ratio=feat_ratio)
        for dd, i in zip(np.atleast_1d(d), np.atleast_1d(is_)):
            i = int(i)
            if 0 <= i < Np and np.isfinite(dd):
                reverse[(int(i), int(j))] = float(dd)

    # combine pairs
    if require_mutual:
        pairs = list(set(forward.keys()) & set(reverse.keys()))
        pair_d = np.array([min(forward[p], reverse[p]) for p in pairs], float) if pairs else np.array([], float)

        # Mutual can be too harsh: fallback to forward-only if too few pairs
        if len(pairs) < 3:
            pairs = list(forward.keys())
            pair_d = np.array([forward[p] for p in pairs], float) if pairs else np.array([], float)
    else:
        pairs = list(forward.keys())
        pair_d = np.array([forward[p] for p in pairs], float) if pairs else np.array([], float)

    if len(pairs) < 3:
        return None, None, None, []

    # Robust feature distance cap
    if feat_max_dist is None:
        med = _robust_median(pair_d, fallback=float(np.median(pair_d)) if pair_d.size else 1.0, positive_only=True)
        mad = _robust_mad(pair_d, fallback=1.0)
        feat_max_dist = float(med + 6.0 * mad)

    keep = np.isfinite(pair_d) & (pair_d <= float(feat_max_dist))
    pairs = [p for p, kk in zip(pairs, keep) if kk]
    pair_d = pair_d[keep]

    if len(pairs) < 3:
        return None, None, None, []

    # Cap candidate pairs for speed (keep best distances)
    if len(pairs) > int(max_candidate_pairs):
        order = np.argsort(pair_d)[:int(max_candidate_pairs)]
        pairs = [pairs[int(h)] for h in order]
        pair_d = pair_d[order]

    # Build patch->allowed full list (store as small numpy arrays for fast distance checks)
    cand_map_patch = {}
    for (i, j) in pairs:
        cand_map_patch.setdefault(int(i), []).append(int(j))
    for i in list(cand_map_patch.keys()):
        cand_map_patch[i] = np.asarray(sorted(set(cand_map_patch[i])), dtype=int)
    
    # effective "usable" patch points are those with at least one allowed candidate
    if require_feat_consistency:
        eligible_patch = np.asarray(sorted(cand_map_patch.keys()), dtype=int)
    else:
        eligible_patch = np.arange(Np, dtype=int)

    if eligible_patch.size < 3:
        return None, None, None, []

    # Compute effective min_inliers based on eligible_patch size, not raw Np
    # Interpret user-provided min_inliers as an "absolute request" plus a gentle fraction:
    min_inliers_abs = int(max(3, min_inliers))
    min_inliers_frac = float(min_inliers_abs) / float(max(1, eligible_patch.size))
    min_inliers_eff = compute_min_inliers_stable(
        eligible_patch.size,
        min_inliers_abs=min_inliers_abs,
        min_inliers_frac=min_inliers_frac,
        hard_floor=3,
        cap_frac=float(min_inliers_cap_frac),
    )

    # Spatial tree (only needed for require_feat_consistency=False)
    tree_full_spatial = cKDTree(full_pts_um)

    # inlier radius floor based on typical NN spacing
    med_nn_full = _estimate_full_median_nn(full_pts_um)
    inlier_radius_eff = float(inlier_radius_um)
    floor_r = float(min_inlier_radius_frac_nn) * float(med_nn_full)
    if np.isfinite(floor_r) and floor_r > 0:
        inlier_radius_eff = max(inlier_radius_eff, floor_r)

    # Triangle degeneracy floor (make it scale-aware for stability)
    # _triangle_area2 returns 2*area in µm^2; typical triangle area ~ O(spacing^2)
    area_floor = float(min_triangle_area_um2)
    if np.isfinite(med_nn_full) and med_nn_full > 0:
        area_floor = max(area_floor, float((0.02 * med_nn_full) ** 2))

    best_scale = best_R = best_t = None
    best_inliers_patch = np.array([], dtype=int)
    best_inliers_full  = np.array([], dtype=int)
    best_score = -1
    best_mean_err = np.inf

    # PROSAC-like ordering
    order_cand = np.argsort(pair_d)
    pairs_sorted = [pairs[int(h)] for h in order_cand]
    n_cand = len(pairs_sorted)

    # Pretest: sample ONLY from eligible patch points (much fewer false negatives)
    preN = int(min(max(0, int(pretest_n)), eligible_patch.size))
    pre_idx = rng.choice(eligible_patch, size=preN, replace=False) if preN >= 3 else None

    k_spatial = int(max(1, min(int(k_spatial), Nf)))

    def _score_hypothesis(scale, R, t, *, subset_idx=None):
        """
        Returns (inlier_patch_idx, inlier_full_idx, inlier_dist).
        """
        idx_use = subset_idx if subset_idx is not None else eligible_patch
        pts_use = patch_pts_um[idx_use]
        pts_trans = float(scale) * (pts_use @ np.asarray(R, float).T) + np.asarray(t, float)

        in_p, in_f, in_d = [], [], []

        if require_feat_consistency:
            # For each patch point, test only its allowed full indices (usually 1..3)
            for row, p_i in enumerate(idx_use):
                allowed = cand_map_patch.get(int(p_i), None)
                if allowed is None or allowed.size == 0:
                    continue
                diffs = full_pts_um[allowed] - pts_trans[row]
                d2 = np.einsum("ij,ij->i", diffs, diffs)
                jbest = int(np.argmin(d2))
                dd = float(np.sqrt(float(d2[jbest])))
                if np.isfinite(dd) and dd <= float(inlier_radius_eff):
                    in_p.append(int(p_i))
                    in_f.append(int(allowed[jbest]))
                    in_d.append(dd)
        else:
            # Pure spatial kNN
            dists, idxs = tree_full_spatial.query(
                pts_trans,
                k=k_spatial,
                distance_upper_bound=float(inlier_radius_eff),
            )
            dists = np.atleast_2d(dists)
            idxs  = np.atleast_2d(idxs)
            for row, p_i in enumerate(idx_use):
                chosen = None
                for dd, jj in zip(dists[row], idxs[row]):
                    if (not np.isfinite(dd)) or int(jj) >= Nf:
                        continue
                    chosen = (int(p_i), int(jj), float(dd))
                    break
                if chosen is not None:
                    in_p.append(chosen[0]); in_f.append(chosen[1]); in_d.append(chosen[2])

        if len(in_p) == 0:
            return np.array([], int), np.array([], int), np.array([], float)

        in_p = np.asarray(in_p, int)
        in_f = np.asarray(in_f, int)
        in_d = np.asarray(in_d, float)

        if enforce_unique_full_matches:
            # keep closest patch for each full index (stable tie-break)
            best_for_full = {}
            for p_i, f_j, dd in zip(in_p, in_f, in_d):
                prev = best_for_full.get(int(f_j), None)
                if prev is None or dd < prev[2]:
                    best_for_full[int(f_j)] = (int(p_i), int(f_j), float(dd))
            vals = list(best_for_full.values())
            in_p = np.asarray([v[0] for v in vals], int)
            in_f = np.asarray([v[1] for v in vals], int)
            in_d = np.asarray([v[2] for v in vals], float)

        return in_p, in_f, in_d

    # -----------------
    # Main RANSAC loop
    # -----------------
    for it in range(int(n_iters)):
        if prosac:
            base = max(80, int(0.10 * n_cand))
            pool = min(n_cand, base + int((n_cand - base) * (it / max(1, n_iters - 1))))
            pool = max(pool, 3)
            cand_pool = pairs_sorted[:pool]
        else:
            cand_pool = pairs_sorted

        pick = rng.choice(len(cand_pool), size=3, replace=False)
        i_list = [cand_pool[h][0] for h in pick]
        j_list = [cand_pool[h][1] for h in pick]
        if len(set(i_list)) < 3 or len(set(j_list)) < 3:
            continue

        src = patch_pts_um[i_list]
        dst = full_pts_um[j_list]

        if _triangle_area2(src[0], src[1], src[2]) < area_floor:
            continue
        if _triangle_area2(dst[0], dst[1], dst[2]) < area_floor:
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

        # Pretest: relaxed threshold to reduce false negatives
        if pre_idx is not None:
            in_p_s, _, _ = _score_hypothesis(scale, R, t, subset_idx=pre_idx)
            target_frac = float(min_inliers_eff) / float(max(1, eligible_patch.size))
            need = max(3, int(np.ceil(float(pretest_relax) * target_frac * len(pre_idx))))
            if len(in_p_s) < need:
                continue

        in_p, in_f, in_d = _score_hypothesis(scale, R, t, subset_idx=None)
        score = int(len(in_p))
        if score == 0:
            continue

        mean_err = float(np.mean(in_d)) if in_d.size else np.inf

        if (score > best_score) or (score == best_score and mean_err < best_mean_err):
            best_score = score
            best_mean_err = mean_err
            best_scale, best_R, best_t = float(scale), np.asarray(R, float), np.asarray(t, float)
            best_inliers_patch = in_p
            best_inliers_full  = in_f

            # Early stop when very good relative to eligible points
            if best_score >= max(int(min_inliers_eff), int(0.95 * eligible_patch.size)):
                break
            
            # --- ADDED: Check explicit early stop ---
            if best_score >= early_stop_inliers:
                break

    # -----------------
    # Stable return policy
    # -----------------
    if best_scale is None or best_inliers_patch.size < 3:
        return None, None, None, []

    # Refit on inliers (recommended for stability)
    if refit_on_inliers and best_inliers_patch.size >= 3:
        try:
            src_in = patch_pts_um[best_inliers_patch]
            dst_in = full_pts_um[best_inliers_full]
            scale2, R2, t2 = estimate_similarity(src_in, dst_in)
            if float(scale_min) <= float(scale2) <= float(scale_max):
                best_scale, best_R, best_t = float(scale2), np.asarray(R2, float), np.asarray(t2, float)
                # Re-score once after refit to refresh inliers
                in_p2, in_f2, in_d2 = _score_hypothesis(best_scale, best_R, best_t, subset_idx=None)
                if in_p2.size >= best_inliers_patch.size:
                    best_inliers_patch = in_p2
                    best_inliers_full  = in_f2
                    best_score = int(in_p2.size)
        except Exception:
            pass

    # Hard success threshold
    if best_score < int(min_inliers_eff):
        if bool(soft_fail_return_best):
            # Return best anyway; let outer "frac_inliers_thresh" define success.
            return best_scale, best_R, best_t, best_inliers_patch
        else:
            return None, None, None, []

    return best_scale, best_R, best_t, best_inliers_patch


def run_graph_based_matching_um(
    centroids_crop_um,
    centroids_full_um,
    # CHANGED: Accept shapes, not images
    full_shape_px: tuple[int, int],
    patch_shape_px: tuple[int, int],
    
    features_crop,
    features_full,
    pixel_size_full_um,
    pixel_size_patch_um,

    # graph construction + fusion
    k_nn_graph=8,
    k_ngh_feat=6,
    standardize=True,
    w_shape=0.3,
    w_graph=1.0,
    w_triangles=0.7,
    n_triangles=10,

    # RANSAC / correspondences
    n_feat_neighbors=5,
    n_iters=50_000,
    min_inliers=30,
    min_triangle_area_um2=1e-4,
    enforce_unique_full_matches=True,
    feat_ratio=0.85,
    feat_max_dist=None,
    require_mutual=True,
    k_spatial=3,
    require_feat_consistency=True,
    prosac=True,
    pretest_n=200,
    refit_on_inliers=True,
    min_inlier_radius_frac_nn=0.12,
    max_candidate_pairs=30_000,

    # common
    inlier_radius_um=2.0,
    scale_min=0.5,
    scale_max=2.0,
    angle_max_deg=None,
    random_state=42,
    use_icp_refinement=True,
    margin_um=5.0,
    df_full=None,
    df_crop=None,
    use_dynamic_scale=True,
    dynamic_rel_tol=0.1,
    n_candidates_per_patch=3,
    n_candidates_per_full=3,
    pretest_relax=0.8,
    soft_fail_return_best=True,
    min_inliers_cap_frac=0.80,
    
    # NEW: early stop kwargs
    early_stop_frac=None,
    early_stop_inliers=None,
    **graph_kwargs,
):
    """
    End-to-end graph-based matching in µm.
    Returns: (best_scale, best_R, best_t, bbox_full_px)
    """

    centroids_crop_um = _ensure_points_yx(centroids_crop_um, name="centroids_crop_um")
    centroids_full_um = _ensure_points_yx(centroids_full_um, name="centroids_full_um")
    Nc = int(len(centroids_crop_um))

    feats_full, stats_full = build_combined_node_features(
        centroids_full_um,
        features_full,
        k_nn_graph=int(k_nn_graph),
        k_ngh_feat=int(k_ngh_feat),
        standardize=bool(standardize),
        w_shape=float(w_shape),
        w_graph=float(w_graph),
        w_triangles=float(w_triangles),
        n_triangles=int(n_triangles),
        min_triangle_area_um2=float(min_triangle_area_um2),
        ref_stats=None,
        return_stats=True,
    )

    feats_patch = build_combined_node_features(
        centroids_crop_um,
        features_crop,
        k_nn_graph=int(k_nn_graph),
        k_ngh_feat=int(k_ngh_feat),
        standardize=bool(standardize),
        w_shape=float(w_shape),
        w_graph=float(w_graph),
        w_triangles=float(w_triangles),
        n_triangles=int(n_triangles),
        min_triangle_area_um2=float(min_triangle_area_um2),
        ref_stats=stats_full,
        return_stats=False,
    )

    scale_min_eff, scale_max_eff = float(scale_min), float(scale_max)
    if use_dynamic_scale and (df_full is not None) and (df_crop is not None):
        scale_prior, scale_min_eff, scale_max_eff = estimate_dynamic_scale_bounds(
            df_full=df_full,
            df_patch=df_crop,
            pixel_size_full_um=float(pixel_size_full_um),
            pixel_size_patch_um=float(pixel_size_patch_um),
            
            # CHANGED: Use shape arguments, not image objects
            full_shape_px=full_shape_px,
            patch_shape_px=patch_shape_px,
            
            coarse_scale_min=float(scale_min),
            coarse_scale_max=float(scale_max),
            rel_tol=float(dynamic_rel_tol),
        )
        print(f"[scale prior] s ≈ {scale_prior:.3f}, effective range = [{scale_min_eff:.3f}, {scale_max_eff:.3f}]")

    # Early stop setup
    early_stop_target = None
    if early_stop_inliers is not None:
        early_stop_target = int(early_stop_inliers)
    elif early_stop_frac is not None:
        early_stop_target = int(np.ceil(float(early_stop_frac) * max(1, Nc)))

    best_scale, best_R, best_t, inliers = graph_match_similarity(
        centroids_crop_um,
        centroids_full_um,
        feats_patch,
        feats_full,
        n_feat_neighbors=int(n_feat_neighbors),
        inlier_radius_um=float(inlier_radius_um),
        n_iters=int(n_iters),
        scale_min=float(scale_min_eff),
        scale_max=float(scale_max_eff),
        angle_max_deg=angle_max_deg,
        min_inliers=int(min_inliers),
        random_state=None if random_state is None else int(random_state),

        min_triangle_area_um2=float(min_triangle_area_um2),
        enforce_unique_full_matches=bool(enforce_unique_full_matches),
        feat_ratio=float(feat_ratio),
        feat_max_dist=feat_max_dist,
        require_mutual=bool(require_mutual),
        k_spatial=int(k_spatial),
        require_feat_consistency=bool(require_feat_consistency),
        prosac=bool(prosac),
        pretest_n=int(pretest_n),
        refit_on_inliers=bool(refit_on_inliers),
        min_inlier_radius_frac_nn=float(min_inlier_radius_frac_nn),
        max_candidate_pairs=int(max_candidate_pairs),
        n_candidates_per_patch=int(n_candidates_per_patch),
        n_candidates_per_full=int(n_candidates_per_full),
        pretest_relax=float(pretest_relax),
        soft_fail_return_best=bool(soft_fail_return_best),
        min_inliers_cap_frac=float(min_inliers_cap_frac),
        early_stop_inliers=early_stop_target, # Passed down
    )

    if best_scale is None:
        print("Graph-based matching failed.")
        return None, None, None, None

    print("RANSAC (µm): scale =", best_scale)
    print("RANSAC (µm): R =\n", best_R)
    print("RANSAC (µm): t (dy,dx) [µm] =", best_t)
    print("RANSAC: # inliers =", len(inliers))

    if use_icp_refinement:
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
        print("\nICP refined (µm): scale =", best_scale)
        print("ICP refined (µm): R =\n", best_R)
        print("ICP refined (µm): t (dy,dx) [µm] =", best_t)
    
    Hf, Wf = full_shape_px[:2]
    Hc, Wc = patch_shape_px[:2]

    bbox_px = bbox_full_px_from_similarity_um(
        crop_shape_px=(Hc, Wc),
        pixel_size_full_um=float(pixel_size_full_um),
        pixel_size_crop_um=float(pixel_size_patch_um),
        scale=float(best_scale),
        R_yx=np.asarray(best_R),
        t_um_yx=np.asarray(best_t),
        margin_um=float(margin_um),
        full_shape_px=(Hf, Wf),
    )

    return best_scale, best_R, best_t, bbox_px
