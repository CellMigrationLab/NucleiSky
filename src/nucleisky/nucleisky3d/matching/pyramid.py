"""pyramid.py Helper utilities for tetrahedron-based 3D matching."""

from __future__ import annotations

import itertools
import networkx as nx
import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from .geometry import (
    apply_similarity_3d,
    bbox_full_px_from_similarity_um_3d,
    estimate_dynamic_scale_bounds_3d,
    estimate_similarity_3d,
    icp_similarity_3d,
    rotation_angle_deg_3d,
)
from ..utils import compute_min_inliers_stable  




@njit(fastmath=True)
def _tetrahedron_volume_core(p0, p1, p2, p3):
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    vol6 = float(np.dot(v1, np.cross(v2, v3)))
    if vol6 < 0.0:
        vol6 = -vol6
    return vol6 / 6.0


@njit(fastmath=True)
def _tetrahedron_descriptor_core(p0, p1, p2, p3, eps):
    edge_lengths = np.empty(6, dtype=np.float64)
    edge_lengths[0] = np.linalg.norm(p0 - p1)
    edge_lengths[1] = np.linalg.norm(p0 - p2)
    edge_lengths[2] = np.linalg.norm(p0 - p3)
    edge_lengths[3] = np.linalg.norm(p1 - p2)
    edge_lengths[4] = np.linalg.norm(p1 - p3)
    edge_lengths[5] = np.linalg.norm(p2 - p3)

    for i in range(6):
        if (not np.isfinite(edge_lengths[i])) or edge_lengths[i] <= eps:
            return np.zeros(7, dtype=np.float32), False

    volume = _tetrahedron_volume_core(p0, p1, p2, p3)
    if (not np.isfinite(volume)) or volume <= eps:
        return np.zeros(7, dtype=np.float32), False

    sorted_edges = np.sort(edge_lengths)
    mean_edge = float(np.mean(sorted_edges))
    if (not np.isfinite(mean_edge)) or mean_edge <= eps:
        return np.zeros(7, dtype=np.float32), False

    out = np.empty(7, dtype=np.float32)
    out[0] = np.float32((volume / (mean_edge ** 3 + eps)) * 15.0)
    for i in range(6):
        out[i + 1] = np.float32(sorted_edges[i] / (mean_edge + eps))
    return out, True


@njit(fastmath=True)
def _is_degenerate_tetrahedron_core(p0, p1, p2, p3, min_volume_ratio, max_aspect_ratio, eps):
    edge_lengths = np.empty(6, dtype=np.float64)
    edge_lengths[0] = np.linalg.norm(p0 - p1)
    edge_lengths[1] = np.linalg.norm(p0 - p2)
    edge_lengths[2] = np.linalg.norm(p0 - p3)
    edge_lengths[3] = np.linalg.norm(p1 - p2)
    edge_lengths[4] = np.linalg.norm(p1 - p3)
    edge_lengths[5] = np.linalg.norm(p2 - p3)

    for i in range(6):
        if (not np.isfinite(edge_lengths[i])) or edge_lengths[i] <= eps:
            return True

    min_edge = float(np.min(edge_lengths))
    max_edge = float(np.max(edge_lengths))
    if min_edge <= eps or (max_edge / min_edge) > max_aspect_ratio:
        return True

    mean_edge = float(np.mean(edge_lengths))
    volume = _tetrahedron_volume_core(p0, p1, p2, p3)
    volume_ratio = volume / (mean_edge ** 3 + eps)
    return (not np.isfinite(volume_ratio)) or volume_ratio <= min_volume_ratio

def _ensure_points_zyx(points_um, name="points_um"):
    pts = np.asarray(points_um, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3). Got {pts.shape}.")
    if not np.isfinite(pts).all():
        raise ValueError(f"{name} contains non-finite values.")
    return pts


def build_geometric_knn_graph(points_um, k=10, min_edge_dist=1e-6):
    pts = _ensure_points_zyx(points_um, name="points_um")
    N = int(len(pts))

    G = nx.Graph()
    for i, (z, y, x) in enumerate(pts):
        G.add_node(i, pos=(float(z), float(y), float(x)))

    if N == 0:
        return G

    kk = max(1, min(int(k) + 1, N))
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=kk)
    
    # FIX: Safely reshape to guarantee 2D indexing, np.atleast_2d fails on N=1 edge cases
    dists = dists.reshape(N, -1)
    idxs = idxs.reshape(N, -1)

    for i in range(N):
        for d, j in zip(dists[i][1:], idxs[i][1:]):  # skip self
            j = int(j)
            if i == j:
                continue
            if (not np.isfinite(d)) or float(d) < float(min_edge_dist):
                continue
            if G.has_edge(i, j):
                continue
            delta = pts[j] - pts[i]
            G.add_edge(i, j, dist=float(d), delta=delta.astype(np.float32))

    for i in G.nodes:
        neigh_sorted = sorted(G.neighbors(i), key=lambda j: G[i][j].get("dist", np.inf))
        G.nodes[i]["neighbors"] = list(neigh_sorted)

    return G


def _tetrahedron_volume(p0, p1, p2, p3):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    return float(_tetrahedron_volume_core(p0, p1, p2, p3))


def _tetrahedron_descriptor(p0, p1, p2, p3, eps=1e-12):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    desc, is_valid = _tetrahedron_descriptor_core(p0, p1, p2, p3, float(eps))
    if not bool(is_valid):
        return None
    return desc


def _is_degenerate_tetrahedron(p0, p1, p2, p3, *, min_volume_ratio=1e-4, max_aspect_ratio=50.0, eps=1e-12):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    return bool(
        _is_degenerate_tetrahedron_core(
            p0, p1, p2, p3, float(min_volume_ratio), float(max_aspect_ratio), float(eps)
        )
    )


def build_tetrahedron_node_features(pts, graph, n_tetrahedra):
    pts = _ensure_points_zyx(pts, name="pts")
    N = int(len(pts))
    # FIX: Initialize with NaNs to explicitly track nodes that fail to form tetrahedra
    tetra_feats = np.full((N, 7), np.nan, dtype=np.float32)
    if N == 0 or int(n_tetrahedra) <= 0:
        return tetra_feats

    n_tetrahedra = int(n_tetrahedra)

    for i in graph.nodes:
        neighs = graph.nodes[i].get("neighbors", [])
        if len(neighs) < 3:
            continue
        descs = []
        for combo in itertools.combinations(neighs, 3):
            if len(descs) >= n_tetrahedra:
                break
            p0 = pts[i]
            p1, p2, p3 = pts[combo[0]], pts[combo[1]], pts[combo[2]]
            
            if _is_degenerate_tetrahedron(p0, p1, p2, p3):
                continue
                
            desc = _tetrahedron_descriptor(p0, p1, p2, p3)
            if desc is None:
                continue
            descs.append(desc)
            
        if descs:
            tetra_feats[i] = np.mean(descs, axis=0).astype(np.float32)

    return tetra_feats


def _zscore_with_ref(feat, ref_mu=None, ref_sigma=None, eps=1e-8):
    X = np.asarray(feat, float)
    if X.ndim == 1:
        X = X[:, None]

    # Mask valid rows so we don't skew the mean with NaNs
    valid = np.all(np.isfinite(X), axis=1)

    if ref_mu is None or ref_sigma is None:
        if not np.any(valid):
            mu = np.zeros((1, X.shape[1]))
            sigma = np.ones((1, X.shape[1]))
        else:
            mu = np.mean(X[valid], axis=0, keepdims=True)
            sigma = np.std(X[valid], axis=0, keepdims=True)
    else:
        mu = np.asarray(ref_mu, float)
        sigma = np.asarray(ref_sigma, float)

    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)

    # Leave invalid features as NaN
    Z = np.full_like(X, np.nan, dtype=np.float32)
    Z[valid] = ((X[valid] - mu) / (sigma + float(eps))).astype(np.float32)
    
    return Z, mu, sigma


def _filter_mutual_nearest_neighbors(tetra_crop, tetra_full, k_feat=1):
    """    
    Enforces Mutual Nearest Neighbors (MNN) to drastically reduce false positive matches.
    Only allows a match if Crop Node A thinks Full Node B is closest, AND B thinks A is closest.
    """
    valid_c = np.all(np.isfinite(tetra_crop), axis=1)
    valid_f = np.all(np.isfinite(tetra_full), axis=1)

    idx_c = np.where(valid_c)[0]
    idx_f = np.where(valid_f)[0]

    if len(idx_c) < 4 or len(idx_f) < 4:
        return []

    # Forward query (Crop -> Full)
    tree_f = cKDTree(tetra_full[idx_f])
    dists_c2f, inds_c2f = tree_f.query(tetra_crop[idx_c], k=k_feat)
    if k_feat == 1:
        inds_c2f = inds_c2f.reshape(-1, 1)
        dists_c2f = dists_c2f.reshape(-1, 1)

    # Backward query (Full -> Crop)
    tree_c = cKDTree(tetra_crop[idx_c])
    _, inds_f2c = tree_c.query(tetra_full[idx_f], k=1)

    pairs = []
    for i_enum, c_orig in enumerate(idx_c):
        for k in range(inds_c2f.shape[1]):
            f_enum = inds_c2f[i_enum, k]
            d = dists_c2f[i_enum, k]
            if not np.isfinite(d): 
                continue
            
            # MNN Check
            if inds_f2c[f_enum] == i_enum:
                pairs.append((c_orig, idx_f[f_enum], float(d)))
                
    return pairs


def run_pyramid_based_matching_um(
    centroids_crop_um,
    centroids_full_um,
    *,
    df_full=None,
    df_crop=None,
    voxel_size_full_um_zyx=None,
    voxel_size_crop_um_zyx=None,
    full_shape_px_zyx=None,
    crop_shape_px_zyx=None,
    use_dynamic_scale=False,
    dynamic_rel_tol=0.1,
    inlier_radius_um=2.0,
    scale_min=0.8,
    scale_max=1.2,
    angle_max_deg=None,
    n_tetrahedra=5,
    n_iters=50_000,
    min_inliers=5,
    random_state=42,
    k_nn_tetra=8,
    n_feat_neighbors=1,
    max_candidate_pairs=None,
    early_stop_inliers=None,
    early_stop_frac=None,
    use_icp_refinement=True,
    icp_iters=10,
    margin_um=5.0,
):
    rng = np.random.default_rng(random_state)

    crop_pts_um = _ensure_points_zyx(centroids_crop_um, name="centroids_crop_um")
    full_pts_um = _ensure_points_zyx(centroids_full_um, name="centroids_full_um")

    Nc = int(len(crop_pts_um))
    Nf = int(len(full_pts_um))
    if Nc < 4 or Nf < 4:
        return None, None, None, None

    # Dynamically scale the minimum inliers required based on crop size
    min_inliers_eff = compute_min_inliers_stable(Nc, min_inliers_abs=min_inliers, min_inliers_frac=0.1)

    scale_min_eff, scale_max_eff = float(scale_min), float(scale_max)
    if use_dynamic_scale and df_full is not None and df_crop is not None:
        if (
            voxel_size_full_um_zyx is None
            or voxel_size_crop_um_zyx is None
            or full_shape_px_zyx is None
            or crop_shape_px_zyx is None
        ):
            raise ValueError("Dynamic scale requires voxel sizes and shapes for full/crop volumes.")
            
        scale_prior, scale_min_eff, scale_max_eff = estimate_dynamic_scale_bounds_3d(
            df_full=df_full, df_crop=df_crop,
            voxel_size_full_um_zyx=voxel_size_full_um_zyx, voxel_size_crop_um_zyx=voxel_size_crop_um_zyx,
            full_shape_px_zyx=full_shape_px_zyx, crop_shape_px_zyx=crop_shape_px_zyx,
            coarse_scale_min=float(scale_min), coarse_scale_max=float(scale_max), rel_tol=float(dynamic_rel_tol),
        )

    if early_stop_inliers is None and early_stop_frac is not None:
        early_stop_inliers = int(np.ceil(float(early_stop_frac) * max(1, Nc)))
    if early_stop_inliers is None:
        early_stop_inliers = Nc
    early_stop_inliers = int(max(1, min(int(early_stop_inliers), Nc)))

    k_nn_tetra = int(max(3, k_nn_tetra))
    G_full = build_geometric_knn_graph(full_pts_um, k=k_nn_tetra)
    G_crop = build_geometric_knn_graph(crop_pts_um, k=k_nn_tetra)

    tetra_full = build_tetrahedron_node_features(full_pts_um, G_full, n_tetrahedra=int(n_tetrahedra))
    tetra_crop = build_tetrahedron_node_features(crop_pts_um, G_crop, n_tetrahedra=int(n_tetrahedra))

    tetra_full_z, tetra_mu, tetra_sigma = _zscore_with_ref(tetra_full, ref_mu=None, ref_sigma=None)
    tetra_crop_z, _, _ = _zscore_with_ref(tetra_crop, ref_mu=tetra_mu, ref_sigma=tetra_sigma)
    
    # -------------------------------------------------------------
    # FEATURE MATCHING 
    # -------------------------------------------------------------
    k_feat = int(max(1, min(int(n_feat_neighbors), Nf)))
    pairs = _filter_mutual_nearest_neighbors(tetra_crop_z, tetra_full_z, k_feat=k_feat)

    if len(pairs) < 4:
        print("Not enough valid feature matches to run RANSAC.")
        return None, None, None, None

    if max_candidate_pairs is not None:
        max_candidate_pairs = int(max_candidate_pairs)
        if len(pairs) > max_candidate_pairs:
            pairs.sort(key=lambda x: x[2])
            pairs = pairs[:max_candidate_pairs]

    tree_full_spatial = cKDTree(full_pts_um)

    best_scale = best_R = best_t = None
    best_inliers = np.array([], dtype=int)
    best_score = 0
    
    consecutive_failures = 0
    max_failures = 5000 

    for _ in range(int(n_iters)):
        idxs_h = rng.choice(len(pairs), size=4, replace=False)
        i_list = [pairs[h][0] for h in idxs_h]
        j_list = [pairs[h][1] for h in idxs_h]

        if len(set(i_list)) < 4 or len(set(j_list)) < 4:
            continue

        src = crop_pts_um[i_list]
        dst = full_pts_um[j_list]

        if _is_degenerate_tetrahedron(*src) or _is_degenerate_tetrahedron(*dst):
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                print("RANSAC terminating early: too many degenerate geometries sampled.")
                break
            continue
            
        consecutive_failures = 0

        try:
            scale, R, t = estimate_similarity_3d(src, dst)
        except Exception:
            continue

        if not (float(scale_min_eff) <= float(scale) <= float(scale_max_eff)):
            continue

        if angle_max_deg is not None and rotation_angle_deg_3d(R) > float(angle_max_deg):
            continue

        pts_trans = apply_similarity_3d(crop_pts_um, scale, R, t)
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

    # Now using the dynamically scaled threshold logic
    if best_scale is None or best_score < min_inliers_eff:
        return None, None, None, None

    if bool(use_icp_refinement):
        best_scale, best_R, best_t = icp_similarity_3d(
            crop_pts_um, full_pts_um, best_scale, best_R, best_t,
            n_iters=int(icp_iters), inlier_radius_um=float(inlier_radius_um),
        )

    if angle_max_deg is not None and rotation_angle_deg_3d(best_R) > float(angle_max_deg):
        return None, None, None, None

    bbox = None
    if (
        voxel_size_full_um_zyx is not None
        and voxel_size_crop_um_zyx is not None
        and full_shape_px_zyx is not None
        and crop_shape_px_zyx is not None
    ):
        bbox = bbox_full_px_from_similarity_um_3d(
            crop_shape_px=tuple(crop_shape_px_zyx[:3]),
            pixel_size_full_um_zyx=voxel_size_full_um_zyx,
            pixel_size_crop_um_zyx=voxel_size_crop_um_zyx,
            scale=float(best_scale),
            R_zyx=np.asarray(best_R),
            t_um_zyx=np.asarray(best_t),
            margin_um=float(margin_um),
            full_shape_px=tuple(full_shape_px_zyx[:3]),
        )

    return best_scale, best_R, best_t, bbox
