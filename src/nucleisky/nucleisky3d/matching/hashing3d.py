"""hashing3d.py Geometric hashing matcher for 3D datasets."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from ..utils import compute_min_inliers_stable
from .geometry import (
    apply_similarity_3d,
    bbox_full_px_from_similarity_um_3d,
    estimate_similarity_3d,
    estimate_dynamic_scale_bounds_3d,
    icp_similarity_3d,
    rotation_angle_deg_3d,
)




@njit(fastmath=True)
def _build_local_frame_core(a, b, c, min_height_ratio, eps):
    ab = b - a
    d = float(np.linalg.norm(ab))
    if (not np.isfinite(d)) or d <= eps:
        return np.eye(3, dtype=np.float64), 0.0, False

    ex = ab / d
    ac = c - a
    ez = np.cross(ex, ac)
    nz = float(np.linalg.norm(ez))
    if (not np.isfinite(nz)) or nz < (min_height_ratio * d) or nz <= eps:
        return np.eye(3, dtype=np.float64), 0.0, False

    ez = ez / nz
    ey = np.cross(ez, ex)
    B = np.empty((3, 3), dtype=np.float64)
    B[0, :] = ex
    B[1, :] = ey
    B[2, :] = ez
    return B, d, True

def _ensure_points_zyx(points_um, name="points_um"):
    pts = np.asarray(points_um, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3). Got {pts.shape}.")
    if not np.isfinite(pts).all():
        raise ValueError(f"{name} contains non-finite values.")
    return pts


def _build_local_frame(a, b, c, min_height_ratio=0.1, eps=1e-8):
    """
    Builds a robust 3D local coordinate frame from 3 points.
    min_height_ratio ensures C is not dangerously collinear with A->B.
    Returns (B, d, is_valid).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    B, d, is_valid = _build_local_frame_core(a, b, c, float(min_height_ratio), float(eps))
    return B, float(d), bool(is_valid)


def geometric_hashing_match_similarity_3d(
    patch_pts_um,
    full_pts_um,
    base_distance_um=10.0,
    bin_size_xyz=0.15,
    vote_thresh=3,
    inlier_radius_um=2.0,
    scale_min=0.5,
    scale_max=2.0,
    n_iters=100000,
    min_inliers_abs=5,
    min_inliers_frac=0.12,
    min_inliers_hard_floor=3,
    min_inliers_cap_frac=0.80,
    random_state=None,
    max_neighbors_full=40,
    max_pairs_per_anchor=20,
    max_k_per_pair=16,
    max_l_per_base=20,
    max_candidates_per_bin=300,
    max_neighbors_patch=40,
    max_pairs_per_anchor_patch=20,
    max_k_per_pair_patch=16,
    max_l_per_base_patch=20,
    neighbor_bin_radius=1,
    max_candidates_test=120,
    randomize_candidates=False,
    pretest_n=80,
    early_stop_frac=1.0,
    min_height_ratio=0.1,  # Added stability parameter
    angle_max_deg=None,
):
    rng = np.random.default_rng(random_state)

    patch_pts_um = _ensure_points_zyx(patch_pts_um, name="patch_pts_um")
    full_pts_um = _ensure_points_zyx(full_pts_um, name="full_pts_um")

    Np = len(patch_pts_um)
    Nf = len(full_pts_um)
    if Np < 4 or Nf < 4:
        return None, None, None, np.array([], dtype=int)

    base = float(base_distance_um)
    if base <= 0:
        raise ValueError("base_distance_um must be > 0.")
    bin_size = float(bin_size_xyz)
    if bin_size <= 0:
        raise ValueError("bin_size_xyz must be > 0.")

    d_patch_lo = (base * 0.5) / float(scale_max)
    d_patch_hi = (base * 1.5) / float(scale_min)

    tree_full = cKDTree(full_pts_um)
    kN_full = min(int(max_neighbors_full) + 1, Nf)
    _, nn_full = tree_full.query(full_pts_um, k=kN_full)
    nn_full = np.atleast_2d(nn_full)[:, 1:]

    hash_table: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int, np.ndarray]]] = defaultdict(list)

    # ==========================================
    # OPTIMIZATION 1: Vectorized Hash Table Build
    # ==========================================
    for i in range(Nf):
        p1 = full_pts_um[i]
        
        js = nn_full[i][: int(max_pairs_per_anchor)].astype(int)
        ks = nn_full[i][: int(max_k_per_pair)].astype(int)
        ls = nn_full[i][: int(max_l_per_base)].astype(int)
        
        V_k = full_pts_um[ks] - p1  # Shape: (K, 3)
        V_l = full_pts_um[ls] - p1  # Shape: (L, 3)

        for j in js:
            if j == i:
                continue
            
            p2 = full_pts_um[j]
            ab = p2 - p1
            d_base = float(np.linalg.norm(ab))
            
            if not np.isfinite(d_base) or d_base <= 0 or not (base * 0.5 < d_base < base * 1.5):
                continue

            ex = ab / d_base
            
            # Vectorized cross product
            ez_all = np.cross(ex, V_k) 
            nz_all = np.linalg.norm(ez_all, axis=1)
            
            # STABILITY FIX: Reject geometries where point K is dangerously collinear with line I->J
            min_height = max(1e-8, float(min_height_ratio) * d_base)
            valid_k = nz_all > min_height

            for k_idx in np.where(valid_k)[0]:
                k = ks[k_idx]
                if k in (i, j):
                    continue

                # Build robust local frame B
                ez = ez_all[k_idx] / nz_all[k_idx]
                ey = np.cross(ez, ex)
                B = np.stack([ex, ey, ez], axis=0) 

                # Vectorized projection of ALL L neighbors instantly
                rels = V_l @ B.T 
                rel_norms = rels / d_base
                bins = np.floor(rel_norms / bin_size).astype(int)

                # Populate hash table
                for l_idx, l in enumerate(ls):
                    if l in (i, j, k):
                        continue
                        
                    key = (bins[l_idx, 0], bins[l_idx, 1], bins[l_idx, 2])
                    bucket = hash_table[key]
                    if len(bucket) < int(max_candidates_per_bin):
                        bucket.append((i, j, k, l, rel_norms[l_idx].astype(np.float32)))

    # ==========================================
    # End Hash Table Build
    # ==========================================

    tree_patch = cKDTree(patch_pts_um)
    kN_patch = min(int(max_neighbors_patch) + 1, Np)
    _, nn_patch = tree_patch.query(patch_pts_um, k=kN_patch)
    nn_patch = np.atleast_2d(nn_patch)[:, 1:]

    best_score, best_scale, best_R, best_t = 0, None, None, None
    best_inliers = np.array([], dtype=int)
    min_inliers_eff = compute_min_inliers_stable(
        Np,
        min_inliers_abs=int(min_inliers_abs),
        min_inliers_frac=float(min_inliers_frac),
        hard_floor=int(min_inliers_hard_floor),
        cap_frac=float(min_inliers_cap_frac),
    )

    early_stop_inliers = int(np.ceil(float(early_stop_frac) * Np))
    early_stop_inliers = max(int(min_inliers_eff), min(early_stop_inliers, Np))

    if int(pretest_n) > 0 and Np > int(pretest_n):
        pre_idx = rng.choice(Np, size=int(pretest_n), replace=False)
    else:
        pre_idx = None

    def _score_inliers(pts_trans):
        dists, _ = tree_full.query(pts_trans, distance_upper_bound=float(inlier_radius_um))
        return np.where(np.isfinite(dists))[0]

    for _ in range(int(n_iters)):
        i = int(rng.integers(Np))
        neighs_i = nn_patch[i][: int(max_pairs_per_anchor_patch)]
        if len(neighs_i) < 3:
            continue

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

        k = None
        neighs_k = nn_patch[i][: int(max_k_per_pair_patch)]
        for _try in range(8):
            kk = int(neighs_k[int(rng.integers(len(neighs_k)))])
            if kk in (i, j):
                continue
            B, _, is_valid = _build_local_frame(patch_pts_um[i], patch_pts_um[j], patch_pts_um[kk], min_height_ratio=min_height_ratio)
            if is_valid:
                k = kk
                break
        if k is None:
            continue

        l = None
        neighs_l = nn_patch[i][: int(max_l_per_base_patch)]
        for _try in range(8):
            ll = int(neighs_l[int(rng.integers(len(neighs_l)))])
            if ll not in (i, j, k):
                l = ll
                break
        if l is None:
            continue

        cp1, cp2, cp3, cp4 = (
            patch_pts_um[i],
            patch_pts_um[j],
            patch_pts_um[k],
            patch_pts_um[l],
        )
        B, d_base, is_valid = _build_local_frame(cp1, cp2, cp3, min_height_ratio=min_height_ratio)
        if not is_valid:
            continue

        rel = B @ (cp4 - cp1)
        rel_norm_p = rel / d_base
        if not np.isfinite(rel_norm_p).all():
            continue

        x_bin0 = int(np.floor(rel_norm_p[0] / bin_size))
        y_bin0 = int(np.floor(rel_norm_p[1] / bin_size))
        z_bin0 = int(np.floor(rel_norm_p[2] / bin_size))

        cand = []
        rb = int(neighbor_bin_radius)
        for dx in range(-rb, rb + 1):
            for dy in range(-rb, rb + 1):
                for dz in range(-rb, rb + 1):
                    key = (x_bin0 + dx, y_bin0 + dy, z_bin0 + dz)
                    if key in hash_table:
                        cand.extend(hash_table[key])

        if not cand:
            continue
        if int(vote_thresh) > 1 and len(cand) < int(vote_thresh):
            continue

        # ==========================================
        # OPTIMIZATION 2: Vectorized Candidate Scoring
        # ==========================================
        if randomize_candidates and len(cand) > int(max_candidates_test):
            idx = rng.choice(len(cand), size=int(max_candidates_test), replace=False)
            cand_sel = [cand[t] for t in idx]
        else:
            cand_rels = np.array([c[4] for c in cand])
            dists = np.linalg.norm(cand_rels - rel_norm_p, axis=1)
            scores = dists / max(bin_size, 1e-8)
            
            K_sel = min(int(max_candidates_test), len(cand))
            if K_sel < len(cand):
                keep = np.argpartition(scores, K_sel - 1)[:K_sel]
                cand_sel = [cand[t] for t in keep]
            else:
                cand_sel = cand
        # ==========================================

        src = np.stack([cp1, cp2, cp3, cp4], axis=0)

        for fi, fj, fk, fl, _ in cand_sel:
            dst = np.stack(
                [
                    full_pts_um[int(fi)],
                    full_pts_um[int(fj)],
                    full_pts_um[int(fk)],
                    full_pts_um[int(fl)],
                ],
                axis=0,
            )

            try:
                scale, R, t = estimate_similarity_3d(src, dst)
            except Exception:
                continue

            scale = float(scale)
            if not (float(scale_min) <= scale <= float(scale_max)):
                continue

            if angle_max_deg is not None:
                if rotation_angle_deg_3d(R) > float(angle_max_deg):
                    continue

            if pre_idx is not None:
                pts_trans_pre = apply_similarity_3d(patch_pts_um[pre_idx], scale, R, t)
                inl_pre = _score_inliers(pts_trans_pre)
                ub = int(len(inl_pre)) + (Np - int(pretest_n))
                if ub <= best_score:
                    continue

            pts_trans = apply_similarity_3d(patch_pts_um, scale, R, t)
            inliers = _score_inliers(pts_trans)
            score = int(len(inliers))

            if score > best_score:
                best_score = score
                best_scale, best_R, best_t = scale, np.asarray(R, float), np.asarray(t, float)
                best_inliers = inliers

                if best_score >= early_stop_inliers:
                    break

        if best_score >= early_stop_inliers:
            break

    if best_score < int(min_inliers_eff):
        return None, None, None, np.array([], dtype=int)

    return best_scale, best_R, best_t, best_inliers


def run_geometric_hashing_matching_3d_um(
    centroids_crop_um,
    centroids_full_um,
    full_shape_px,
    patch_shape_px,
    pixel_size_full_um_zyx,
    pixel_size_patch_um_zyx,
    inlier_radius_um=2.0,
    scale_min=0.8,
    scale_max=1.2,
    random_state=42,
    margin_um=5.0,
    df_full=None,
    df_crop=None,
    use_dynamic_scale=False,
    dynamic_rel_tol=0.1,
    **hash_kwargs,
):
    def _kw(key, default, *aliases):
        for name in (key, *aliases):
            if name in hash_kwargs:
                return hash_kwargs[name]
        return default

    centroids_crop_um = _ensure_points_zyx(centroids_crop_um, name="centroids_crop_um")
    centroids_full_um = _ensure_points_zyx(centroids_full_um, name="centroids_full_um")

    scale_min_eff, scale_max_eff = float(scale_min), float(scale_max)
    if use_dynamic_scale and df_full is not None and df_crop is not None:
        scale_prior, scale_min_eff, scale_max_eff = estimate_dynamic_scale_bounds_3d(
            df_full=df_full,
            df_crop=df_crop,
            voxel_size_full_um_zyx=pixel_size_full_um_zyx,
            voxel_size_crop_um_zyx=pixel_size_patch_um_zyx,
            full_shape_px_zyx=full_shape_px,
            crop_shape_px_zyx=patch_shape_px,
            coarse_scale_min=float(scale_min),
            coarse_scale_max=float(scale_max),
            rel_tol=float(dynamic_rel_tol),
        )
        print(
            f"[hashing3d / scale prior] s ≈ {scale_prior:.3f}, "
            f"effective range = [{scale_min_eff:.3f}, {scale_max_eff:.3f}]"
        )

    scale, R, t, inliers = geometric_hashing_match_similarity_3d(
        patch_pts_um=centroids_crop_um,
        full_pts_um=centroids_full_um,
        base_distance_um=float(_kw("base_distance_um", 10.0)),
        bin_size_xyz=float(_kw("bin_size_xyz", 0.15, "bin_size")),
        vote_thresh=int(_kw("vote_thresh", 3)),
        inlier_radius_um=float(inlier_radius_um),
        scale_min=float(scale_min_eff),
        scale_max=float(scale_max_eff),
        n_iters=int(_kw("n_iters", 50_000)),
        min_inliers_abs=int(_kw("min_inliers_abs", _kw("min_inliers", 20))),
        min_inliers_frac=float(_kw("min_inliers_frac", 0.12)),
        min_inliers_hard_floor=int(_kw("min_inliers_hard_floor", 3)),
        min_inliers_cap_frac=float(_kw("min_inliers_cap_frac", 0.80)),
        random_state=random_state,
        max_neighbors_full=int(_kw("max_neighbors_full", 40, "max_neighbors_ref", "max_neighbors")),
        max_pairs_per_anchor=int(_kw("max_pairs_per_anchor", 20, "max_pairs_per_anchor_full")),
        max_k_per_pair=int(_kw("max_k_per_pair", 16, "max_k_per_pair_full")),
        max_l_per_base=int(_kw("max_l_per_base", 20, "max_l_per_base_full")),
        max_candidates_per_bin=int(_kw("max_candidates_per_bin", 300)),
        max_neighbors_patch=int(_kw("max_neighbors_patch", 40, "max_neighbors_crop")),
        max_pairs_per_anchor_patch=int(_kw("max_pairs_per_anchor_patch", 20, "max_pairs_per_anchor_crop")),
        max_k_per_pair_patch=int(_kw("max_k_per_pair_patch", 16, "max_k_per_pair_crop")),
        max_l_per_base_patch=int(_kw("max_l_per_base_patch", 20, "max_l_per_base_crop")),
        neighbor_bin_radius=int(_kw("neighbor_bin_radius", 1)),
        max_candidates_test=int(_kw("max_candidates_test", 120)),
        randomize_candidates=bool(_kw("randomize_candidates", False)),
        pretest_n=int(_kw("pretest_n", 80)),
        early_stop_frac=float(_kw("early_stop_frac", 1.0)),
        min_height_ratio=float(_kw("min_height_ratio", 0.1)),
        angle_max_deg=_kw("angle_max_deg", None),
    )

    if scale is None:
        print("Geometric hashing 3D matcher failed.")
        return None, None, None, None

    print("HASHING 3D RANSAC (µm): scale =", scale)
    print("HASHING 3D RANSAC (µm): R =\n", R)
    print("HASHING 3D RANSAC (µm): t (dz,dy,dx) [µm] =", t)
    print("HASHING 3D RANSAC: # inliers =", len(inliers))

    if bool(_kw("use_icp_refinement", True)):
        ref_scale, ref_R, ref_t = icp_similarity_3d(
            centroids_crop_um,
            centroids_full_um,
            scale,
            R,
            t,
            n_iters=int(_kw("icp_iters", 10)),
            inlier_radius_um=float(inlier_radius_um),
        )
        scale, R, t = ref_scale, ref_R, ref_t
        print("\nHASHING 3D ICP refined (µm): scale =", scale)
        print("HASHING 3D ICP refined (µm): R =\n", R)
        print("HASHING 3D ICP refined (µm): t (dz,dy,dx) [µm] =", t)

    angle_max_deg = _kw("angle_max_deg", None)
    if angle_max_deg is not None and rotation_angle_deg_3d(R) > float(angle_max_deg):
        print("Geometric hashing 3D result rejected: rotation angle exceeds angle_max_deg.")
        return None, None, None, None

    Zf, Yf, Xf = full_shape_px[:3]
    Zc, Yc, Xc = patch_shape_px[:3]

    bbox = bbox_full_px_from_similarity_um_3d(
        crop_shape_px=(Zc, Yc, Xc),
        pixel_size_full_um_zyx=pixel_size_full_um_zyx,
        pixel_size_crop_um_zyx=pixel_size_patch_um_zyx,
        scale=float(scale),
        R_zyx=np.asarray(R),
        t_um_zyx=np.asarray(t),
        margin_um=float(margin_um),
        full_shape_px=(Zf, Yf, Xf),
    )

    return scale, R, t, bbox
