
"""geometry.py Shared geometry utilities used by multiple matchers."""

import math
from typing import Any, Dict, Optional, Tuple
import networkx as nx

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from ..types import BBox


@njit(fastmath=True)
def _estimate_similarity_2d_core(src, dst):
    src_mean = np.empty(2, dtype=np.float64)
    dst_mean = np.empty(2, dtype=np.float64)
    for c in range(2):
        src_mean[c] = np.mean(src[:, c])
        dst_mean[c] = np.mean(dst[:, c])

    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = float(np.sum(src_c * src_c))
    if (not np.isfinite(var_src)) or var_src < 1e-12:
        return 0.0, np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64), False

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0.0:
        Vt2 = Vt.copy()
        Vt2[-1, :] *= -1.0
        R = Vt2.T @ U.T

    scale = float(np.sum(S)) / var_src
    if (not np.isfinite(scale)) or scale <= 0.0:
        return 0.0, np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64), False

    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t, True


@njit(fastmath=True)
def _apply_similarity_2d_core(pts, scale, R, t):
    return scale * (pts @ R.T) + t


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
def _triangle_descriptor_core(p1, p2, p3, eps):
    v1 = p2 - p1
    v2 = p3 - p1
    v1_norm = float(np.linalg.norm(v1))
    v2_norm = float(np.linalg.norm(v2))
    if v1_norm < eps or v2_norm < eps:
        return np.zeros(2, dtype=np.float32), False

    vb = float(np.dot(v1, v2) / (v2_norm * v2_norm + eps))
    a = float(v1_norm / (v2_norm + eps))
    cos_th = vb / (a + eps)
    if cos_th < -1.0:
        cos_th = -1.0
    elif cos_th > 1.0:
        cos_th = 1.0
    inside = 1.0 - cos_th * cos_th
    if inside < 0.0:
        inside = 0.0
    vh = float(np.sqrt(inside) * a)

    cross = v1[0] * v2[1] - v1[1] * v2[0]
    out = np.empty(2, dtype=np.float32)
    out[0] = vb
    out[1] = -vh if cross < 0.0 else vh
    return out, True



def estimate_similarity_2d(src, dst):
    """
    Estimate 2D similarity transform (scale, rotation, translation) mapping src -> dst.

    Inputs:
      src, dst: (N,2) arrays, coordinate order (y,x), same units (e.g., µm)

    Returns:
      scale (float), R (2,2) ndarray, t (2,) ndarray
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError(f"estimate_similarity expects src and dst shaped (N,2). Got src={src.shape}, dst={dst.shape}")

    N = src.shape[0]
    if N < 2:
        raise ValueError("estimate_similarity requires at least 2 points (preferably 3+).")

    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("estimate_similarity received non-finite values in src/dst.")

    scale, R, t, success = _estimate_similarity_2d_core(src, dst)
    if not success:
        raise ValueError("Degenerate source configuration or invalid similarity estimate.")

    return float(scale), np.asarray(R, float), np.asarray(t, float)



def estimate_similarity(src, dst):
    return estimate_similarity_2d(src, dst)


def apply_similarity_2d(pts, scale, R, t):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"apply_similarity_2d expects pts shaped (N,2). Got pts={pts.shape}")

    R = np.asarray(R, dtype=float).reshape(2, 2)
    t = np.asarray(t, dtype=float).reshape(2,)
    s = float(scale)

    if not np.isfinite(s):
        raise ValueError("scale must be finite.")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("R and t must be finite.")

    return _apply_similarity_2d_core(pts, s, R, t)



def icp_similarity(
    src_pts, dst_pts,
    init_scale, init_R, init_t,
    n_iters=10,
    inlier_radius_um=2.0,
):
    """
    Simple ICP refinement for similarity transform in 2D (all in µm).
    src_pts: (N,2) patch points in µm
    dst_pts: (M,2) full points in µm
    """
    src = np.asarray(src_pts, float)
    dst = np.asarray(dst_pts, float)

    scale = float(init_scale)
    R = init_R.copy()
    t = init_t.copy()

    tree_dst = cKDTree(dst)  # distances in µm

    for _ in range(n_iters):
        # Transform src (still in µm)
        src_trans = scale * (src @ R.T) + t

        # Find closest dst (µm)
        dists, idxs = tree_dst.query(src_trans, distance_upper_bound=inlier_radius_um)
        mask = np.isfinite(dists)
        if mask.sum() < 3:
            break

        src_in = src[mask]
        dst_in = dst[idxs[mask]]

        # Re-estimate similarity on inliers
        scale, R, t = estimate_similarity(src_in, dst_in)

    return scale, R, t


def similarity_um_to_affine_px(
    scale: float,
    R_yx,
    t_um_yx,
    *,
    pixel_size_src_um: float,
    pixel_size_dst_um: float,
):
    """
    NucleiSky similarity convention with points stored as row-vectors (y,x):
        dst_um_row = scale * (src_um_row @ R.T) + t_um_row

    Equivalent column-vector form (used by scipy.ndimage affine helpers):
        dst_um_col = scale * (R @ src_um_col) + t_um_col

    Pixel conversion:
        src_um = src_px * pix_src
        dst_px = dst_um / pix_dst

    Therefore in column-vector pixels:
        dst_px_col = A_px @ src_px_col + b_px
    where:
        A_px = (scale * pix_src / pix_dst) * R
        b_px = t_um / pix_dst
    """
    R_yx = np.asarray(R_yx, float).reshape(2, 2)
    t_um_yx = np.asarray(t_um_yx, float).reshape(2,)
    s = float(scale)

    pix_src = float(pixel_size_src_um)
    pix_dst = float(pixel_size_dst_um)
    if pix_src <= 0 or pix_dst <= 0 or (not np.isfinite(pix_src)) or (not np.isfinite(pix_dst)):
        raise ValueError("pixel_size_src_um and pixel_size_dst_um must be positive finite floats.")

    if not np.isfinite(s):
        raise ValueError("scale must be finite.")
    if not np.isfinite(R_yx).all() or not np.isfinite(t_um_yx).all():
        raise ValueError("R_yx and t_um_yx must be finite.")

    A_px = (s * pix_src / pix_dst) * R_yx
    b_px = t_um_yx / pix_dst
    return A_px, b_px


def invert_affine_px(A_px, b_px):
    """
    Forward map: dst = A*src + b  (y,x)
    Return inverse as forward map: src = Ainv*dst + binv
    """
    A = np.asarray(A_px, float).reshape(2, 2)
    b = np.asarray(b_px, float).reshape(2,)
    det = float(np.linalg.det(A))
    if not np.isfinite(det) or abs(det) < 1e-12:
        raise ValueError(f"Affine matrix A is singular/ill-conditioned (det={det}).")
    Ainv = np.linalg.inv(A)
    binv = -Ainv @ b
    return Ainv, binv


def rotation_deg_from_R(R_yx) -> float:
    R_yx = np.asarray(R_yx, float).reshape(2, 2)
    return float(np.degrees(np.arctan2(R_yx[1, 0], R_yx[0, 0])))


def bbox_full_px_from_similarity_um(
    crop_shape_px,
    pixel_size_full_um,
    pixel_size_crop_um,
    scale,
    R_yx,
    t_um_yx,
    margin_um=0.0,
    full_shape_px=None,
) -> BBox:
    """
    Compute a bounding box in FULL pixel coordinates covering the entire transformed crop footprint.

    Canonical return:
        BBox(y0, y1, x0, x1) with half-open bounds:
            y in [y0, y1)
            x in [x0, x1)

    Important:
      - Footprint is computed from pixel-edge corners (extent), not pixel-center corners.
      - Transform convention (y,x), row-vector points:
            full_um = scale * (crop_um @ R.T) + t
    """
    Hc, Wc = map(int, crop_shape_px[:2])
    pixF = float(pixel_size_full_um)
    pixC = float(pixel_size_crop_um)
    s = float(scale)
    R = np.asarray(R_yx, float).reshape(2, 2)
    t = np.asarray(t_um_yx, float).reshape(2,)

    if Hc <= 0 or Wc <= 0:
        raise ValueError(f"crop_shape_px must be positive, got {crop_shape_px}")
    if pixF <= 0 or pixC <= 0 or (not np.isfinite(pixF)) or (not np.isfinite(pixC)):
        raise ValueError("pixel sizes must be positive finite floats (µm/px).")
    if s <= 0 or (not np.isfinite(s)):
        raise ValueError("scale must be positive finite.")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("R_yx and t_um_yx must be finite.")

    # Pixel-edge corners for a half-open crop footprint: [0,Hc) x [0,Wc)
    corners_px = np.array(
        [[0.0, 0.0],
         [float(Hc), 0.0],
         [0.0, float(Wc)],
         [float(Hc), float(Wc)]],
        dtype=float,
    )  # (y,x)

    corners_um = corners_px * pixC

    # Forward map in µm (row-vector points)
    corners_full_um = (s * (corners_um @ R.T)) + t[None, :]

    m = float(margin_um)
    y0_um = float(corners_full_um[:, 0].min()) - m
    y1_um = float(corners_full_um[:, 0].max()) + m
    x0_um = float(corners_full_um[:, 1].min()) - m
    x1_um = float(corners_full_um[:, 1].max()) + m

    # Convert to half-open FULL pixel bounds (no "+1" fudge)
    y0 = int(np.floor(y0_um / pixF))
    y1 = int(np.ceil(y1_um / pixF))
    x0 = int(np.floor(x0_um / pixF))
    x1 = int(np.ceil(x1_um / pixF))

    bb = BBox(y0, y1, x0, x1)

    if full_shape_px is not None:
        Hf, Wf = map(int, full_shape_px[:2])
        bb = bb.clamp((Hf, Wf), min_size=1)

    return bb


def estimate_dynamic_scale_bounds(
    df_full,
    df_patch,
    pixel_size_full_um,
    pixel_size_patch_um,
    full_shape_px,
    patch_shape_px,
    coarse_scale_min=0.5,
    coarse_scale_max=2.0,
    rel_tol=0.1,
):
    """
    Estimate a dynamic scale prior and effective [scale_min, scale_max] using:
      - nearest-neighbour distance (geometry)
      - equivalent nucleus diameter (morphology)
      - nuclei density per µm²

    Returns
    -------
    scale_prior, scale_min_eff, scale_max_eff
    """

    eps = 1e-8
    rel_tol = float(rel_tol)
    rel_tol = max(0.0, min(0.95, rel_tol))

    def _median_nn_all(df):
        cols = [c for c in df.columns if c.startswith("nn") and c.endswith("_dist_um")]
        if not cols:
            return None
        vals = df[cols].to_numpy().ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    # 1) geometric cue
    med_nn_full = _median_nn_all(df_full)
    med_nn_patch = _median_nn_all(df_patch)
    s_nn = None
    if med_nn_full not in (None, 0) and med_nn_patch not in (None, 0):
        s_nn = med_nn_full / (med_nn_patch + eps)

    # 2) morphology cue
    s_size = None
    if "equiv_diameter_um" in df_full.columns and "equiv_diameter_um" in df_patch.columns:
        d_full = df_full["equiv_diameter_um"].to_numpy()
        d_patch = df_patch["equiv_diameter_um"].to_numpy()
        d_full = d_full[np.isfinite(d_full) & (d_full > 0)]
        d_patch = d_patch[np.isfinite(d_patch) & (d_patch > 0)]
        if d_full.size and d_patch.size:
            s_size = float(np.median(d_full)) / (float(np.median(d_patch)) + eps)

    # 3) density cue:
    # Density scales ~ 1/s^2 => s_dens ~ sqrt(dens_full / dens_patch)
    Hf, Wf = map(int, full_shape_px[:2])
    Hp, Wp = map(int, patch_shape_px[:2])

    area_full_um2 = (Hf * float(pixel_size_full_um)) * (Wf * float(pixel_size_full_um))
    area_patch_um2 = (Hp * float(pixel_size_patch_um)) * (Wp * float(pixel_size_patch_um))

    dens_full = len(df_full) / (area_full_um2 + eps)
    dens_patch = len(df_patch) / (area_patch_um2 + eps)

    s_dens = None
    if dens_full > 0 and dens_patch > 0:
        s_dens = float(np.sqrt(dens_full / (dens_patch + eps)))

    # 4) combine robustly
    candidates = [s for s in (s_nn, s_size, s_dens) if s is not None and np.isfinite(s) and 0.1 < s < 10.0]
    scale_prior = float(np.median(candidates)) if candidates else 1.0

    scale_min_eff = max(float(coarse_scale_min), scale_prior * (1.0 - rel_tol))
    scale_max_eff = min(float(coarse_scale_max), scale_prior * (1.0 + rel_tol))

    if not (scale_min_eff < scale_max_eff):
        scale_min_eff = float(coarse_scale_min)
        scale_max_eff = float(coarse_scale_max)

    return scale_prior, scale_min_eff, scale_max_eff


def compute_min_inliers_stable(Nc_eff, min_inliers_abs, min_inliers_frac, *, hard_floor=3, cap_frac=0.80):
    Nc_eff = int(max(0, Nc_eff))
    hard_floor = int(max(3, hard_floor))
    cap_frac = float(np.clip(cap_frac, 0.1, 1.0))
    if Nc_eff <= 0:
        return hard_floor
    base = max(int(min_inliers_abs), int(np.ceil(float(min_inliers_frac) * Nc_eff)))
    cap  = max(hard_floor, int(np.floor(cap_frac * Nc_eff)))
    return int(np.clip(base, hard_floor, cap))


def _ensure_points_yx(pts, name="pts"):
    pts = np.asarray(pts, float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must be (N,2) (y,x). Got {pts.shape}")
    if len(pts) and (not np.isfinite(pts).all()):
        bad = np.where(~np.isfinite(pts).all(axis=1))[0]
        raise ValueError(f"{name} contains non-finite rows at indices: {bad[:20].tolist()}")
    return pts


def _knn_candidates(tree, q, k, keep_k=3, ratio=0.85):
    """
    Return candidate indices for a query point.
    - If the match is distinctive (passes ratio), return only the best.
    - If ambiguous (fails ratio), return the top keep_k instead of returning none.
    """
    dists, idxs = tree.query(q, k=int(k))
    dists = np.atleast_1d(dists).astype(float, copy=False)
    idxs  = np.atleast_1d(idxs).astype(int, copy=False)

    keep = np.isfinite(dists)
    dists = dists[keep]
    idxs  = idxs[keep]
    if dists.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    # ensure sorted
    order = np.argsort(dists)
    dists = dists[order]
    idxs  = idxs[order]

    keep_k = max(1, min(int(keep_k), int(dists.size)))

    if ratio is not None and 0.0 < float(ratio) < 1.0 and dists.size >= 2:
        # distinctive -> keep only best
        if dists[0] <= float(ratio) * dists[1]:
            return dists[:1], idxs[:1]
        # ambiguous -> keep top keep_k rather than dropping all
        return dists[:keep_k], idxs[:keep_k]

    return dists[:keep_k], idxs[:keep_k]


def build_geometric_knn_graph(points_um, k=10, min_edge_dist=1e-6):
    pts = _ensure_points_yx(points_um, name="points_um")
    N = int(len(pts))

    G = nx.Graph()
    for i, (y, x) in enumerate(pts):
        G.add_node(i, pos=(float(y), float(x)))

    if N == 0:
        return G

    kk = max(1, min(int(k) + 1, N))
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=kk)
    dists = np.atleast_2d(dists)
    idxs  = np.atleast_2d(idxs)

    for i in range(N):
        for d, j in zip(dists[i][1:], idxs[i][1:]):  # skip self
            j = int(j)
            if i == j:
                continue
            if (not np.isfinite(d)) or float(d) < float(min_edge_dist):
                continue
            if G.has_edge(i, j):
                continue
            dy, dx = pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1]
            G.add_edge(i, j, dist=float(d), angle=float(np.arctan2(dy, dx)))

    for i in G.nodes:
        neigh_sorted = sorted(G.neighbors(i), key=lambda j: G[i][j].get("dist", np.inf))
        G.nodes[i]["neighbors"] = list(neigh_sorted)

    return G


def _wrap_pi(a):
    return ((a + np.pi) % (2*np.pi)) - np.pi


def _wrap_angle_rad(a):
    return (a + 2.0 * np.pi) % (2.0 * np.pi)


def _circ_diff_rad(a, b):
    # shortest circular difference in radians
    d = abs(a - b) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def _triangle_area2(p1, p2, p3):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    return float(_triangle_area2_core(p1, p2, p3))
    
def _triangle_descriptor(p1, p2, p3, eps=1e-8):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    out, _ = _triangle_descriptor_core(p1, p2, p3, float(eps))
    return out

def build_triangle_node_features(pts_um, G, n_triangles=3, min_triangle_area_um2=1e-6):
    pts_um = np.asarray(pts_um, float)
    N = len(pts_um)
    tri_feats = np.zeros((N, 2), dtype=np.float32)
    if N == 0 or int(n_triangles) <= 0:
        return tri_feats

    n_triangles = int(n_triangles)
    min_triangle_area_um2 = float(min_triangle_area_um2)

    for i in G.nodes:
        neighs = G.nodes[i].get("neighbors", [])
        if len(neighs) < 2:
            continue

        anchor = neighs[0]
        descs = []
        for nk in neighs[1:1 + n_triangles]:
            if _triangle_area2(pts_um[i], pts_um[anchor], pts_um[nk]) < min_triangle_area_um2:
                continue
            descs.append(_triangle_descriptor(pts_um[i], pts_um[anchor], pts_um[nk]))
        if descs:
            tri_feats[i] = np.mean(descs, axis=0).astype(np.float32)

    return tri_feats
