"""geometry.py Shared geometry utilities for 3D matching."""

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from ..export import similarity_um_to_affine_px_3d

from ..types import BBox3D


@njit(fastmath=True)
def _estimate_similarity_3d_core(src, dst):
    src_mean = np.empty(3, dtype=np.float64)
    dst_mean = np.empty(3, dtype=np.float64)
    for c in range(3):
        src_mean[c] = np.mean(src[:, c])
        dst_mean[c] = np.mean(dst[:, c])

    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = float(np.sum(src_c * src_c))
    if (not np.isfinite(var_src)) or var_src < 1e-12:
        return 0.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False

    covariance = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(covariance, full_matrices=False)

    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(Vt.T @ U.T) < 0.0:
        D[2, 2] = -1.0

    R = Vt.T @ D @ U.T
    scale = float(np.sum(S * np.diag(D))) / var_src
    if (not np.isfinite(scale)) or scale <= 0.0:
        return 0.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False

    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t, True


@njit(fastmath=True)
def _apply_similarity_3d_core(pts, scale, R, t):
    return scale * (pts @ R.T) + t



def bbox_add_margin_px_3d(bbox_zyx6, margin_px, shape_zyx=None):
    """Expand a half-open 3D bounding box by a pixel margin.

    Parameters
    ----------
    bbox_zyx6
        Input bbox as ``(z0, z1, y0, y1, x0, x1)`` in half-open pixel coordinates.
    margin_px
        Either a scalar margin applied to all axes, or a 3-tuple
        ``(mz, my, mx)`` in pixels.
    shape_zyx
        Optional full-volume shape ``(Z, Y, X)``. If provided, the expanded
        bbox is clamped to ``[0, Z] x [0, Y] x [0, X]``.
    """
    z0, z1, y0, y1, x0, x1 = map(int, np.asarray(bbox_zyx6, dtype=int).reshape(6,))

    m = np.asarray(margin_px, dtype=float)
    if m.ndim == 0:
        mz = my = mx = float(m)
    elif m.shape == (3,):
        mz, my, mx = map(float, m)
    else:
        raise ValueError("margin_px must be a scalar or a 3-tuple (mz, my, mx).")

    if not np.isfinite([mz, my, mx]).all() or min(mz, my, mx) < 0:
        raise ValueError("margin_px must contain finite non-negative values.")

    mz_i = int(np.ceil(mz))
    my_i = int(np.ceil(my))
    mx_i = int(np.ceil(mx))

    out = (
        z0 - mz_i,
        z1 + mz_i,
        y0 - my_i,
        y1 + my_i,
        x0 - mx_i,
        x1 + mx_i,
    )

    if shape_zyx is None:
        return out

    Z, Y, X = map(int, np.asarray(shape_zyx, dtype=int).reshape(3,))
    return (
        max(0, out[0]),
        min(Z, out[1]),
        max(0, out[2]),
        min(Y, out[3]),
        max(0, out[4]),
        min(X, out[5]),
    )


def bbox_full_px_from_similarity_um_3d(
    crop_shape_px,
    pixel_size_full_um_zyx,
    pixel_size_crop_um_zyx,
    scale,
    R_zyx,
    t_um_zyx,
    margin_um=0.0,
    full_shape_px=None,
) -> BBox3D:
    """
    Compute a 3D bounding box in FULL pixel coordinates covering the transformed crop footprint.
    """
    Zc, Yc, Xc = map(int, crop_shape_px[:3])
    pix_full = np.asarray(pixel_size_full_um_zyx, dtype=float).reshape(3,)
    pix_crop = np.asarray(pixel_size_crop_um_zyx, dtype=float).reshape(3,)
    s = float(scale)
    R = np.asarray(R_zyx, float).reshape(3, 3)
    t = np.asarray(t_um_zyx, float).reshape(3,)

    if Zc <= 0 or Yc <= 0 or Xc <= 0:
        raise ValueError(f"crop_shape_px must be positive, got {crop_shape_px}")
    if (pix_full <= 0).any() or (pix_crop <= 0).any() or (not np.isfinite(pix_full).all()) or (
        not np.isfinite(pix_crop).all()
    ):
        raise ValueError("pixel sizes must be positive finite floats (µm/px).")
    if s <= 0 or (not np.isfinite(s)):
        raise ValueError("scale must be positive finite.")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("R_zyx and t_um_zyx must be finite.")

    corners_px = np.array(
        [
            [0.0, 0.0, 0.0],
            [float(Zc), 0.0, 0.0],
            [0.0, float(Yc), 0.0],
            [0.0, 0.0, float(Xc)],
            [float(Zc), float(Yc), 0.0],
            [float(Zc), 0.0, float(Xc)],
            [0.0, float(Yc), float(Xc)],
            [float(Zc), float(Yc), float(Xc)],
        ],
        dtype=float,
    )

    corners_um = corners_px * pix_crop
    corners_full_um = (s * (corners_um @ R.T)) + t[None, :]

    m = float(margin_um)
    z0_um = float(corners_full_um[:, 0].min()) - m
    z1_um = float(corners_full_um[:, 0].max()) + m
    y0_um = float(corners_full_um[:, 1].min()) - m
    y1_um = float(corners_full_um[:, 1].max()) + m
    x0_um = float(corners_full_um[:, 2].min()) - m
    x1_um = float(corners_full_um[:, 2].max()) + m

    z0 = int(np.floor(z0_um / pix_full[0]))
    z1 = int(np.ceil(z1_um / pix_full[0]))
    y0 = int(np.floor(y0_um / pix_full[1]))
    y1 = int(np.ceil(y1_um / pix_full[1]))
    x0 = int(np.floor(x0_um / pix_full[2]))
    x1 = int(np.ceil(x1_um / pix_full[2]))

    bbox = BBox3D(z0, z1, y0, y1, x0, x1)

    if full_shape_px is not None:
        bbox = bbox.clamp(tuple(map(int, full_shape_px[:3])), min_size=1)

    return bbox


def bbox_intersection_volume_3d(bbox_a, bbox_b) -> int:
    """Compute intersection volume of two half-open 3D bounding boxes."""
    a = np.asarray(bbox_a, dtype=int).reshape(6,)
    b = np.asarray(bbox_b, dtype=int).reshape(6,)

    z0 = max(a[0], b[0])
    z1 = min(a[1], b[1])
    y0 = max(a[2], b[2])
    y1 = min(a[3], b[3])
    x0 = max(a[4], b[4])
    x1 = min(a[5], b[5])

    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return 0
    
    dz = float(max(0, z1 - z0))
    dy = float(max(0, y1 - y0))
    dx = float(max(0, x1 - x0))
    return int(dz * dy * dx)


def bbox_overlap_iou_3d(bbox_a, bbox_b) -> float:
    """Compute 3D intersection-over-union (IoU) for two half-open bounding boxes."""
    a = np.asarray(bbox_a, dtype=float).reshape(6,)
    b = np.asarray(bbox_b, dtype=float).reshape(6,)

    inter = float(bbox_intersection_volume_3d(a, b))
    vol_a = max(0.0, (a[1] - a[0]) * (a[3] - a[2]) * (a[5] - a[4]))
    vol_b = max(0.0, (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4]))
    union = vol_a + vol_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def estimate_dynamic_scale_bounds_3d(
    df_full,
    df_crop,
    voxel_size_full_um_zyx,
    voxel_size_crop_um_zyx,
    full_shape_px_zyx,
    crop_shape_px_zyx,
    coarse_scale_min=0.5,
    coarse_scale_max=2.0,
    rel_tol=0.1,
):
    """Estimate a dynamic scale prior and effective bounds using 3D cues."""
    eps = 1e-8
    rel_tol = float(rel_tol)
    rel_tol = max(0.0, min(0.95, rel_tol))

    def _median_nn_all(df):
        cols = [c for c in df.columns if c.startswith("nn") and c.endswith("_dist_um")]
        if cols:
            vals = df[cols].to_numpy().ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                return float(np.median(vals))

        centroid_cols = ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
        if not all(c in df.columns for c in centroid_cols):
            return None

        pts = df[centroid_cols].to_numpy(dtype=float, copy=False)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
            return None

        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            return None

        tree = cKDTree(pts)
        dists = tree.query(pts, k=2)[0]
        dists = np.asarray(dists, dtype=float)
        if dists.ndim != 2 or dists.shape[1] < 2:
            return None

        vals = dists[:, 1]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    # 1) geometric cue
    med_nn_full = _median_nn_all(df_full)
    med_nn_crop = _median_nn_all(df_crop)
    s_nn = None
    if med_nn_full not in (None, 0) and med_nn_crop not in (None, 0):
        s_nn = med_nn_full / (med_nn_crop + eps)

    # 2) morphology cue
    s_size = None
    if (
        "equiv_spherical_diameter_um" in df_full.columns
        and "equiv_spherical_diameter_um" in df_crop.columns
    ):
        d_full = df_full["equiv_spherical_diameter_um"].to_numpy()
        d_crop = df_crop["equiv_spherical_diameter_um"].to_numpy()
        d_full = d_full[np.isfinite(d_full) & (d_full > 0)]
        d_crop = d_crop[np.isfinite(d_crop) & (d_crop > 0)]
        if d_full.size and d_crop.size:
            s_size = float(np.median(d_full)) / (float(np.median(d_crop)) + eps)

    # 3) density cue
    Zf, Yf, Xf = map(int, full_shape_px_zyx[:3])
    Zc, Yc, Xc = map(int, crop_shape_px_zyx[:3])

    vox_full = np.asarray(voxel_size_full_um_zyx, dtype=float).reshape(3,)
    vox_crop = np.asarray(voxel_size_crop_um_zyx, dtype=float).reshape(3,)

    vol_full_um3 = (Zf * vox_full[0]) * (Yf * vox_full[1]) * (Xf * vox_full[2])
    vol_crop_um3 = (Zc * vox_crop[0]) * (Yc * vox_crop[1]) * (Xc * vox_crop[2])

    dens_full = len(df_full) / (vol_full_um3 + eps)
    dens_crop = len(df_crop) / (vol_crop_um3 + eps)

    s_dens = None
    if dens_full > 0 and dens_crop > 0:
        s_dens = float((dens_full / (dens_crop + eps)) ** (1.0 / 3.0))

    density_is_useful = False
    if s_dens is not None:
        vol_ok = (
            np.isfinite(vol_full_um3)
            and np.isfinite(vol_crop_um3)
            and vol_full_um3 > eps
            and vol_crop_um3 > eps
        )
        enough_points = min(len(df_full), len(df_crop)) >= 3
        finite_positive = np.isfinite(s_dens) and s_dens > 0
        non_default = abs(np.log(s_dens)) >= np.log(1.05)
        density_is_useful = bool(vol_ok and enough_points and finite_positive and non_default)

    # 4) combine robustly
    primary_candidates = [
        s for s in (s_nn, s_size)
        if s is not None and np.isfinite(s) and 0.1 < s < 10.0
    ]
    scale_candidates = list(primary_candidates)
    if density_is_useful and 0.1 < s_dens < 10.0:
        scale_candidates.append(float(s_dens))

    if scale_candidates:
        scale_prior = float(np.median(scale_candidates))
    elif density_is_useful and 0.1 < s_dens < 10.0:
        scale_prior = float(s_dens)
    else:
        scale_prior = 1.0

    scale_min_eff = max(float(coarse_scale_min), scale_prior * (1.0 - rel_tol))
    scale_max_eff = min(float(coarse_scale_max), scale_prior * (1.0 + rel_tol))

    if not (scale_min_eff < scale_max_eff):
        scale_min_eff = float(coarse_scale_min)
        scale_max_eff = float(coarse_scale_max)

    return scale_prior, scale_min_eff, scale_max_eff


def estimate_similarity_3d(src_pts, dst_pts):
    """Estimate 3D similarity transform (scale, rotation, translation) mapping src -> dst."""
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)

    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError(
            "estimate_similarity_3d expects src and dst shaped (N,3). "
            f"Got src={src.shape}, dst={dst.shape}"
        )

    n_points = src.shape[0]
    if n_points < 3:
        raise ValueError("estimate_similarity_3d requires at least 3 points (preferably 4+).")

    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("estimate_similarity_3d received non-finite values in src/dst.")

    scale, R, t, success = _estimate_similarity_3d_core(src, dst)
    if not success:
        raise ValueError("Degenerate source configuration or invalid similarity estimate.")

    return float(scale), np.asarray(R, float), np.asarray(t, float)


def apply_similarity_3d(pts, scale, R, t):
    """Apply a 3D similarity transform to points."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"apply_similarity_3d expects pts shaped (N,3). Got pts={pts.shape}")

    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3,)
    s = float(scale)

    if not np.isfinite(s):
        raise ValueError("scale must be finite.")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("R and t must be finite.")

    return _apply_similarity_3d_core(pts, s, R, t)


def rotation_angle_deg_3d(R_zyx) -> float:
    """Return principal SO(3) rotation angle (degrees) from a 3x3 rotation matrix."""
    R = np.asarray(R_zyx, dtype=float).reshape(3, 3)
    if not np.isfinite(R).all():
        raise ValueError("R_zyx must be finite.")

    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta_rad = float(np.arccos(cos_theta))
    return float(np.degrees(theta_rad))


def sanitize_points_zyx_um(
    points_um_zyx,
    *,
    dedup_radius_um: float = 0.0,
    drop_nonfinite: bool = True,
    nn_outlier_percentile: float | None = None,
    min_points: int = 4,
    name: str = "points",
):
    """Sanitize a ZYX point cloud in microns and return sanitized points with stats."""
    points = np.asarray(points_um_zyx, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) in ZYX order. Got {points.shape}.")

    n_in = int(points.shape[0])
    keep = np.ones(n_in, dtype=bool)

    if drop_nonfinite:
        keep &= np.isfinite(points).all(axis=1)

    points = points[keep]
    n_finite = int(points.shape[0])

    n_deduped = 0
    if dedup_radius_um is not None and float(dedup_radius_um) > 0.0 and points.shape[0] > 0:
        radius = float(dedup_radius_um)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"dedup_radius_um must be a positive finite float, got {dedup_radius_um}.")

        voxel = np.rint(points / radius).astype(np.int64)
        _, inverse = np.unique(voxel, axis=0, return_inverse=True)
        counts = np.bincount(inverse)

        merged = np.zeros((counts.size, 3), dtype=float)
        for axis in range(3):
            merged[:, axis] = np.bincount(inverse, weights=points[:, axis]) / np.maximum(counts, 1)

        n_deduped = int(points.shape[0] - merged.shape[0])
        points = merged

    n_outliers_dropped = 0
    if nn_outlier_percentile is not None and points.shape[0] >= 2:
        pct = float(nn_outlier_percentile)
        if not np.isfinite(pct) or pct <= 0.0 or pct >= 100.0:
            raise ValueError(
                f"nn_outlier_percentile must be in (0, 100), got {nn_outlier_percentile}."
            )

        tree = cKDTree(points)
        dists = tree.query(points, k=2)[0][:, 1]
        cutoff = float(np.percentile(dists, pct))
        keep = dists <= cutoff
        n_outliers_dropped = int((~keep).sum())
        points = points[keep]

    if points.shape[0] < int(min_points):
        raise ValueError(
            f"{name} has insufficient points after sanitization: {points.shape[0]} < {int(min_points)}."
        )

    stats = {
        "n_in": n_in,
        "n_finite": n_finite,
        "n_deduped": int(n_deduped),
        "n_outliers_dropped": int(n_outliers_dropped),
    }
    return points.astype(np.float32, copy=False), stats



def icp_similarity_3d(
    src_pts,
    dst_pts,
    init_scale,
    init_R,
    init_t,
    n_iters=10,
    inlier_radius_um=2.0,
):
    """Simple ICP refinement for similarity transform in 3D (all in µm)."""
    src = np.asarray(src_pts, float)
    dst = np.asarray(dst_pts, float)

    scale = float(init_scale)
    R = np.asarray(init_R, float).copy()
    t = np.asarray(init_t, float).copy()

    tree_dst = cKDTree(dst)

    for _ in range(int(n_iters)):
        src_trans = apply_similarity_3d(src, scale, R, t)
        dists, idxs = tree_dst.query(src_trans, distance_upper_bound=float(inlier_radius_um))
        mask = np.isfinite(dists)
        if int(mask.sum()) < 3:
            break

        src_in = src[mask]
        dst_in = dst[idxs[mask]]

        scale, R, t = estimate_similarity_3d(src_in, dst_in)

    return float(scale), np.asarray(R, float), np.asarray(t, float)
