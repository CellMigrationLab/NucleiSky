from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Similarity2D:
    scale: float
    angle_deg: float
    translation_yx: np.ndarray


@dataclass(frozen=True)
class Similarity3D:
    scale: float
    rotation_zyx: np.ndarray
    translation_zyx: np.ndarray


def make_constellation_2d(n: int = 48, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(low=[-35.0, -55.0], high=[45.0, 65.0], size=(n, 2))
    pts += 0.15 * np.stack([pts[:, 0] ** 2 / 100.0, np.sin(pts[:, 1] / 15.0)], axis=1)
    return pts.astype(np.float64)


def make_constellation_3d(n: int = 72, seed: int = 456) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(low=[-18.0, -42.0, -55.0], high=[34.0, 48.0, 66.0], size=(n, 3))
    pts[:, 0] += 0.1 * np.sin(pts[:, 1] / 8.0)
    pts[:, 2] += 0.07 * (pts[:, 0] ** 2) / 30.0
    return pts.astype(np.float64)


def rot2d(angle_deg: float) -> np.ndarray:
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def rot3d_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = map(np.deg2rad, (rx_deg, ry_deg, rz_deg))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float64)


def apply_similarity_2d(pts: np.ndarray, scale: float, R: np.ndarray, t_yx: np.ndarray) -> np.ndarray:
    return (float(scale) * (pts @ R.T) + np.asarray(t_yx, dtype=np.float64)).astype(np.float64)


def apply_similarity_3d(pts: np.ndarray, scale: float, R: np.ndarray, t_zyx: np.ndarray) -> np.ndarray:
    return (float(scale) * (pts @ R.T) + np.asarray(t_zyx, dtype=np.float64)).astype(np.float64)


def add_noise(pts: np.ndarray, sigma_um: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (pts + rng.normal(0.0, float(sigma_um), size=pts.shape)).astype(np.float64)


def drop_fraction(pts: np.ndarray, drop_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(pts)
    k = int(round((1.0 - float(drop_frac)) * n))
    k = max(1, min(k, n))
    idx = rng.choice(n, size=k, replace=False)
    return pts[np.sort(idx)]


def add_outliers(pts: np.ndarray, n_outliers: int, bounds_low, bounds_high, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rng.uniform(low=np.asarray(bounds_low), high=np.asarray(bounds_high), size=(n_outliers, pts.shape[1]))
    return np.concatenate([pts, out], axis=0).astype(np.float64)


def duplicate_points(pts: np.ndarray, n_dup: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=n_dup, replace=True)
    return np.concatenate([pts, pts[idx]], axis=0).astype(np.float64)


def residual_metrics_nn(pred_full: np.ndarray, full_pts: np.ndarray, inlier_radius_um: float = 2.0) -> dict:
    tree = cKDTree(full_pts)
    d, _ = tree.query(pred_full, k=1)
    return {
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "inlier_frac": float(np.mean(d <= float(inlier_radius_um))),
    }


def rotation_error_deg_2d(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    a_est = math.degrees(math.atan2(R_est[1, 0], R_est[0, 0]))
    a_gt = math.degrees(math.atan2(R_gt[1, 0], R_gt[0, 0]))
    d = (a_est - a_gt + 180.0) % 360.0 - 180.0
    return abs(float(d))


def rotation_error_deg_3d(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    Rrel = np.asarray(R_est, float) @ np.asarray(R_gt, float).T
    tr = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

