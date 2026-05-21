import numpy as np
import pytest

from nucleisky3d.pipeline import NucleiSky3D
from nucleisky3d.matching.geometry import bbox_full_px_from_similarity_um_3d

from geometry_synth import (
    add_noise,
    add_outliers,
    apply_similarity_3d,
    drop_fraction,
    make_constellation_3d,
    residual_metrics_nn,
    rot3d_xyz,
    rotation_error_deg_3d,
)


def _run_3d(crop, full, matcher="pyramid"):
    return NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(256, 256, 256),
        crop_shape_px_zyx=(96, 96, 96),
        pixel_size_full_um_zyx=(1.0, 0.6, 0.6),
        pixel_size_crop_um_zyx=(1.0, 0.6, 0.6),
        matcher=matcher,
        matcher_kwargs={"n_iters": 12000, "inlier_radius_um": 1.7, "random_state": 11},
    )


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["pyramid", "hashing"])
def test_3d_identity_and_similarity_recovery(matcher):
    full = make_constellation_3d()
    scale = 1.18
    R = rot3d_xyz(18.0, -11.0, 29.0)
    t = np.array([5.0, -8.0, 12.0], dtype=float)
    crop = apply_similarity_3d(full, 1.0 / scale, R.T, -((1.0 / scale) * (t @ R)))

    out = _run_3d(crop, full, matcher=matcher)
    assert out["success"] is True

    s_est = float(out["best_scale"])
    R_est = np.asarray(out["best_R"], float)
    t_est = np.asarray(out["best_t"], float)
    pred_full = apply_similarity_3d(crop, s_est, R_est, t_est)
    mets = residual_metrics_nn(pred_full, full, inlier_radius_um=2.0)

    assert abs(s_est - scale) / scale < 0.04
    assert rotation_error_deg_3d(R_est, R) < 3.0
    assert np.linalg.norm(t_est - t) < 3.0
    assert mets["median"] < 1.2
    assert mets["p95"] < 2.2
    assert mets["inlier_frac"] > 0.85


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["pyramid", "hashing"])
def test_3d_noise_partial_overlap_outliers(matcher):
    full = make_constellation_3d(seed=778)
    scale = 0.91
    R = rot3d_xyz(-12.0, 21.0, 15.0)
    t = np.array([-7.0, 10.0, 6.0], dtype=float)
    crop = apply_similarity_3d(full, 1.0 / scale, R.T, -((1.0 / scale) * (t @ R)))
    crop = drop_fraction(crop, drop_frac=0.30, seed=9)
    crop = add_noise(crop, sigma_um=0.35, seed=10)
    full_noisy = add_noise(full, sigma_um=0.2, seed=12)
    full_noisy = add_outliers(full_noisy, n_outliers=24, bounds_low=[-80, -110, -130], bounds_high=[90, 120, 140], seed=13)

    out = _run_3d(crop, full_noisy, matcher=matcher)
    assert out["success"] is True
    pred_full = apply_similarity_3d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
    mets = residual_metrics_nn(pred_full, full_noisy, inlier_radius_um=2.8)

    assert abs(float(out["best_scale"]) - scale) / scale < 0.1
    assert rotation_error_deg_3d(np.asarray(out["best_R"]), R) < 8.0
    assert np.linalg.norm(np.asarray(out["best_t"]) - t) < 8.0
    assert mets["median"] < 2.0
    assert mets["p95"] < 5.0
    assert mets["inlier_frac"] > 0.60


@pytest.mark.geometry
def test_3d_anisotropic_voxel_bbox_consistency():
    scale = 1.0
    R = np.eye(3, dtype=float)
    t = np.array([10.0, 18.0, 26.0], dtype=float)
    bbox = bbox_full_px_from_similarity_um_3d(
        crop_shape_px=(12, 20, 28),
        pixel_size_full_um_zyx=(2.0, 0.5, 0.25),
        pixel_size_crop_um_zyx=(2.0, 0.5, 0.25),
        scale=scale,
        R_zyx=R,
        t_um_zyx=t,
        full_shape_px=(200, 300, 400),
    )
    z0, z1, y0, y1, x0, x1 = tuple(int(v) for v in bbox)
    assert (z0, y0, x0) == (5, 36, 104)
    assert (z1 - z0, y1 - y0, x1 - x0) == (12, 20, 28)


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["pyramid", "hashing"])
def test_3d_insufficient_points_fail_cleanly(matcher):
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 3.0, 2.0]], dtype=float)
    out = _run_3d(pts, pts, matcher=matcher)
    assert out["success"] is False


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["pyramid", "hashing"])
def test_3d_coplanar_points_have_matcher_dependent_behavior(matcher):
    rng = np.random.default_rng(99)
    yz = rng.uniform(-20.0, 20.0, size=(40, 2))
    full = np.column_stack([np.zeros(40), yz])
    R = rot3d_xyz(0.0, 0.0, 40.0)
    t = np.array([8.0, -4.0, 11.0], dtype=float)
    crop = apply_similarity_3d(full, 1.0, R.T, -(t @ R))
    out = _run_3d(crop, full, matcher=matcher)
    if matcher == "pyramid":
        # Pyramid can legitimately fail on strictly coplanar inputs due to tetrahedral feature constraints.
        assert out["success"] is False
    else:
        assert out["success"] is True
        assert rotation_error_deg_3d(np.asarray(out["best_R"]), R) < 1.0
        assert np.linalg.norm(np.asarray(out["best_t"]) - t) < 1.0
