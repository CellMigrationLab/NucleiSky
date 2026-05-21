import numpy as np
import pytest

from nucleisky2d.pipeline import NucleiSky

from geometry_synth import (
    add_noise,
    add_outliers,
    apply_similarity_2d,
    drop_fraction,
    make_constellation_2d,
    residual_metrics_nn,
    rot2d,
    rotation_error_deg_2d,
)


def _run_2d(crop, full, matcher="hashing"):
    img_full = np.zeros((1024, 1024), dtype=np.float32)
    img_crop = np.zeros((512, 512), dtype=np.float32)
    return NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=img_full,
        img_crop=img_crop,
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher=matcher,
        matcher_kwargs={"n_iters": 15000, "inlier_radius_um": 1.5, "random_state": 7},
    )


@pytest.mark.geometry
@pytest.mark.parametrize(
    "scale, angle_deg, t",
    [
        (1.0, 0.0, np.array([0.0, 0.0])),
        (1.0, 31.0, np.array([18.0, -12.0])),
        (1.24, -27.0, np.array([-8.0, 23.0])),
    ],
)
def test_2d_recovers_similarity_transform(scale, angle_deg, t):
    full = make_constellation_2d()
    R = rot2d(angle_deg)
    crop = apply_similarity_2d(full, 1.0 / scale, R.T, -((1.0 / scale) * (t @ R)))

    out = _run_2d(crop, full)
    assert out["success"] is True

    s_est = float(out["best_scale"])
    R_est = np.asarray(out["best_R"], float)
    t_est = np.asarray(out["best_t"], float)
    pred_full = apply_similarity_2d(crop, s_est, R_est, t_est)
    mets = residual_metrics_nn(pred_full, full, inlier_radius_um=1.8)

    assert abs(s_est - scale) / scale < 0.03
    assert rotation_error_deg_2d(R_est, R) < 2.0
    assert float(np.linalg.norm(t_est - t)) < 2.5
    assert mets["median"] < 1.2
    assert mets["p95"] < 2.0
    assert mets["inlier_frac"] > 0.85


@pytest.mark.geometry
def test_2d_recovery_with_noise_partial_overlap_and_outliers():
    full = make_constellation_2d(seed=321)
    scale, ang, t = 0.84, 22.0, np.array([15.0, -10.0], dtype=float)
    R = rot2d(ang)
    crop = apply_similarity_2d(full, 1.0 / scale, R.T, -((1.0 / scale) * (t @ R)))
    crop = drop_fraction(crop, drop_frac=0.35, seed=13)
    crop = add_noise(crop, sigma_um=0.35, seed=14)
    full_noisy = add_noise(full, sigma_um=0.2, seed=15)
    full_noisy = add_outliers(full_noisy, n_outliers=18, bounds_low=[-120, -120], bounds_high=[120, 120], seed=16)

    out = _run_2d(crop, full_noisy)
    assert out["success"] is True
    pred_full = apply_similarity_2d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
    mets = residual_metrics_nn(pred_full, full_noisy, inlier_radius_um=2.2)

    assert abs(float(out["best_scale"]) - scale) / scale < 0.08
    assert rotation_error_deg_2d(np.asarray(out["best_R"]), R) < 6.0
    assert np.linalg.norm(np.asarray(out["best_t"]) - t) < 6.5
    assert mets["median"] < 1.7
    assert mets["p95"] < 4.0
    assert mets["inlier_frac"] > 0.65


@pytest.mark.geometry
def test_2d_insufficient_points_fail_cleanly():
    full = np.array([[0.0, 0.0], [4.0, 1.0]], dtype=float)
    crop = full.copy()
    out = _run_2d(crop, full)
    assert out["success"] is False


@pytest.mark.geometry
def test_2d_collinear_points_can_match_but_are_transform_ambiguous():
    y = np.linspace(-10.0, 10.0, 24)
    full = np.stack([y, 2.0 * y + 1.0], axis=1)
    R = rot2d(35.0)
    t = np.array([5.0, -7.0], dtype=float)
    crop = apply_similarity_2d(full, 1.0, R.T, -(t @ R))

    out = _run_2d(crop, full)
    assert out["success"] is True
    pred_full = apply_similarity_2d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
    mets = residual_metrics_nn(pred_full, full, inlier_radius_um=1.0)
    # Collinear constellations are mirror/rotation ambiguous; enforce fit quality, not unique pose.
    assert mets["median"] < 0.7
    assert mets["inlier_frac"] > 0.95
