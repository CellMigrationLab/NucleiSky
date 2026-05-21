import numpy as np
import pytest

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
from nucleisky3d.pipeline import NucleiSky3D


def _run3d(crop, full, *, matcher, n_iters=9000, random_state=17, inlier_radius_um=2.4):
    return NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(320, 360, 420),
        crop_shape_px_zyx=(110, 130, 150),
        pixel_size_full_um_zyx=(3.2, 0.8, 0.35),
        pixel_size_crop_um_zyx=(3.2, 0.8, 0.35),
        matcher=matcher,
        matcher_kwargs={"n_iters": n_iters, "random_state": random_state, "inlier_radius_um": inlier_radius_um},
    )


def assert_not_high_confidence_wrong_match(out, crop_bad, full, *, bad_inlier_radius_um, min_bad_residual_um, max_good_inlier_frac):
    """Wrong-geometry matches must not look high-confidence and accurate simultaneously."""
    assert isinstance(out, dict)
    mq = out.get("match_quality", {}) if isinstance(out.get("match_quality", {}), dict) else {}

    if out.get("success") is True:
        pred = apply_similarity_3d(crop_bad, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
        mets = residual_metrics_nn(pred, full, inlier_radius_um=bad_inlier_radius_um)
        frac_in = float(mq.get("frac_inliers", 0.0))
        forbidden = (mets["median"] < min_bad_residual_um) and (frac_in > max_good_inlier_frac)
        assert not forbidden, (
            "Forbidden silent false positive: wrong geometry reported success with "
            f"median residual {mets['median']:.3f}um and frac_inliers {frac_in:.3f}."
        )


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["hashing", "pyramid"])
def test_3d_axis_order_corruption_never_yields_high_confidence_match(matcher):
    full = make_constellation_3d(seed=1401, n=78)
    R = rot3d_xyz(28.0, -17.0, 132.0)
    s = 1.12
    t = np.array([56.0, -38.0, 72.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))

    out_ok = _run3d(crop, full, matcher=matcher, n_iters=9500, random_state=31)
    if out_ok["success"]:
        pred = apply_similarity_3d(crop, out_ok["best_scale"], np.asarray(out_ok["best_R"]), np.asarray(out_ok["best_t"]))
        mets = residual_metrics_nn(pred, full, inlier_radius_um=2.6)
        assert mets["median"] < 1.8

    for perm in ([2, 1, 0], [0, 2, 1]):
        crop_bad = crop[:, perm]
        out_bad = _run3d(crop_bad, full, matcher=matcher, n_iters=9500, random_state=31)
        assert_not_high_confidence_wrong_match(
            out_bad,
            crop_bad,
            full,
            bad_inlier_radius_um=2.8,
            min_bad_residual_um=3.5,
            max_good_inlier_frac=0.72,
        )


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["hashing", "pyramid"])
def test_3d_near_degenerate_inputs_no_silent_high_confidence_wrong_pose(matcher):
    rng = np.random.default_rng(1501)
    yz = rng.uniform([-22.0, -28.0], [24.0, 30.0], size=(44, 2))
    # Near-coplanar: tiny but non-zero z variation.
    full = np.column_stack([rng.normal(0.0, 0.03, size=44), yz])
    R = rot3d_xyz(4.0, 11.0, 102.0)
    s = 0.97
    t = np.array([12.0, -17.0, 21.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))

    out = _run3d(crop, full, matcher=matcher, n_iters=9000, random_state=41)
    if out["success"]:
        pred = apply_similarity_3d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
        mets = residual_metrics_nn(pred, full, inlier_radius_um=3.0)
        assert mets["median"] < 2.5

    crop_bad = crop.copy()
    crop_bad[:20] = crop_bad[:20][:, [0, 2, 1]]
    crop_bad[:10] += np.array([0.0, 0.2, -0.2])  # mixed coordinate convention corruption
    out_bad = _run3d(crop_bad, full, matcher=matcher, n_iters=9000, random_state=41)
    assert_not_high_confidence_wrong_match(
        out_bad, crop_bad, full, bad_inlier_radius_um=3.2, min_bad_residual_um=3.8, max_good_inlier_frac=0.75
    )


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["hashing", "pyramid"])
def test_3d_sparse_clustered_points_guard_against_confident_wrong_match(matcher):
    full = make_constellation_3d(seed=1601, n=20)
    full[:8] = np.mean(full[:8], axis=0, keepdims=True) + np.array([0.0, 0.05, -0.05])
    R = rot3d_xyz(-7.0, 95.0, 33.0)
    s = 1.05
    t = np.array([-26.0, 19.0, 48.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))

    out = _run3d(crop, full, matcher=matcher, n_iters=11000, random_state=51)
    if out["success"]:
        pred = apply_similarity_3d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
        mets = residual_metrics_nn(pred, full, inlier_radius_um=3.0)
        assert mets["median"] < 2.7

    crop_bad = np.concatenate([crop[:, [1, 0, 2]], crop[:3]], axis=0)
    out_bad = _run3d(crop_bad, full, matcher=matcher, n_iters=11000, random_state=51)
    assert_not_high_confidence_wrong_match(
        out_bad, crop_bad, full, bad_inlier_radius_um=3.0, min_bad_residual_um=4.0, max_good_inlier_frac=0.70
    )


@pytest.mark.geometry
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "scenario",
    [
        {"name": "large_rot_aniso_partial", "rot": (117.0, -34.0, 128.0), "scale": 0.88, "drop": 0.30, "noise": 0.35, "outliers": 18, "t": [78.0, -55.0, 66.0]},
        {"name": "near_180_case", "rot": (174.0, 7.0, -166.0), "scale": 1.23, "drop": 0.26, "noise": 0.30, "outliers": 14, "t": [-62.0, 71.0, -53.0]},
        {"name": "compound_rot_scale_far_shift", "rot": (102.0, 96.0, -121.0), "scale": 1.34, "drop": 0.34, "noise": 0.40, "outliers": 20, "t": [91.0, -64.0, 83.0]},
    ],
)
def test_3d_combined_stress_matrix_no_silent_false_positives(scenario):
    full = make_constellation_3d(seed=1701, n=96)
    R = rot3d_xyz(*scenario["rot"])
    s = float(scenario["scale"])
    t = np.asarray(scenario["t"], dtype=float)

    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))
    crop = drop_fraction(crop, drop_frac=float(scenario["drop"]), seed=1702)
    crop = add_noise(crop, sigma_um=float(scenario["noise"]), seed=1703)
    full_use = add_outliers(add_noise(full, sigma_um=0.2, seed=1704), scenario["outliers"], [-170, -190, -210], [180, 200, 220], seed=1705)

    for matcher in ("hashing", "pyramid"):
        out = _run3d(crop, full_use, matcher=matcher, n_iters=13000, random_state=61, inlier_radius_um=2.8)
        if out["success"]:
            pred = apply_similarity_3d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
            mets = residual_metrics_nn(pred, full_use, inlier_radius_um=3.4)
            assert abs(float(out["best_scale"]) - s) / s < 0.22
            assert rotation_error_deg_3d(np.asarray(out["best_R"]), R) < 24.0
            assert np.linalg.norm(np.asarray(out["best_t"]) - t) < 26.0
            assert mets["median"] < 3.2
            assert out["match_quality"]["frac_inliers"] > 0.38
        else:
            # Hard regimes may legitimately fail, but must not be high-confidence.
            frac = float((out.get("match_quality") or {}).get("frac_inliers", 0.0))
            assert frac < 0.72
