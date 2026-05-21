import numpy as np
import pytest

from nucleisky2d.pipeline import NucleiSky
from nucleisky3d.pipeline import NucleiSky3D

from geometry_synth import (
    apply_similarity_2d,
    apply_similarity_3d,
    make_constellation_2d,
    make_constellation_3d,
    residual_metrics_nn,
    rot2d,
    rot3d_xyz,
    rotation_error_deg_2d,
    rotation_error_deg_3d,
)


def _run2(crop, full):
    return NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((1000, 1000), dtype=np.float32),
        img_crop=np.zeros((500, 500), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="hashing",
        matcher_kwargs={"n_iters": 12000, "random_state": 22, "inlier_radius_um": 1.7},
    )


def _run3(crop, full):
    return NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(200, 220, 240),
        crop_shape_px_zyx=(90, 100, 110),
        pixel_size_full_um_zyx=(1.0, 0.6, 0.4),
        pixel_size_crop_um_zyx=(1.0, 0.6, 0.4),
        matcher="hashing",
        matcher_kwargs={"n_iters": 10000, "random_state": 33, "inlier_radius_um": 2.0},
    )


def _assert_not_confident_axis_mismatch(
    out,
    crop_bad,
    full,
    *,
    apply_similarity,
    bad_inlier_radius_um: float,
    min_bad_residual_um: float,
    max_good_inlier_frac: float,
):
    if out["success"] is True:
        pred = apply_similarity(crop_bad, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
        mets = residual_metrics_nn(pred, full, inlier_radius_um=bad_inlier_radius_um)
        is_false_positive = (mets["median"] < min_bad_residual_um) and (mets["inlier_frac"] > max_good_inlier_frac)
        assert not is_false_positive, (
            "Forbidden silent false positive: axis-mismatched geometry reported success with "
            f"median residual {mets['median']:.3f}um and inlier fraction {mets['inlier_frac']:.3f}."
        )


@pytest.mark.geometry
def test_2d_translation_invariance_of_relative_transform():
    full = make_constellation_2d(seed=10)
    R = rot2d(19.0)
    scale = 1.13
    t = np.array([7.0, -12.0], dtype=float)
    crop = apply_similarity_2d(full, 1 / scale, R.T, -((1 / scale) * (t @ R)))
    base = _run2(crop, full)

    shift = np.array([100.0, -200.0], dtype=float)
    shifted = _run2(crop + shift, full + shift)
    assert base["success"] and shifted["success"]
    assert abs(base["best_scale"] - shifted["best_scale"]) < 1e-6
    assert rotation_error_deg_2d(np.asarray(base["best_R"]), np.asarray(shifted["best_R"])) < 1e-4
    pred_shift = apply_similarity_2d(
        crop + shift, shifted["best_scale"], np.asarray(shifted["best_R"]), np.asarray(shifted["best_t"])
    )
    mets = residual_metrics_nn(pred_shift, full + shift, inlier_radius_um=1.8)
    assert mets["median"] < 1e-4


@pytest.mark.geometry
def test_2d_point_order_and_seed_determinism():
    full = make_constellation_2d(seed=11)
    R = rot2d(-28.0)
    crop = apply_similarity_2d(full, 1.0, R.T, np.array([4.0, 5.0]))
    out_a = _run2(crop, full)
    out_b = _run2(crop, full)
    perm = np.random.default_rng(11).permutation(len(crop))
    out_c = _run2(crop[perm], full[perm])
    for out in (out_a, out_b, out_c):
        assert out["success"] is True
    np.testing.assert_allclose(out_a["best_scale"], out_b["best_scale"], atol=1e-9)
    np.testing.assert_allclose(out_a["best_t"], out_b["best_t"], atol=1e-8)
    assert rotation_error_deg_2d(np.asarray(out_a["best_R"]), np.asarray(out_b["best_R"])) < 1e-6
    assert rotation_error_deg_2d(np.asarray(out_a["best_R"]), np.asarray(out_c["best_R"])) < 0.2


@pytest.mark.geometry
def test_2d_axis_swap_robustness_detects_mismatch():
    full = make_constellation_2d(seed=12)
    out_ok = _run2(full.copy(), full.copy())
    crop_bad = full[:, ::-1]
    out_bad = _run2(crop_bad, full)
    assert out_ok["success"] is True
    # Deliberate y/x mismatch must not look like a perfect identity match.
    _assert_not_confident_axis_mismatch(
        out_bad,
        crop_bad,
        full,
        apply_similarity=apply_similarity_2d,
        bad_inlier_radius_um=1.8,
        min_bad_residual_um=4.0,
        max_good_inlier_frac=0.55,
    )


@pytest.mark.geometry
def test_3d_global_translation_and_scaling_invariance():
    full = make_constellation_3d(seed=20)
    R = rot3d_xyz(14.0, 9.0, -17.0)
    s = 1.08
    t = np.array([-6.0, 8.0, 11.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))
    base = _run3(crop, full)
    assert base["success"] is True

    shift = np.array([40.0, -20.0, 10.0], dtype=float)
    moved = _run3(crop + shift, full + shift)
    assert moved["success"] is True
    np.testing.assert_allclose(base["best_scale"], moved["best_scale"], atol=1e-6)
    pred_moved = apply_similarity_3d(
        crop + shift, moved["best_scale"], np.asarray(moved["best_R"]), np.asarray(moved["best_t"])
    )
    mets = residual_metrics_nn(pred_moved, full + shift, inlier_radius_um=2.0)
    assert mets["median"] < 1e-4

    k = 2.5
    scaled = _run3(crop * k, full * k)
    assert scaled["success"] is True
    assert abs(float(scaled["best_scale"]) - float(base["best_scale"])) < 1e-5


@pytest.mark.geometry
@pytest.mark.stochastic
def test_3d_seed_determinism_and_axis_order_robustness():
    full = make_constellation_3d(seed=21)
    R = rot3d_xyz(-11.0, 13.0, 22.0)
    crop = apply_similarity_3d(full, 1.0, R.T, np.array([2.0, -3.0, 4.0]))
    out1 = _run3(crop, full)
    out2 = _run3(crop, full)
    assert out1["success"] and out2["success"]
    np.testing.assert_allclose(out1["best_scale"], out2["best_scale"], atol=1e-10)
    np.testing.assert_allclose(out1["best_t"], out2["best_t"], atol=1e-8)
    assert rotation_error_deg_3d(np.asarray(out1["best_R"]), np.asarray(out2["best_R"])) < 1e-6

    # swap z/x on crop only -> should fail or strongly degrade quality
    crop_bad = crop[:, [2, 1, 0]]
    out_bad = _run3(crop_bad, full)
    _assert_not_confident_axis_mismatch(
        out_bad,
        crop_bad,
        full,
        apply_similarity=apply_similarity_3d,
        bad_inlier_radius_um=2.0,
        min_bad_residual_um=6.5,
        max_good_inlier_frac=0.40,
    )
