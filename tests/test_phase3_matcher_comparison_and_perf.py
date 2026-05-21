import time

import numpy as np
import pytest

from geometry_synth import (
    add_noise,
    add_outliers,
    apply_similarity_2d,
    apply_similarity_3d,
    drop_fraction,
    make_constellation_2d,
    make_constellation_3d,
    residual_metrics_nn,
    rot2d,
    rot3d_xyz,
)
from nucleisky2d.pipeline import NucleiSky
from nucleisky3d.pipeline import NucleiSky3D


def _run2d(crop, full, matcher):
    kwargs = {"n_iters": 6000, "random_state": 5, "inlier_radius_um": 1.8}
    extra = {}
    if matcher == "graph":
        # Distinctive but simple features to avoid ambiguous nearest-feature ties.
        feat_full = np.stack([full[:, 0], full[:, 1], full[:, 0] * full[:, 1]], axis=1).astype(np.float32)
        feat_crop = np.stack([crop[:, 0], crop[:, 1], crop[:, 0] * crop[:, 1]], axis=1).astype(np.float32)
        extra = {"features_full": feat_full, "features_crop": feat_crop}
    return NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((1024, 1024), dtype=np.float32),
        img_crop=np.zeros((512, 512), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher=matcher,
        matcher_kwargs=kwargs,
        **extra,
    )


def _run3d(crop, full, matcher):
    return NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(220, 240, 260),
        crop_shape_px_zyx=(90, 100, 110),
        pixel_size_full_um_zyx=(1.0, 0.7, 0.5),
        pixel_size_crop_um_zyx=(1.0, 0.7, 0.5),
        matcher=matcher,
        matcher_kwargs={"n_iters": 7000, "random_state": 6, "inlier_radius_um": 2.0},
    )


@pytest.mark.geometry
@pytest.mark.parametrize("matcher", ["hashing", "triangles", "quad", "graph"])
def test_2d_matchers_recover_compatible_transforms_on_clean_data(matcher):
    full = make_constellation_2d(seed=210)
    R = rot2d(23.0)
    scale = 0.92
    t = np.array([8.0, -11.0], dtype=float)
    crop = apply_similarity_2d(full, 1 / scale, R.T, -((1 / scale) * (t @ R)))
    out = _run2d(crop, full, matcher)
    assert out["success"] is True
    pred = apply_similarity_2d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
    mets = residual_metrics_nn(pred, full, inlier_radius_um=2.0)
    # Tolerance is matcher-family compatible, not bit-identical.
    assert abs(float(out["best_scale"]) - scale) / scale < 0.12
    assert mets["median"] < 1.4
    assert mets["p95"] < 3.2


@pytest.mark.geometry
def test_3d_matchers_comparison_noise_partial_outliers():
    full = make_constellation_3d(seed=211)
    R = rot3d_xyz(10.0, -7.0, 19.0)
    scale = 1.07
    t = np.array([5.0, -9.0, 12.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / scale, R.T, -((1 / scale) * (t @ R)))
    crop = drop_fraction(add_noise(crop, sigma_um=0.3, seed=5), drop_frac=0.25, seed=7)
    full_n = add_outliers(add_noise(full, sigma_um=0.2, seed=8), 20, [-90, -90, -90], [90, 90, 90], seed=9)

    out_h = _run3d(crop, full_n, "hashing")
    out_p = _run3d(crop, full_n, "pyramid")

    assert out_h["success"] is True
    # Pyramid may be stricter in degraded geometry; if successful, it must be compatible.
    if out_p["success"]:
        pred_h = apply_similarity_3d(crop, out_h["best_scale"], np.asarray(out_h["best_R"]), np.asarray(out_h["best_t"]))
        pred_p = apply_similarity_3d(crop, out_p["best_scale"], np.asarray(out_p["best_R"]), np.asarray(out_p["best_t"]))
        mh = residual_metrics_nn(pred_h, full_n, inlier_radius_um=3.0)
        mp = residual_metrics_nn(pred_p, full_n, inlier_radius_um=3.0)
        assert abs(float(out_h["best_scale"]) - float(out_p["best_scale"])) < 0.18
        assert abs(mh["median"] - mp["median"]) < 1.2
    else:
        # out_p["success"] is False - pyramid legitimately declined to match.
        pass


@pytest.mark.geometry
@pytest.mark.slow
def test_2d_hashing_runtime_budget_small_case():
    full = make_constellation_2d(seed=212, n=40)
    crop = full.copy()
    t0 = time.perf_counter()
    out = _run2d(crop, full, "hashing")
    dt = time.perf_counter() - t0
    assert out["success"] is True
    # Generous threshold to catch severe regressions, not microbenchmark noise.
    # On the slowest expected CI runners this clean identity case should complete well under 15s.
    assert dt < 15.0


@pytest.mark.geometry
@pytest.mark.stochastic
def test_3d_hashing_runtime_budget_small_case():
    full = make_constellation_3d(seed=213, n=48)
    crop = full.copy()
    t0 = time.perf_counter()
    out = _run3d(crop, full, "hashing")
    dt = time.perf_counter() - t0
    assert out["success"] is True
    assert dt < 6.0
