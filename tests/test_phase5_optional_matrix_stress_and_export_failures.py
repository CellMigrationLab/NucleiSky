import json

import numpy as np
import pandas as pd
import pytest

from geometry_synth import (
    add_noise,
    add_outliers,
    apply_similarity_3d,
    make_constellation_3d,
    residual_metrics_nn,
    rot3d_xyz,
    rotation_error_deg_3d,
)
from nucleisky2d import segmentation as seg2d
from nucleisky2d.io import load_nucleisky_transform, save_nucleisky_transform
from nucleisky2d.pipeline import NucleiSky
from nucleisky3d.io import load_transforms_any_3d, save_nucleisky_transform_3d
from nucleisky3d.pipeline import NucleiSky3D


@pytest.mark.optional_backend
def test_optional_backend_instanseg_boundary(monkeypatch):
    pytest.importorskip("torch")
    called = {}

    def _fake_segment_instanseg(self, img, pixel_size_um, **kwargs):
        called["shape"] = tuple(np.asarray(img).shape)
        out = np.zeros_like(img, dtype=np.int32)
        out[2:5, 3:6] = 1
        return out

    monkeypatch.setattr(seg2d.Segmentor, "segment_instanseg", _fake_segment_instanseg, raising=True)
    img = np.zeros((12, 14), dtype=np.float32)
    labels = seg2d.segment_nuclei_dispatch(img, method="instanseg", pixel_size_um=0.5, settings={"instanseg": {}})
    assert called["shape"] == (12, 14)
    assert labels.dtype.kind in {"i", "u"}
    assert labels.max() >= 1


@pytest.mark.optional_backend
def test_optional_backend_cellpose_boundary(monkeypatch):
    pytest.importorskip("torch")
    called = {}

    def _fake_segment_cellpose(self, img2d, **kwargs):
        called["ndim"] = np.asarray(img2d).ndim
        out = np.zeros_like(img2d, dtype=np.int32)
        out[1:4, 1:4] = 3
        return out

    monkeypatch.setattr(seg2d.Segmentor, "segment_cellpose", _fake_segment_cellpose, raising=True)
    labels = seg2d.segment_nuclei_dispatch(np.zeros((8, 9), dtype=np.float32), method="cellpose", pixel_size_um=1.0, settings={"cellpose": {}})
    assert called["ndim"] == 2
    assert set(np.unique(labels)).issuperset({0, 3})


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "scenario",
    [
        {"name": "clean", "scale": 1.05, "rot": (8, -5, 12), "noise": 0.0, "drop": 0.0, "outliers": 0},
        {"name": "high_rot", "scale": 1.0, "rot": (42, -37, 55), "noise": 0.15, "drop": 0.10, "outliers": 6},
        {"name": "large_scale_partial", "scale": 1.7, "rot": (14, -9, 28), "noise": 0.2, "drop": 0.25, "outliers": 10},
        {"name": "sparse_near_deg", "scale": 1.2, "rot": (5, 2, 8), "noise": 0.2, "drop": 0.45, "outliers": 4},
    ],
)
def test_3d_matcher_stress_matrix_hashing_vs_pyramid(scenario):
    full = make_constellation_3d(seed=500, n=90)
    R = rot3d_xyz(*scenario["rot"])
    s = float(scenario["scale"])
    t = np.array([7.0, -6.0, 9.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))
    keep = int(round((1.0 - scenario["drop"]) * len(crop)))
    crop = crop[:max(keep, 8)]
    if scenario["noise"] > 0:
        crop = add_noise(crop, sigma_um=scenario["noise"], seed=501)
    full_use = full.copy()
    if scenario["outliers"] > 0:
        full_use = add_outliers(full_use, scenario["outliers"], [-150, -150, -150], [150, 150, 150], seed=502)

    def run(matcher):
        return NucleiSky3D(
            centroids_crop_um=crop,
            centroids_full_um=full_use,
            full_shape_px_zyx=(240, 260, 280),
            crop_shape_px_zyx=(90, 100, 100),
            pixel_size_full_um_zyx=(2.0, 0.9, 0.4),
            pixel_size_crop_um_zyx=(2.0, 0.9, 0.4),
            matcher=matcher,
            matcher_kwargs={"n_iters": 7000, "random_state": 17},
        )

    out_h = run("hashing")
    out_p = run("pyramid")
    for out in (out_h, out_p):
        if out["success"]:
            pred = apply_similarity_3d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
            m = residual_metrics_nn(pred, full_use, inlier_radius_um=3.0)
            assert m["median"] < 3.5
            assert out["match_quality"]["frac_inliers"] > 0.35
        else:
            assert out["match_quality"]["frac_inliers"] < 0.6

    # For successful recoveries, geometric plausibility should be retained.
    if out_h["success"]:
        assert abs(float(out_h["best_scale"]) - s) / s < 0.22
        assert rotation_error_deg_3d(np.asarray(out_h["best_R"]), R) < 20.0


@pytest.mark.integration
def test_2d_record_to_export_geometric_consistency(tmp_path):
    from geometry_synth import make_constellation_2d

    full = make_constellation_2d(seed=520, n=24)
    crop = full.copy()
    out = NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((100, 120), dtype=np.float32),
        img_crop=np.zeros((100, 120), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="hashing",
        matcher_kwargs={"n_iters": 2000, "random_state": 9},
    )
    rec = save_nucleisky_transform(out, tmp_path / "r2d.json", pixel_size_full_um=1.0, pixel_size_crop_um=1.0, require_success=True)
    loaded = load_nucleisky_transform(tmp_path / "r2d.json")
    p0 = np.asarray(rec["A_px"]) @ crop.T + np.asarray(rec["b_px"]).reshape(2, 1)
    p1 = np.asarray(loaded["A_px"]) @ crop.T + np.asarray(loaded["b_px"]).reshape(2, 1)
    np.testing.assert_allclose(p0, p1, atol=1e-8)


@pytest.mark.integration
def test_3d_export_failure_modes_and_extra_fields(tmp_path):
    ok = {
        "success": True,
        "best_scale": 1.0,
        "best_R": np.eye(3).tolist(),
        "best_t": [0.0, 0.0, 0.0],
        "match_quality": {"frac_inliers": 1.0, "mean_error_um": 0.0},
    }
    # Missing required rotation => clear failure from loader path when normalizing.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scale": 1.0, "t_um_zyx": [0, 0, 0]}), encoding="utf-8")
    with pytest.raises(Exception):
        load_transforms_any_3d(bad)

    rec = save_nucleisky_transform_3d(
        ok,
        tmp_path / "ok.json",
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        require_success=True,
    )
    rec["extra_field_unknown"] = "kept"
    p = tmp_path / "ok_extra.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    got = load_transforms_any_3d(p)[0]
    assert got["scale"] == 1.0
    assert got["R_zyx"] is not None
