import json

import numpy as np
import pandas as pd
import pytest

from geometry_synth import (
    add_noise,
    add_outliers,
    apply_similarity_2d,
    apply_similarity_3d,
    make_constellation_2d,
    make_constellation_3d,
    residual_metrics_nn,
    rot2d,
    rot3d_xyz,
    rotation_error_deg_3d,
)
from nucleisky2d.io import save_nucleisky_transform
from nucleisky2d.pipeline import NucleiSky, run_adaptive_matching_and_export
from nucleisky3d.features import extract_nuclear_features_3d
from nucleisky3d.io import load_transforms_any_3d, save_nucleisky_transform_3d
from nucleisky3d.pipeline import NucleiSky3D


@pytest.mark.integration
def test_3d_roundtrip_point_mapping_equivalence(tmp_path):
    full = make_constellation_3d(seed=401, n=64)
    R = rot3d_xyz(13.0, -9.0, 26.0)
    s = 1.09
    t = np.array([9.0, -7.0, 5.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))

    out = NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(200, 220, 240),
        crop_shape_px_zyx=(90, 100, 110),
        pixel_size_full_um_zyx=(1.2, 0.6, 0.4),
        pixel_size_crop_um_zyx=(1.2, 0.6, 0.4),
        matcher="hashing",
        matcher_kwargs={"n_iters": 8000, "random_state": 7},
    )
    rec = save_nucleisky_transform_3d(
        out,
        tmp_path / "rt3d.json",
        pixel_size_full_um_zyx=(1.2, 0.6, 0.4),
        pixel_size_crop_um_zyx=(1.2, 0.6, 0.4),
        require_success=True,
    )
    loaded = load_transforms_any_3d(tmp_path / "rt3d.json")[0]

    src = crop[:20]
    p0 = apply_similarity_3d(src, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
    p1 = apply_similarity_3d(src, loaded["scale"], np.asarray(loaded["R_zyx"]), np.asarray(loaded["t_um_zyx"]))
    np.testing.assert_allclose(p0, p1, atol=1e-6)
    assert rec["pixel_size_full_um_zyx"] == loaded["pixel_size_full_um_zyx"]
    assert rec["pixel_size_crop_um_zyx"] == loaded["pixel_size_crop_um_zyx"]


@pytest.mark.integration
def test_3d_loader_legacy_and_current_schema_compatibility(tmp_path):
    base = {
        "scale": 1.03,
        "R_zyx": np.eye(3).tolist(),
        "t_um_zyx": [1.0, 2.0, 3.0],
        "pixel_size_full_um_zyx": [1.0, 0.5, 0.5],
        "pixel_size_crop_um_zyx": [1.0, 0.5, 0.5],
        "match_quality": {"frac_inliers": 0.8, "mean_error_um": 0.4},
    }
    legacy = {
        "best_scale": 1.03,
        "best_R": np.eye(3).tolist(),
        "best_t": [1.0, 2.0, 3.0],
        "pixel_size_full_orig_um_zyx": [1.0, 0.5, 0.5],
        "pixel_size_crop_orig_um_zyx": [1.0, 0.5, 0.5],
        "success": True,
        "extra_unknown_field": 123,
    }
    p = tmp_path / "multi.jsonl"
    p.write_text(json.dumps(base) + "\n" + json.dumps(legacy) + "\n", encoding="utf-8")
    recs = load_transforms_any_3d(p)
    assert len(recs) == 2
    src = np.array([[0.2, 1.3, -0.7], [2.0, 3.0, 4.0]], dtype=float)
    a = apply_similarity_3d(src, recs[0]["scale"], np.asarray(recs[0]["R_zyx"]), np.asarray(recs[0]["t_um_zyx"]))
    b = apply_similarity_3d(src, recs[1]["scale"], np.asarray(recs[1]["R_zyx"]), np.asarray(recs[1]["t_um_zyx"]))
    np.testing.assert_allclose(a, b, atol=1e-9)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scale": 1.0}), encoding="utf-8")
    with pytest.raises(Exception):
        load_transforms_any_3d(bad)


@pytest.mark.geometry
@pytest.mark.parametrize("ang", [0.0, 90.0, 179.0, 271.0])
def test_2d_hashing_rotation_coverage_and_extreme_scale(ang):
    full = make_constellation_2d(seed=402, n=50)
    scale = 1.8
    R = rot2d(ang)
    t = np.array([4.0, -6.0], dtype=float)
    crop = apply_similarity_2d(full, 1 / scale, R.T, -((1 / scale) * (t @ R)))
    crop = np.concatenate([crop[:35], add_outliers(crop[35:], 8, [-130, -130], [130, 130], seed=8)], axis=0)
    out = NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((900, 900), dtype=np.float32),
        img_crop=np.zeros((400, 400), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="hashing",
        matcher_kwargs={"n_iters": 9000, "random_state": 4, "inlier_radius_um": 2.2},
    )
    assert out["success"] is True
    # Outlier-heavy scenario: use quality metrics rather than raw p95 over injected outliers.
    assert out["match_quality"]["frac_inliers"] > 0.75


@pytest.mark.geometry
def test_3d_matchers_extreme_scale_compound_rotation_anisotropic_partial_overlap():
    full = make_constellation_3d(seed=403, n=80)
    R = rot3d_xyz(22.0, -18.0, 31.0)
    s = 1.65
    t = np.array([6.0, 11.0, -9.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))
    crop = add_noise(crop[:60], sigma_um=0.25, seed=9)
    full_o = add_outliers(full, 12, [-120, -120, -120], [120, 120, 120], seed=10)
    out_h = NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full_o,
        full_shape_px_zyx=(260, 280, 300),
        crop_shape_px_zyx=(100, 110, 120),
        pixel_size_full_um_zyx=(2.0, 0.8, 0.4),
        pixel_size_crop_um_zyx=(2.0, 0.8, 0.4),
        matcher="hashing",
        matcher_kwargs={"n_iters": 9000, "random_state": 9},
    )
    # In this stress regime (large scale + anisotropy + outliers), matcher must either recover
    # a plausible transform or fail explicitly (not silently claim high confidence).
    if out_h["success"]:
        assert abs(float(out_h["best_scale"]) - s) / s < 0.18
        assert rotation_error_deg_3d(np.asarray(out_h["best_R"]), R) < 14.0
    else:
        assert out_h["match_quality"]["frac_inliers"] < 0.5


@pytest.mark.integration
@pytest.mark.optional_backend
def test_simpleitk_features_to_registration_smoke():
    sitk = pytest.importorskip("SimpleITK")
    _ = sitk  # keep linter quiet
    labels = np.zeros((8, 20, 20), dtype=np.int32)
    labels[1:3, 2:6, 2:6] = 1
    labels[3:5, 10:14, 4:8] = 2
    labels[5:7, 6:10, 12:16] = 3
    labels[2:4, 13:17, 13:17] = 4
    df = extract_nuclear_features_3d(labels, pixel_size_um=(1.0, 0.5, 0.5))
    pts = df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
    out = NucleiSky3D(
        centroids_crop_um=pts,
        centroids_full_um=pts,
        full_shape_px_zyx=labels.shape,
        crop_shape_px_zyx=labels.shape,
        pixel_size_full_um_zyx=(1.0, 0.5, 0.5),
        pixel_size_crop_um_zyx=(1.0, 0.5, 0.5),
        matcher="hashing",
    )
    assert out["success"] is True


@pytest.mark.integration
def test_2d_qc_artifact_consistency(tmp_path):
    full = make_constellation_2d(seed=404, n=40)
    R = rot2d(14.0)
    crop = apply_similarity_2d(full[:30], 1.0, R.T, np.array([3.0, -2.0]))
    out = NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((300, 400), dtype=np.float32),
        img_crop=np.zeros((120, 160), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="hashing",
        matcher_kwargs={"n_iters": 6000, "random_state": 4},
    )
    rec = save_nucleisky_transform(out, tmp_path / "r2d.json", pixel_size_full_um=1.0, pixel_size_crop_um=1.0, require_success=True)
    df_full = pd.DataFrame({"centroid_y_um": full[:, 0], "centroid_x_um": full[:, 1]})
    df_crop = pd.DataFrame({"centroid_y_um": crop[:, 0], "centroid_x_um": crop[:, 1]})
    run_adaptive_matching_and_export(
        df_full=df_full,
        df_crop=df_crop,
        img_full=np.zeros((300, 400), dtype=np.float32),
        img_crop=np.zeros((120, 160), dtype=np.float32),
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        result_dir=tmp_path / "qc_out",
        save_segmentation_masks=False,
    )
    tf = tmp_path / "qc_out" / "matching" / "adaptive" / "exports_adaptive" / "transforms.jsonl"
    assert tf.exists()
    rec2 = json.loads(tf.read_text().strip().splitlines()[0])
    assert {"scale", "R_yx", "t_um_yx", "success"}.issubset(rec2.keys())
    if rec2["success"]:
        assert rec2["match_quality"]["frac_inliers"] > 0.5
