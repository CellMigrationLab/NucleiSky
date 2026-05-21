import json

import numpy as np
import pandas as pd
import pytest

from geometry_synth import (
    apply_similarity_2d,
    apply_similarity_3d,
    make_constellation_2d,
    make_constellation_3d,
    residual_metrics_nn,
    rot2d,
    rot3d_xyz,
)
from nucleisky2d.io import load_nucleisky_transform, save_nucleisky_transform
from nucleisky2d.pipeline import NucleiSky, run_adaptive_matching_and_export
from nucleisky3d.io import load_transforms_any_3d
from nucleisky3d.pipeline import NucleiSky3D, run_adaptive_matching_and_export_3d


@pytest.mark.integration
def test_2d_partial_overlap_integration_and_transform_roundtrip(tmp_path):
    full = make_constellation_2d(seed=301, n=55)
    R = rot2d(17.0)
    scale = 1.1
    t = np.array([-6.0, 10.0], dtype=float)
    crop = apply_similarity_2d(full, 1 / scale, R.T, -((1 / scale) * (t @ R)))
    crop = crop[:38]  # realistic partial overlap

    out = NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((800, 800), dtype=np.float32),
        img_crop=np.zeros((300, 300), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=0.5,
        pixel_size_crop_um=0.5,
        matcher="hashing",
        matcher_kwargs={"n_iters": 8000, "random_state": 22, "inlier_radius_um": 1.8},
    )
    assert out["success"] is True
    pred = apply_similarity_2d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
    assert residual_metrics_nn(pred, full, inlier_radius_um=2.0)["median"] < 1.2

    p = tmp_path / "tx2d.json"
    rec = save_nucleisky_transform(
        out,
        p,
        pixel_size_full_um=0.5,
        pixel_size_crop_um=0.5,
        require_success=True,
    )
    rec2 = load_nucleisky_transform(p)
    assert {"scale", "R_yx", "t_um_yx", "A_px", "b_px"}.issubset(rec2.keys())
    np.testing.assert_allclose(np.asarray(rec["A_px"]), np.asarray(rec2["A_px"]))


@pytest.mark.integration
def test_3d_partial_overlap_anisotropic_integration_and_jsonl_roundtrip(tmp_path):
    full = make_constellation_3d(seed=302, n=70)
    R = rot3d_xyz(9.0, -8.0, 21.0)
    s = 0.96
    t = np.array([7.0, -5.0, 12.0], dtype=float)
    crop = apply_similarity_3d(full, 1 / s, R.T, -((1 / s) * (t @ R)))[:45]

    out = NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(180, 220, 240),
        crop_shape_px_zyx=(70, 90, 95),
        pixel_size_full_um_zyx=(2.0, 0.7, 0.5),
        pixel_size_crop_um_zyx=(2.0, 0.7, 0.5),
        matcher="hashing",
        matcher_kwargs={"n_iters": 8000, "random_state": 12, "inlier_radius_um": 2.2},
    )
    assert out["success"] is True
    pred = apply_similarity_3d(crop, out["best_scale"], np.asarray(out["best_R"]), np.asarray(out["best_t"]))
    assert residual_metrics_nn(pred, full, inlier_radius_um=2.6)["median"] < 1.8

    df_full = pd.DataFrame({"centroid_z_um": full[:, 0], "centroid_y_um": full[:, 1], "centroid_x_um": full[:, 2]})
    df_crop = pd.DataFrame({"centroid_z_um": crop[:, 0], "centroid_y_um": crop[:, 1], "centroid_x_um": crop[:, 2]})
    run_adaptive_matching_and_export_3d(
        df_full=df_full,
        df_crop=df_crop,
        pixel_size_full_orig_um_zyx=(2.0, 0.7, 0.5),
        pixel_size_crop_orig_um_zyx=(2.0, 0.7, 0.5),
        result_dir=str(tmp_path / "out3d"),
        save_segmentation_masks=False,
        verbose=False,
    )
    recs = load_transforms_any_3d(tmp_path / "out3d" / "matching" / "adaptive_3d" / "exports_adaptive" / "transforms.jsonl")
    assert len(recs) >= 1
    assert {"scale", "R_zyx", "t_um_zyx", "pixel_size_full_um_zyx", "pixel_size_crop_um_zyx"}.issubset(recs[0].keys())


@pytest.mark.integration
def test_metadata_realism_wrong_pixel_size_changes_pixel_space_interpretation(tmp_path):
    full = make_constellation_2d(seed=303, n=45)
    R = rot2d(12.0)
    scale = 1.0
    t = np.array([10.0, -4.0], dtype=float)
    crop = apply_similarity_2d(full, 1 / scale, R.T, -((1 / scale) * (t @ R)))

    out = NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((600, 600), dtype=np.float32),
        img_crop=np.zeros((300, 300), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=0.5,
        pixel_size_crop_um=0.5,
        matcher="hashing",
        matcher_kwargs={"n_iters": 7000, "random_state": 3},
    )
    p_good = tmp_path / "tx_good.json"
    p_bad = tmp_path / "tx_bad.json"
    rec_good = save_nucleisky_transform(
        out,
        p_good,
        pixel_size_full_um=0.5,
        pixel_size_crop_um=0.5,
        require_success=True,
    )
    rec_bad = save_nucleisky_transform(
        out,
        p_bad,
        pixel_size_full_um=0.5,
        pixel_size_crop_um=2.0,  # wrong metadata for crop spacing
        require_success=True,
    )
    assert out["success"] is True
    # Physical transform (um) is unchanged, but pixel-space mapping must differ.
    np.testing.assert_allclose(np.asarray(rec_good["t_um_yx"]), np.asarray(rec_bad["t_um_yx"]))
    assert np.linalg.norm(np.asarray(rec_good["A_px"]) - np.asarray(rec_bad["A_px"])) > 0.5


@pytest.mark.integration
def test_2d_adaptive_export_jsonl_schema_smoke(tmp_path):
    full = make_constellation_2d(seed=304, n=42)
    crop = full[:30].copy()
    df_full = pd.DataFrame({"centroid_y_um": full[:, 0], "centroid_x_um": full[:, 1]})
    df_crop = pd.DataFrame({"centroid_y_um": crop[:, 0], "centroid_x_um": crop[:, 1]})
    best, _ = run_adaptive_matching_and_export(
        df_full=df_full,
        df_crop=df_crop,
        img_full=np.zeros((400, 400), dtype=np.float32),
        img_crop=np.zeros((200, 200), dtype=np.float32),
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        result_dir=tmp_path / "adaptive2d",
        save_segmentation_masks=False,
    )
    assert isinstance(best, dict)
    line = (tmp_path / "adaptive2d" / "matching" / "adaptive" / "exports_adaptive" / "transforms.jsonl").read_text().strip().splitlines()[0]
    rec = json.loads(line)
    assert {"scale", "R_yx", "t_um_yx", "match_quality"}.issubset(rec.keys())
