import json

import numpy as np
import pandas as pd
import pytest

from artifact_consistency import assert_2d_record_and_artifact_consistent, assert_3d_record_and_artifact_consistent
from geometry_synth import apply_similarity_2d, apply_similarity_3d, make_constellation_2d, make_constellation_3d, rot2d, rot3d_xyz
from nucleisky2d.export import export_aligned_dataset
from nucleisky2d.io import load_nucleisky_transform, save_nucleisky_transform
from nucleisky2d.pipeline import NucleiSky, run_adaptive_matching_and_export
from nucleisky3d.export import export_aligned_crop_tiff
from nucleisky3d.io import load_transforms_any_3d, save_nucleisky_transform_3d
from nucleisky3d.pipeline import NucleiSky3D


@pytest.mark.integration
def test_2d_additional_adaptive_export_consistency_with_helper(tmp_path):
    full = make_constellation_2d(seed=800, n=50)
    R = rot2d(-17.0)
    crop = apply_similarity_2d(full[:34], 1.0, R.T, np.array([7.0, 2.0]))
    df_full = pd.DataFrame({"centroid_y_um": full[:, 0], "centroid_x_um": full[:, 1]})
    df_crop = pd.DataFrame({"centroid_y_um": crop[:, 0], "centroid_x_um": crop[:, 1]})
    best, _ = run_adaptive_matching_and_export(
        df_full=df_full,
        df_crop=df_crop,
        img_full=np.zeros((260, 300), dtype=np.float32),
        img_crop=np.zeros((120, 140), dtype=np.float32),
        pixel_size_full_um=1.2,
        pixel_size_crop_um=1.2,
        result_dir=tmp_path / "a2d",
        save_segmentation_masks=False,
    )
    rec = json.loads((tmp_path / "a2d" / "matching" / "adaptive" / "exports_adaptive" / "transforms.jsonl").read_text().strip().splitlines()[0])
    out_paths = export_aligned_dataset(
        best,
        out_dir=tmp_path / "a2d_exp",
        img_full=np.zeros((260, 300), np.float32),
        img_crop=np.zeros((120, 140), np.float32),
        pixel_size_full_um=1.2,
        pixel_size_crop_um=1.2,
        export_region="roi",
        margin_px=0,
        bbox_full_px=rec["bbox_full_px_y0y1x0x1"],
    )
    assert_2d_record_and_artifact_consistent(rec, out_paths["aligned_on_full_px"])


@pytest.mark.integration
def test_3d_additional_export_consistency_with_helper(tmp_path):
    full = make_constellation_3d(seed=801, n=54)
    R = rot3d_xyz(9.0, 7.0, -13.0)
    crop = apply_similarity_3d(full[:36], 1.0, R.T, np.array([3.0, -5.0, 4.0]))
    out = NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(80, 90, 110),
        crop_shape_px_zyx=(24, 26, 30),
        pixel_size_full_um_zyx=(1.7, 0.9, 0.5),
        pixel_size_crop_um_zyx=(1.7, 0.9, 0.5),
        matcher="hashing",
        matcher_kwargs={"n_iters": 6000, "random_state": 33},
    )
    rec = save_nucleisky_transform_3d(out, tmp_path / "r3d.json", pixel_size_full_um_zyx=(1.7, 0.9, 0.5), pixel_size_crop_um_zyx=(1.7, 0.9, 0.5), require_success=True)
    p = export_aligned_crop_tiff(
        img_full=np.zeros((80, 90, 110), dtype=np.float32),
        img_crop=np.zeros((24, 26, 30), dtype=np.float32),
        output_path=tmp_path / "roi3d.tif",
        pixel_size_full_um=(1.7, 0.9, 0.5),
        pixel_size_crop_um=(1.7, 0.9, 0.5),
        res=out,
        export_region="bbox",
    )
    assert_3d_record_and_artifact_consistent(rec, p)


@pytest.mark.geometry
def test_2d_unit_conversion_invariant_save_load_affine_equivalence(tmp_path):
    full = make_constellation_2d(seed=802, n=28)
    crop = full.copy()
    out = NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((90, 100), dtype=np.float32),
        img_crop=np.zeros((90, 100), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=0.4,
        pixel_size_crop_um=1.2,
        matcher="hashing",
    )
    rec = save_nucleisky_transform(out, tmp_path / "r2d.json", pixel_size_full_um=0.4, pixel_size_crop_um=1.2, require_success=True)
    rec2 = load_nucleisky_transform(tmp_path / "r2d.json")
    pts = crop[:8]
    m0 = (np.asarray(rec["A_px"]) @ pts.T + np.asarray(rec["b_px"]).reshape(2, 1)).T
    m1 = (np.asarray(rec2["A_px"]) @ pts.T + np.asarray(rec2["b_px"]).reshape(2, 1)).T
    np.testing.assert_allclose(m0, m1, atol=1e-8)

    rec_badpix = save_nucleisky_transform(out, tmp_path / "r2d_badpix.json", pixel_size_full_um=0.4, pixel_size_crop_um=0.8, require_success=True)
    # Changing crop pixel size must change pixel-space affine while physical transform stays same.
    np.testing.assert_allclose(rec["t_um_yx"], rec_badpix["t_um_yx"])
    assert np.linalg.norm(np.asarray(rec["A_px"]) - np.asarray(rec_badpix["A_px"])) > 1e-3


@pytest.mark.geometry
def test_3d_unit_conversion_invariant_anisotropic_spacing_roundtrip(tmp_path):
    full = make_constellation_3d(seed=803, n=40)
    crop = full[:25]
    out = NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(70, 80, 90),
        crop_shape_px_zyx=(20, 22, 24),
        pixel_size_full_um_zyx=(2.0, 0.7, 0.3),
        pixel_size_crop_um_zyx=(2.0, 0.7, 0.3),
        matcher="hashing",
        matcher_kwargs={"n_iters": 5000, "random_state": 4},
    )
    save_nucleisky_transform_3d(out, tmp_path / "r3d.json", pixel_size_full_um_zyx=(2.0, 0.7, 0.3), pixel_size_crop_um_zyx=(2.0, 0.7, 0.3), require_success=True)
    rec = load_transforms_any_3d(tmp_path / "r3d.json")[0]
    mapped = apply_similarity_3d(crop, rec["scale"], np.asarray(rec["R_zyx"]), np.asarray(rec["t_um_zyx"]))
    assert mapped.shape[1] == 3
    assert np.asarray(rec["pixel_size_full_um_zyx"]).tolist() == [2.0, 0.7, 0.3]


@pytest.mark.integration
def test_malformed_unit_metadata_failures(tmp_path):
    # zero/negative or wrong-length spacing should fail helper/loader expectations.
    bads = [
        {"scale": 1.0, "R_zyx": np.eye(3).tolist(), "t_um_zyx": [0, 0, 0], "pixel_size_full_um_zyx": [0, 0.5, 0.5], "pixel_size_crop_um_zyx": [1, 0.5, 0.5]},
        {"scale": 1.0, "R_zyx": np.eye(3).tolist(), "t_um_zyx": [0, 0, 0], "pixel_size_full_um_zyx": [-1, 0.5, 0.5], "pixel_size_crop_um_zyx": [1, 0.5, 0.5]},
        {"scale": 1.0, "R_zyx": np.eye(3).tolist(), "t_um_zyx": [0, 0, 0], "pixel_size_full_um_zyx": [1, 0.5], "pixel_size_crop_um_zyx": [1, 0.5, 0.5]},
    ]
    for i, rec in enumerate(bads):
        p = tmp_path / f"bad{i}.json"
        p.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(Exception):
            load_transforms_any_3d(p)
