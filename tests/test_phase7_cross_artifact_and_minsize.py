import json

import numpy as np
import pandas as pd
import pytest

from artifact_consistency import assert_2d_record_and_artifact_consistent, assert_3d_record_and_artifact_consistent
from geometry_synth import apply_similarity_2d, apply_similarity_3d, make_constellation_2d, make_constellation_3d, rot2d, rot3d_xyz
from nucleisky2d.features import extract_nuclear_features, extract_centroids_um
from nucleisky2d.export import export_aligned_dataset
from nucleisky2d.pipeline import NucleiSky
from nucleisky3d.export import export_aligned_crop_tiff
from nucleisky3d.pipeline import NucleiSky3D
from nucleisky3d.segmentation import segment_nuclei_2p5d


@pytest.mark.integration
def test_2d_export_bbox_matches_exported_roi_dimensions(tmp_path):
    full = make_constellation_2d(seed=700, n=44)
    R = rot2d(13.0)
    crop_pts = apply_similarity_2d(full[:30], 1.0, R.T, np.array([4.0, -3.0]))
    out = NucleiSky(
        centroids_crop_um=crop_pts,
        centroids_full_um=full,
        img_full=np.zeros((240, 320), dtype=np.float32),
        img_crop=np.zeros((120, 140), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="hashing",
        matcher_kwargs={"n_iters": 5000, "random_state": 10},
    )
    from nucleisky2d.io import save_nucleisky_transform
    rec = save_nucleisky_transform(out, tmp_path / "r2d.json", pixel_size_full_um=1.0, pixel_size_crop_um=1.0, require_success=True)
    out_paths = export_aligned_dataset(
        rec,
        out_dir=tmp_path / "exp2d",
        img_full=np.zeros((240, 320), dtype=np.float32),
        img_crop=np.zeros((120, 140), dtype=np.float32),
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        export_region="roi",
    )
    assert_2d_record_and_artifact_consistent(rec, out_paths["aligned_on_full_px"])


@pytest.mark.integration
def test_3d_export_bbox_matches_exported_volume_dimensions(tmp_path):
    full = make_constellation_3d(seed=701, n=56)
    R = rot3d_xyz(10.0, -8.0, 15.0)
    crop_pts = apply_similarity_3d(full[:36], 1.0, R.T, np.array([3.0, 6.0, -2.0]))
    out = NucleiSky3D(
        centroids_crop_um=crop_pts,
        centroids_full_um=full,
        full_shape_px_zyx=(70, 90, 100),
        crop_shape_px_zyx=(25, 30, 35),
        pixel_size_full_um_zyx=(2.0, 0.8, 0.4),
        pixel_size_crop_um_zyx=(2.0, 0.8, 0.4),
        matcher="hashing",
        matcher_kwargs={"n_iters": 7000, "random_state": 11},
    )
    from nucleisky3d.io import save_nucleisky_transform_3d
    rec = save_nucleisky_transform_3d(out, tmp_path / "r3d.json", pixel_size_full_um_zyx=(2.0, 0.8, 0.4), pixel_size_crop_um_zyx=(2.0, 0.8, 0.4), require_success=True)
    p = export_aligned_crop_tiff(
        img_full=np.zeros((70, 90, 100), dtype=np.float32),
        img_crop=np.zeros((25, 30, 35), dtype=np.float32),
        output_path=tmp_path / "roi3d.tif",
        pixel_size_full_um=(2.0, 0.8, 0.4),
        pixel_size_crop_um=(2.0, 0.8, 0.4),
        res=out,
        export_region="bbox",
    )
    assert_3d_record_and_artifact_consistent(rec, p)
    assert rec["pixel_size_full_um_zyx"] == [2.0, 0.8, 0.4]


@pytest.mark.geometry
def test_2d_minsize_filtering_above_below_exact_threshold_and_empty():
    lbl = np.zeros((20, 20), dtype=np.int32)
    lbl[1:3, 1:3] = 1      # area 4
    lbl[5:8, 5:8] = 2      # area 9
    lbl[10:14, 10:14] = 5  # area 16
    # min_area_px exact threshold retains area==9 and above
    df = extract_nuclear_features(lbl, pixel_size_um=1.0, min_area_px=9)
    assert len(df) == 2
    # below-threshold all removed -> documented empty-feature contract
    df_empty = extract_nuclear_features(lbl, pixel_size_um=1.0, min_area_px=100)
    assert df_empty.empty


@pytest.mark.geometry
def test_3d_minsize_filtering_via_threshold_segmentation_and_anisotropy():
    vol = np.zeros((5, 20, 20), dtype=np.float32)
    vol[:, 2:4, 2:4] = 1.0          # tiny object
    vol[:, 8:14, 8:14] = 1.0        # large object
    labels = segment_nuclei_2p5d(
        volume_zyx=vol,
        method="threshold",
        pixel_size_um_zyx=(2.0, 0.5, 0.5),
        settings={"threshold": {"threshold_method": "otsu", "min_object_size": 20, "do_watershed": False}},
        show_progress=False,
    )
    # Tiny object filtered, large retained.
    assert labels.max() >= 1
    zyx = np.argwhere(labels > 0)
    assert zyx[:, 1].min() >= 8


@pytest.mark.integration
def test_label_filtering_to_registration_success_and_failure_paths():
    lbl_full = np.zeros((30, 30), dtype=np.int32)
    lbl_crop = np.zeros((30, 30), dtype=np.int32)
    lbl_full[2:8, 2:8] = 1
    lbl_full[12:18, 12:18] = 2
    lbl_full[20:26, 6:12] = 3
    lbl_crop[2:8, 2:8] = 1
    lbl_crop[12:18, 12:18] = 2
    lbl_crop[20:26, 6:12] = 3
    # Success path: enough objects after filtering.
    df_full = extract_nuclear_features(lbl_full, pixel_size_um=1.0, min_area_px=10)
    df_crop = extract_nuclear_features(lbl_crop, pixel_size_um=1.0, min_area_px=10)
    c_full = extract_centroids_um(df_full, name="full")
    c_crop = extract_centroids_um(df_crop, name="crop")
    out = NucleiSky(
        centroids_crop_um=c_crop,
        centroids_full_um=c_full,
        img_full=np.zeros((30, 30), dtype=np.float32),
        img_crop=np.zeros((30, 30), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="triangles",
        matcher_kwargs={"n_iters": 2000, "random_state": 2},
    )
    assert out["success"] is True
    # Failure path: too strict filtering -> no features.
    df_empty = extract_nuclear_features(lbl_crop, pixel_size_um=1.0, min_area_px=100)
    assert df_empty.empty


@pytest.mark.integration
def test_export_failure_mode_missing_spacing_metadata_in_3d_record(tmp_path):
    bad = {
        "scale": 1.0,
        "R_zyx": np.eye(3).tolist(),
        "t_um_zyx": [0.0, 0.0, 0.0],
        # missing pixel_size_full_um_zyx / pixel_size_crop_um_zyx
    }
    p = tmp_path / "bad_missing_voxel.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    from nucleisky3d.io import load_transforms_any_3d

    with pytest.raises(Exception):
        load_transforms_any_3d(p)
