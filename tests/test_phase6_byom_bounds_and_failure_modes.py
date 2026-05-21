import json

import numpy as np
import pandas as pd
import pytest

from geometry_synth import apply_similarity_2d, apply_similarity_3d, make_constellation_2d, make_constellation_3d, rot2d, rot3d_xyz
from nucleisky2d.features import extract_nuclear_features, extract_centroids_um
from nucleisky2d.io import load_nucleisky_transform, save_nucleisky_transform
from nucleisky2d.pipeline import NucleiSky
from nucleisky2d.preprocess import require_2d_label_mask
from nucleisky3d.features import extract_nuclear_features_3d
from nucleisky3d.io import load_transforms_any_3d
from nucleisky3d.matching.geometry import bbox_full_px_from_similarity_um_3d
from nucleisky3d.pipeline import NucleiSky3D
from nucleisky3d.preprocess import require_3d_label_mask


@pytest.mark.integration
def test_2d_record_bounds_consistent_with_transformed_extents(tmp_path):
    full = make_constellation_2d(seed=600, n=30)
    R = rot2d(21.0)
    s = 1.15
    t = np.array([6.0, -4.0], dtype=float)
    crop = apply_similarity_2d(full[:22], 1 / s, R.T, -((1 / s) * (t @ R)))
    out = NucleiSky(
        centroids_crop_um=crop,
        centroids_full_um=full,
        img_full=np.zeros((500, 600), dtype=np.float32),
        img_crop=np.zeros((220, 260), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=0.5,
        pixel_size_crop_um=0.5,
        matcher="hashing",
        matcher_kwargs={"n_iters": 6000, "random_state": 12},
    )
    rec = save_nucleisky_transform(out, tmp_path / "r2d.json", pixel_size_full_um=0.5, pixel_size_crop_um=0.5, require_success=True)
    got = load_nucleisky_transform(tmp_path / "r2d.json")
    from nucleisky2d.matching.geometry import bbox_full_px_from_similarity_um

    bbox_expected = bbox_full_px_from_similarity_um(
        crop_shape_px=(220, 260),
        pixel_size_full_um=0.5,
        pixel_size_crop_um=0.5,
        scale=float(got["scale"]),
        R_yx=np.asarray(got["R_yx"], dtype=float),
        t_um_yx=np.asarray(got["t_um_yx"], dtype=float),
        full_shape_px=(500, 600),
    )
    by0, by1, bx0, bx1 = [int(v) for v in got["bbox_full_px_y0y1x0x1"]]
    ey0, ey1, ex0, ex1 = [int(v) for v in bbox_expected]
    # Saved bbox may include extra safety margin; it must still contain computed extents.
    assert by0 <= ey0 <= ey1 <= by1
    assert bx0 <= ex0 <= ex1 <= bx1


@pytest.mark.integration
def test_3d_record_bbox_consistent_with_transformed_extents(tmp_path):
    full = make_constellation_3d(seed=601, n=50)
    R = rot3d_xyz(12.0, -10.0, 17.0)
    s = 1.05
    t = np.array([4.0, 8.0, -3.0], dtype=float)
    crop = apply_similarity_3d(full[:35], 1 / s, R.T, -((1 / s) * (t @ R)))
    out = NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(180, 220, 240),
        crop_shape_px_zyx=(80, 90, 100),
        pixel_size_full_um_zyx=(2.0, 0.8, 0.4),
        pixel_size_crop_um_zyx=(2.0, 0.8, 0.4),
        matcher="hashing",
        matcher_kwargs={"n_iters": 7000, "random_state": 21},
    )
    from nucleisky3d.io import save_nucleisky_transform_3d

    save_nucleisky_transform_3d(out, tmp_path / "r3d.json", pixel_size_full_um_zyx=(2.0, 0.8, 0.4), pixel_size_crop_um_zyx=(2.0, 0.8, 0.4), require_success=True)
    rec = load_transforms_any_3d(tmp_path / "r3d.json")[0]
    bbox = bbox_full_px_from_similarity_um_3d(
        crop_shape_px=(80, 90, 100),
        pixel_size_full_um_zyx=rec["pixel_size_full_um_zyx"],
        pixel_size_crop_um_zyx=rec["pixel_size_crop_um_zyx"],
        scale=rec["scale"],
        R_zyx=np.asarray(rec["R_zyx"]),
        t_um_zyx=np.asarray(rec["t_um_zyx"]),
        full_shape_px=(180, 220, 240),
    )
    rz0, rz1, ry0, ry1, rx0, rx1 = [int(v) for v in rec["bbox_full_px_z0z1y0y1x0x1"]]
    ez0, ez1, ey0, ey1, ex0, ex1 = [int(v) for v in bbox]
    assert rz0 <= ez0 <= ez1 <= rz1
    assert ry0 <= ey0 <= ey1 <= ry1
    assert rx0 <= ex0 <= ex1 <= rx1


@pytest.mark.geometry
def test_byom_2d_label_normalization_and_edge_cases():
    lbl = np.zeros((20, 20), dtype=np.int32)
    lbl[2:6, 2:6] = 5
    lbl[10:14, 10:14] = 9
    lbl[15:18, 3:6] = 20
    lbl[3:5, 12:15] = 9  # merged same id still one object id
    require_2d_label_mask(lbl, label="lbl", expected_shape=(20, 20))
    df = extract_nuclear_features(lbl, pixel_size_um=0.5, min_area_px=1)
    assert len(df) == 3  # non-contiguous ids preserve object count
    C = extract_centroids_um(df, name="df")
    assert C.shape[1] == 2
    with pytest.raises(ValueError):
        extract_nuclear_features(np.zeros((20, 20, 2), dtype=np.int32), pixel_size_um=1.0)
    # bool mask accepted by preprocessor contract as integer-like label mask
    require_2d_label_mask((lbl > 0), label="mask_bool", expected_shape=(20, 20))


@pytest.mark.geometry
def test_byom_3d_label_normalization_spacing_and_invalid_dims():
    lbl = np.zeros((8, 16, 16), dtype=np.int32)
    lbl[1:3, 2:6, 2:6] = 2
    lbl[4:6, 8:12, 10:14] = 10
    require_3d_label_mask(lbl, label="lbl3d", expected_shape=lbl.shape)
    sitk = pytest.importorskip("SimpleITK")
    _ = sitk
    df = extract_nuclear_features_3d(lbl, pixel_size_um=(2.0, 0.5, 0.25))
    assert len(df) == 2
    c = df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy()
    assert np.all(c[:, 0] > 0) and np.all(c[:, 1] > 0) and np.all(c[:, 2] > 0)
    with pytest.raises(ValueError):
        require_3d_label_mask(np.zeros((8, 16), dtype=np.int32), label="bad")


@pytest.mark.integration
def test_registration_failure_modes_from_bad_labels_or_features(tmp_path):
    # Empty extracted features -> explicit centroid extraction failure.
    empty_df = pd.DataFrame(columns=["centroid_y_um", "centroid_x_um"])
    with pytest.raises(ValueError):
        extract_centroids_um(empty_df, name="empty_df")

    # Too few/degenerate centroids must not silently report high-confidence success.
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    out = NucleiSky(
        centroids_crop_um=pts,
        centroids_full_um=pts,
        img_full=np.zeros((30, 30), dtype=np.float32),
        img_crop=np.zeros((20, 20), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="hashing",
        matcher_kwargs={"n_iters": 1500, "random_state": 1},
    )
    assert out["success"] is False

    bad = {"scale": 1.0, "R_zyx": [[1, 0], [0, 1]], "t_um_zyx": [0, 0, 0]}
    p = tmp_path / "phase6_bad_dim.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(bad, f)
    with pytest.raises(Exception):
        load_transforms_any_3d(p)
