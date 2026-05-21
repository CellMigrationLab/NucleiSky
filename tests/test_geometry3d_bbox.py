import numpy as np
import pytest

from nucleisky3d.matching.geometry import (
    bbox_add_margin_px_3d,
    bbox_full_px_from_similarity_um_3d,
    estimate_dynamic_scale_bounds_3d,
    sanitize_points_zyx_um,
)
from nucleisky3d.types import BBox3D


def test_bbox3d_casts_validates_and_iterates():
    bbox = BBox3D(1.2, 5.9, 2, 7, 3.0, 9)
    assert tuple(bbox) == (1, 5, 2, 7, 3, 9)
    assert bbox.slices_zyx() == (slice(1, 5), slice(2, 7), slice(3, 9))

    with pytest.raises(ValueError, match="Invalid BBox3D"):
        BBox3D(2, 1, 0, 1, 0, 1)


def test_bbox3d_clamp_enforces_bounds_and_min_size():
    bbox = BBox3D(-5, -2, 15, 16, 2, 2)
    got = bbox.clamp((10, 10, 10), min_size=2)
    assert got == BBox3D(0, 2, 8, 10, 2, 4)


def test_bbox_from_similarity_returns_bbox3d_and_clamps_to_full_shape():
    got = bbox_full_px_from_similarity_um_3d(
        crop_shape_px=(5, 6, 7),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        scale=1.0,
        R_zyx=np.eye(3),
        t_um_zyx=(-10.0, -10.0, -10.0),
        full_shape_px=(4, 4, 4),
    )
    assert isinstance(got, BBox3D)
    assert tuple(got) == (0, 1, 0, 1, 0, 1)


def test_bbox_add_margin_px_3d_expands_correctly():
    bbox = (10, 20, 30, 40, 50, 60)
    got = bbox_add_margin_px_3d(bbox, margin_px=(2, 3, 4))
    assert got == (8, 22, 27, 43, 46, 64)


def test_bbox_add_margin_px_3d_clamps_to_shape():
    bbox = (1, 4, 2, 5, 3, 6)
    got = bbox_add_margin_px_3d(bbox, margin_px=(5, 5, 5), shape_zyx=(6, 7, 8))
    assert got == (0, 6, 0, 7, 0, 8)


def test_bbox_add_margin_px_3d_scalar_margin_applies_to_all_axes():
    bbox = (3, 8, 10, 20, 30, 35)
    got = bbox_add_margin_px_3d(bbox, margin_px=2)
    assert got == (1, 10, 8, 22, 28, 37)


def test_dynamic_scale_bounds_falls_back_to_centroid_nn_when_nn_columns_missing():
    df_full = {
        "centroid_z_um": [0.0, 10.0, 20.0],
        "centroid_y_um": [0.0, 0.0, 0.0],
        "centroid_x_um": [0.0, 0.0, 0.0],
    }
    df_crop = {
        "centroid_z_um": [0.0, 5.0, 10.0],
        "centroid_y_um": [0.0, 0.0, 0.0],
        "centroid_x_um": [0.0, 0.0, 0.0],
    }

    import pandas as pd

    scale_prior, _, _ = estimate_dynamic_scale_bounds_3d(
        pd.DataFrame(df_full),
        pd.DataFrame(df_crop),
        voxel_size_full_um_zyx=(1.0, 1.0, 1.0),
        voxel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        full_shape_px_zyx=(100, 100, 100),
        crop_shape_px_zyx=(100, 100, 100),
        coarse_scale_min=0.5,
        coarse_scale_max=3.0,
        rel_tol=0.2,
    )

    assert np.isclose(scale_prior, 2.0, atol=1e-6)


def test_sanitize_points_removes_nonfinite_rows():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, np.nan, 1.0],
            [2.0, 2.0, 2.0],
            [np.inf, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ]
    )

    sanitized, stats = sanitize_points_zyx_um(pts, min_points=3, name="pts")

    assert sanitized.shape == (3, 3)
    assert stats["n_in"] == 5
    assert stats["n_finite"] == 3


def test_sanitize_points_removes_exact_duplicates_with_dedup_radius():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            [10.0, 10.0, 10.0],
            [15.0, 15.0, 15.0],
        ]
    )

    sanitized, stats = sanitize_points_zyx_um(pts, dedup_radius_um=0.1, min_points=4)

    assert sanitized.shape == (4, 3)
    assert stats["n_deduped"] == 1


def test_sanitize_points_drops_high_nn_distance_outlier():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [100.0, 100.0, 100.0],
        ]
    )

    sanitized, stats = sanitize_points_zyx_um(
        pts,
        nn_outlier_percentile=80.0,
        min_points=4,
    )

    assert sanitized.shape == (4, 3)
    assert stats["n_outliers_dropped"] == 1
    assert not np.any(np.all(np.isclose(sanitized, [100.0, 100.0, 100.0]), axis=1))
