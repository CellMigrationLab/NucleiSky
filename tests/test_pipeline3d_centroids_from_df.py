import numpy as np
import pandas as pd
import pytest

from nucleisky3d.pipeline import centroids_from_df_3d


def test_centroids_from_df_3d_um_columns_ignore_voxel_size():
    df = pd.DataFrame(
        {
            "centroid_z_um": [1.0, 2.5],
            "centroid_y_um": [3.0, 4.5],
            "centroid_x_um": [5.0, 6.5],
        }
    )

    got = centroids_from_df_3d(df, voxel_size_um_zyx=(9.0, 9.0, 9.0), name="df_um")

    expected = np.array([[1.0, 3.0, 5.0], [2.5, 4.5, 6.5]], dtype=np.float32)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected)


def test_centroids_from_df_3d_px_columns_require_voxel_size_and_convert():
    df = pd.DataFrame(
        {
            "centroid_z_px": [1.0, 2.0],
            "centroid_y_px": [3.0, 4.0],
            "centroid_x_px": [5.0, 6.0],
        }
    )

    got = centroids_from_df_3d(df, voxel_size_um_zyx=(2.0, 0.5, 0.25), name="df_px")

    expected = np.array([[2.0, 1.5, 1.25], [4.0, 2.0, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(got, expected)


def test_centroids_from_df_3d_px_columns_without_voxel_size_raises():
    df = pd.DataFrame(
        {
            "centroid_z_px": [1.0],
            "centroid_y_px": [2.0],
            "centroid_x_px": [3.0],
        }
    )

    with pytest.raises(ValueError, match="voxel_size_um_zyx is required"):
        centroids_from_df_3d(df, name="df_px")


def test_centroids_from_df_3d_missing_columns_raises_informative_error():
    df = pd.DataFrame({"centroid_z_px": [1.0], "centroid_y_px": [2.0]})

    with pytest.raises(ValueError, match="must contain either centroid columns"):
        centroids_from_df_3d(df, voxel_size_um_zyx=(1.0, 1.0, 1.0), name="df_bad")
