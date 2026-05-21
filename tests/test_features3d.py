import numpy as np
import pytest

from nucleisky3d import features
from nucleisky3d.features import extract_nuclear_features_3d

sitk = pytest.importorskip("SimpleITK")


def _single_label_cube():
    label_img = np.zeros((4, 4, 4), dtype=int)
    label_img[1:3, 1:3, 1:3] = 1
    return label_img


def test_anisotropic_volume_and_surface_area():
    label_img = _single_label_cube()
    anisotropic = (2.0, 0.5, 0.5)

    df = extract_nuclear_features_3d(label_img, pixel_size_um=anisotropic)

    assert df["volume_voxels"].iloc[0] == 8
    assert df["volume_um3"].iloc[0] == 8 * 2.0 * 0.5 * 0.5

    df_iso = extract_nuclear_features_3d(label_img, pixel_size_um=(1.0, 0.5, 0.5))
    assert not np.isclose(df["surface_area_um2"].iloc[0], df_iso["surface_area_um2"].iloc[0])


def test_extract_features_raises_clear_error_when_simpleitk_missing(monkeypatch):
    monkeypatch.setattr(features, "sitk", None)

    with pytest.raises(ImportError, match="SimpleITK is required"):
        extract_nuclear_features_3d(_single_label_cube(), pixel_size_um=1.0)


def test_extract_features_adds_knn_distance_columns():
    label_img = np.zeros((6, 6, 6), dtype=int)
    label_img[1:3, 1:3, 1:3] = 1
    label_img[1:3, 1:3, 4:6] = 2
    label_img[3:5, 3:5, 1:3] = 3

    df = extract_nuclear_features_3d(label_img, pixel_size_um=1.0, k_neighbors=3)

    for col in ("nn1_dist_um", "nn2_dist_um", "nn3_dist_um"):
        assert col in df.columns

    assert np.all(np.isfinite(df["nn1_dist_um"].to_numpy()))
    assert np.all(df["nn1_dist_um"].to_numpy() > 0)
    assert df["nn3_dist_um"].isna().all()
