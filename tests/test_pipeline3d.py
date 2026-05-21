import numpy as np
import pandas as pd
import pytest

pytest.importorskip("SimpleITK")

from nucleisky3d import pipeline
from nucleisky3d.visualization import _mip


def _dummy_matcher(**kwargs):
    return 1.0, np.eye(3, dtype=float), np.zeros(3, dtype=float), (0, 1, 0, 1, 0, 1)


def test_nucleisky3d_uses_precomputed_dfs_without_reextracting(monkeypatch):
    labels = np.zeros((4, 4, 4), dtype=np.int32)

    df_full = pd.DataFrame(
        {
            "centroid_z_px": [1.0, 1.0, 2.0, 2.0],
            "centroid_y_px": [1.0, 2.0, 1.0, 2.0],
            "centroid_x_px": [1.0, 2.0, 2.0, 1.0],
        }
    )
    df_crop = df_full.copy()

    def _no_extract(*args, **kwargs):
        raise AssertionError("feature extraction should be skipped when df_full/df_crop are provided")

    monkeypatch.setattr(pipeline, "_extract_features_and_centroids", _no_extract)
    monkeypatch.setattr(pipeline, "run_pyramid_based_matching_um", _dummy_matcher)

    out = pipeline.NucleiSky3D(
        label_full=labels,
        label_crop=labels,
        pixel_size_full_um=(1.0, 1.0, 1.0),
        pixel_size_crop_um=(1.0, 1.0, 1.0),
        matcher="pyramid",
        df_full=df_full,
        df_crop=df_crop,
    )

    assert out["success"] is True


def test_nucleisky3d_rejects_single_precomputed_dataframe():
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    df = pd.DataFrame({"centroid_z_px": [1.0], "centroid_y_px": [1.0], "centroid_x_px": [1.0]})

    with pytest.raises(ValueError, match="df_full and df_crop must both be provided"):
        pipeline.NucleiSky3D(
            label_full=labels,
            label_crop=labels,
            matcher="pyramid",
            df_full=df,
        )


def test_mip_wraps_memoryerror_with_actionable_message(monkeypatch):
    volume = np.ones((3, 3, 3), dtype=np.float32)

    def _boom(*args, **kwargs):
        raise MemoryError("oom")

    monkeypatch.setattr(np, "max", _boom)

    with pytest.raises(MemoryError, match="mip_downsample=2 or 4"):
        _mip(volume, axis=0)


def test_mip_downsample_matches_sliced_reference():
    volume = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    got = _mip(volume, axis=0, downsample=2)
    expected = np.max(volume[::2, ::2, ::2], axis=0)
    np.testing.assert_allclose(got, expected)
