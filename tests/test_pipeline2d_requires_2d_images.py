import numpy as np
import pytest

from nucleisky2d import pipeline


def _minimal_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=float,
    )


def test_nucleisky_rejects_non_2d_images():
    pts = _minimal_points()
    img_3d = np.zeros((2, 10, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        pipeline.NucleiSky(
            centroids_crop_um=pts,
            centroids_full_um=pts,
            img_full=img_3d,
            img_crop=np.zeros((10, 10), dtype=np.float32),
            ij_percentile_normalize=False,
            pixel_size_full_um=1.0,
            pixel_size_crop_um=1.0,
            matcher="quad",
        )


def test_nucleisky_accepts_2d_images_and_returns_result(monkeypatch):
    pts = _minimal_points()

    def _fake_quad(**kwargs):
        return 1.0, np.eye(2, dtype=float), np.zeros(2, dtype=float), (0, 5, 0, 5)

    def _fake_quality(**kwargs):
        return {
            "success": True,
            "frac_inliers": 1.0,
            "mean_error_um": 0.0,
        }

    monkeypatch.setattr(pipeline, "run_quad_based_matching_um", _fake_quad)
    monkeypatch.setattr(pipeline, "evaluate_match_quality", _fake_quality)

    out = pipeline.NucleiSky(
        centroids_crop_um=pts,
        centroids_full_um=pts,
        img_full=np.zeros((20, 20), dtype=np.float32),
        img_crop=np.zeros((10, 10), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="quad",
    )

    assert isinstance(out, dict)
    assert "success" in out


class _Lazy2DArray:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def __array__(self, *args, **kwargs):
        raise AssertionError("Lazy array should not be materialized for shape validation")


def test_nucleisky_accepts_lazy_2d_arrays_without_materializing(monkeypatch):
    pts = _minimal_points()

    def _fake_quad(**kwargs):
        return 1.0, np.eye(2, dtype=float), np.zeros(2, dtype=float), (0, 5, 0, 5)

    def _fake_quality(**kwargs):
        return {
            "success": True,
            "frac_inliers": 1.0,
            "mean_error_um": 0.0,
        }

    monkeypatch.setattr(pipeline, "run_quad_based_matching_um", _fake_quad)
    monkeypatch.setattr(pipeline, "evaluate_match_quality", _fake_quality)

    out = pipeline.NucleiSky(
        centroids_crop_um=pts,
        centroids_full_um=pts,
        img_full=_Lazy2DArray((20, 20)),
        img_crop=_Lazy2DArray((10, 10)),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="quad",
    )

    assert isinstance(out, dict)
    assert "success" in out
