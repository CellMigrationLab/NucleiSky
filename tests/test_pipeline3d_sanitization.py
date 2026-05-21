import numpy as np

from nucleisky3d import pipeline


def test_nucleisky3d_reports_sanitization_stats_in_preflight(monkeypatch):
    seen = {}

    def _fake_matcher(**kwargs):
        seen["n_crop"] = int(kwargs["centroids_crop_um"].shape[0])
        seen["n_full"] = int(kwargs["centroids_full_um"].shape[0])
        return 1.0, np.eye(3, dtype=float), np.zeros(3, dtype=float), (0, 1, 0, 1, 0, 1)

    monkeypatch.setattr(pipeline, "run_pyramid_based_matching_um", _fake_matcher)

    full = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=float,
    )
    crop = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [np.nan, 9.0, 9.0],
        ],
        dtype=float,
    )

    out = pipeline.NucleiSky3D(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px_zyx=(10, 10, 10),
        crop_shape_px_zyx=(6, 6, 6),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        matcher="pyramid",
        matcher_config={"_common": {"sanitize_dedup_radius_um": 0.1}},
    )

    assert out["success"] is True
    assert out["preflight"]["sanitize"]["full"]["n_deduped"] == 1
    assert out["preflight"]["sanitize"]["crop"]["n_in"] == 5
    assert out["preflight"]["sanitize"]["crop"]["n_finite"] == 4
    assert seen == {"n_crop": 4, "n_full": 4}


def test_nucleisky3d_computes_dynamic_min_inliers_for_pyramid(monkeypatch):
    seen = {}

    def _fake_pyramid(**kwargs):
        seen.update(kwargs)
        return 1.0, np.eye(3, dtype=float), np.zeros(3, dtype=float), (0, 1, 0, 1, 0, 1)

    monkeypatch.setattr(pipeline, "run_pyramid_based_matching_um", _fake_pyramid)

    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=float,
    )

    out = pipeline.NucleiSky3D(
        centroids_crop_um=pts,
        centroids_full_um=pts,
        full_shape_px_zyx=(12, 12, 12),
        crop_shape_px_zyx=(10, 10, 10),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        matcher="pyramid",
        matcher_config={"_common": {"min_inliers_frac": 0.5}},
    )

    assert out["success"] is True
    assert seen["min_inliers"] == 5
    assert out["preflight"]["min_inliers"]["min_inliers"] == 5


def test_nucleisky3d_resolves_dict_min_inliers_before_hashing(monkeypatch):
    seen = {}

    def _fake_hashing(**kwargs):
        seen.update(kwargs)
        return 1.0, np.eye(3, dtype=float), np.zeros(3, dtype=float), (0, 1, 0, 1, 0, 1)

    monkeypatch.setattr(pipeline, "run_geometric_hashing_matching_3d_um", _fake_hashing)

    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=float,
    )

    out = pipeline.NucleiSky3D(
        centroids_crop_um=pts,
        centroids_full_um=pts,
        full_shape_px_zyx=(12, 12, 12),
        crop_shape_px_zyx=(10, 10, 10),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        matcher="hashing3d",
        matcher_config={
            "hashing3d": {
                "min_inliers": {
                    "min_inliers_abs": 20,
                    "min_inliers_frac": 0.5,
                    "min_inliers_cap_frac": 0.8,
                }
            }
        },
    )

    assert out["success"] is True
    assert seen["min_inliers_abs"] == 8
    assert seen["min_inliers_frac"] == 0.5
    assert "min_inliers" not in seen
    assert out["preflight"]["min_inliers"]["min_inliers"] == 8


def test_nucleisky3d_accepts_hashing_user_facing_name(monkeypatch):
    seen = {"called": False}

    def _fake_hashing(**kwargs):
        seen["called"] = True
        return 1.0, np.eye(3, dtype=float), np.zeros(3, dtype=float), (0, 1, 0, 1, 0, 1)

    monkeypatch.setattr(pipeline, "run_geometric_hashing_matching_3d_um", _fake_hashing)

    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=float,
    )

    out = pipeline.NucleiSky3D(
        centroids_crop_um=pts,
        centroids_full_um=pts,
        full_shape_px_zyx=(12, 12, 12),
        crop_shape_px_zyx=(10, 10, 10),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        matcher="hashing",
    )

    assert seen["called"] is True
    assert out["matcher"] == "hashing"
    assert out["success"] is True


def test_nucleisky3d_invalid_matcher_message_lists_user_facing_names():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=float,
    )

    try:
        pipeline.NucleiSky3D(
            centroids_crop_um=pts,
            centroids_full_um=pts,
            full_shape_px_zyx=(12, 12, 12),
            crop_shape_px_zyx=(10, 10, 10),
            pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
            pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
            matcher="unknown",
        )
    except ValueError as exc:
        msg = str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid matcher")

    assert "pyramid" in msg
    assert "hashing" in msg
    assert "hashing3d" not in msg
