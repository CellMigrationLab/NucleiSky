import numpy as np

from nucleisky3d.matching import hashing3d


def _sample_points(n=6):
    return np.stack(
        [
            np.linspace(0.0, 5.0, n),
            np.linspace(1.0, 6.0, n),
            np.linspace(2.0, 7.0, n),
        ],
        axis=1,
    )


def test_run_hashing3d_respects_primary_config_keys(monkeypatch):
    seen = {}

    def _fake_match(**kwargs):
        seen.update(kwargs)
        return 1.2, np.eye(3), np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2])

    monkeypatch.setattr(hashing3d, "geometric_hashing_match_similarity_3d", _fake_match)

    crop = _sample_points()
    full = _sample_points() + 1.0

    hashing3d.run_geometric_hashing_matching_3d_um(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px=(32, 32, 32),
        patch_shape_px=(16, 16, 16),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_patch_um_zyx=(1.0, 1.0, 1.0),
        min_height_ratio=0.33,
        use_icp_refinement=False,
    )

    assert seen["min_height_ratio"] == 0.33
    assert seen["min_inliers_abs"] == 20
    assert seen["min_inliers_frac"] == 0.12


def test_run_hashing3d_accepts_backward_compatible_alias_keys(monkeypatch):
    seen = {}

    def _fake_match(**kwargs):
        seen.update(kwargs)
        return 1.0, np.eye(3), np.zeros(3), np.array([0, 1, 2])

    monkeypatch.setattr(hashing3d, "geometric_hashing_match_similarity_3d", _fake_match)

    crop = _sample_points()
    full = _sample_points() + 1.0

    hashing3d.run_geometric_hashing_matching_3d_um(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px=(32, 32, 32),
        patch_shape_px=(16, 16, 16),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_patch_um_zyx=(1.0, 1.0, 1.0),
        bin_size=0.2,
        max_neighbors=55,
        max_pairs_per_anchor_full=11,
        max_k_per_pair_full=12,
        max_l_per_base_full=13,
        max_neighbors_crop=44,
        max_pairs_per_anchor_crop=14,
        max_k_per_pair_crop=15,
        max_l_per_base_crop=16,
        min_inliers=17,
        use_icp_refinement=False,
    )

    assert seen["bin_size_xyz"] == 0.2
    assert seen["max_neighbors_full"] == 55
    assert seen["max_pairs_per_anchor"] == 11
    assert seen["max_k_per_pair"] == 12
    assert seen["max_l_per_base"] == 13
    assert seen["max_neighbors_patch"] == 44
    assert seen["max_pairs_per_anchor_patch"] == 14
    assert seen["max_k_per_pair_patch"] == 15
    assert seen["max_l_per_base_patch"] == 16
    assert seen["min_inliers_abs"] == 17


def test_run_hashing3d_uses_configurable_icp_iters(monkeypatch):
    def _fake_match(**kwargs):
        return 1.0, np.eye(3), np.zeros(3), np.array([0, 1, 2])

    seen_icp = {}

    def _fake_icp(*args, **kwargs):
        seen_icp.update(kwargs)
        return 1.0, np.eye(3), np.zeros(3)

    monkeypatch.setattr(hashing3d, "geometric_hashing_match_similarity_3d", _fake_match)
    monkeypatch.setattr(hashing3d, "icp_similarity_3d", _fake_icp)

    crop = _sample_points()
    full = _sample_points() + 1.0

    hashing3d.run_geometric_hashing_matching_3d_um(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px=(32, 32, 32),
        patch_shape_px=(16, 16, 16),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_patch_um_zyx=(1.0, 1.0, 1.0),
        icp_iters=7,
    )

    assert seen_icp["n_iters"] == 7
