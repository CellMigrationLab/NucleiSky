import numpy as np

from nucleisky3d.matching import hashing3d, pyramid
from nucleisky3d.matching.geometry import rotation_angle_deg_3d


def _rot_z(deg: float) -> np.ndarray:
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def test_rotation_angle_deg_3d_uses_trace_formula():
    assert np.isclose(rotation_angle_deg_3d(np.eye(3)), 0.0)
    assert np.isclose(rotation_angle_deg_3d(_rot_z(180.0)), 180.0)


def test_hashing3d_rejects_result_when_refined_angle_exceeds_limit(monkeypatch):
    crop = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    full = crop + 2.0

    monkeypatch.setattr(
        hashing3d,
        "geometric_hashing_match_similarity_3d",
        lambda **kwargs: (1.0, np.eye(3), np.zeros(3), np.array([0, 1, 2, 3])),
    )
    monkeypatch.setattr(
        hashing3d,
        "icp_similarity_3d",
        lambda *args, **kwargs: (1.0, _rot_z(100.0), np.zeros(3)),
    )

    scale, R, t, bbox = hashing3d.run_geometric_hashing_matching_3d_um(
        centroids_crop_um=crop,
        centroids_full_um=full,
        full_shape_px=(16, 16, 16),
        patch_shape_px=(8, 8, 8),
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_patch_um_zyx=(1.0, 1.0, 1.0),
        angle_max_deg=45.0,
    )

    assert (scale, R, t, bbox) == (None, None, None, None)


def _patch_lightweight_pyramid(monkeypatch):
    monkeypatch.setattr(pyramid, "build_geometric_knn_graph", lambda pts, k=8: object())
    monkeypatch.setattr(pyramid, "build_tetrahedron_node_features", lambda pts, graph, n_tetrahedra: np.zeros((len(pts), 7)))
    monkeypatch.setattr(pyramid, "_zscore_with_ref", lambda feat, ref_mu=None, ref_sigma=None, eps=1e-8: (feat, np.zeros((1, feat.shape[1])), np.ones((1, feat.shape[1]))))
    monkeypatch.setattr(
        pyramid,
        "_filter_mutual_nearest_neighbors",
        lambda tetra_crop, tetra_full, k_feat=1: [(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0), (3, 3, 0.0)],
    )


def test_pyramid_rejects_candidate_rotation_over_limit(monkeypatch):
    _patch_lightweight_pyramid(monkeypatch)

    crop = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    full = crop.copy()

    monkeypatch.setattr(pyramid, "estimate_similarity_3d", lambda src, dst: (1.0, _rot_z(120.0), np.zeros(3)))

    scale, R, t, bbox = pyramid.run_pyramid_based_matching_um(
        centroids_crop_um=crop,
        centroids_full_um=full,
        n_iters=1,
        min_inliers=1,
        use_icp_refinement=False,
        angle_max_deg=45.0,
    )

    assert (scale, R, t, bbox) == (None, None, None, None)


def test_pyramid_rejects_refined_rotation_over_limit(monkeypatch):
    _patch_lightweight_pyramid(monkeypatch)

    crop = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    full = crop.copy()

    monkeypatch.setattr(pyramid, "estimate_similarity_3d", lambda src, dst: (1.0, np.eye(3), np.zeros(3)))
    monkeypatch.setattr(pyramid, "icp_similarity_3d", lambda *args, **kwargs: (1.0, _rot_z(100.0), np.zeros(3)))

    scale, R, t, bbox = pyramid.run_pyramid_based_matching_um(
        centroids_crop_um=crop,
        centroids_full_um=full,
        n_iters=1,
        min_inliers=1,
        use_icp_refinement=True,
        angle_max_deg=45.0,
    )

    assert (scale, R, t, bbox) == (None, None, None, None)
