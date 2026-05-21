import numpy as np

import nucleisky3d
from nucleisky3d.matching import pyramid


def _patch_lightweight_pyramid(monkeypatch):
    monkeypatch.setattr(pyramid, "build_geometric_knn_graph", lambda pts, k=8: object())
    monkeypatch.setattr(
        pyramid,
        "build_tetrahedron_node_features",
        lambda pts, graph, n_tetrahedra: np.zeros((len(pts), 7)),
    )
    monkeypatch.setattr(
        pyramid,
        "_zscore_with_ref",
        lambda feat, ref_mu=None, ref_sigma=None, eps=1e-8: (
            feat,
            np.zeros((1, feat.shape[1])),
            np.ones((1, feat.shape[1])),
        ),
    )
    monkeypatch.setattr(
        pyramid,
        "_filter_mutual_nearest_neighbors",
        lambda tetra_crop, tetra_full, k_feat=1: [
            (0, 0, 0.0),
            (1, 1, 0.0),
            (2, 2, 0.0),
            (3, 3, 0.0),
        ],
    )


def test_nucleisky3d_public_api_exports_matching_entrypoints():
    expected = {
        "run_pyramid_based_matching_um",
        "run_geometric_hashing_matching_3d_um",
        "geometric_hashing_match_similarity_3d",
        "estimate_dynamic_scale_bounds_3d",
    }

    assert expected.issubset(set(nucleisky3d.__all__))
    for name in expected:
        assert hasattr(nucleisky3d, name)


def test_pyramid_bbox_uses_configured_margin_um(monkeypatch):
    _patch_lightweight_pyramid(monkeypatch)

    crop = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    full = crop.copy()

    monkeypatch.setattr(
        pyramid,
        "estimate_similarity_3d",
        lambda src, dst: (1.0, np.eye(3), np.zeros(3)),
    )

    seen = {}

    def _fake_bbox(**kwargs):
        seen.update(kwargs)
        return ((0, 0, 0), (1, 1, 1))

    monkeypatch.setattr(pyramid, "bbox_full_px_from_similarity_um_3d", _fake_bbox)

    scale, R, t, bbox = pyramid.run_pyramid_based_matching_um(
        centroids_crop_um=crop,
        centroids_full_um=full,
        n_iters=1,
        min_inliers=1,
        use_icp_refinement=False,
        margin_um=7.5,
        voxel_size_full_um_zyx=(1.0, 1.0, 1.0),
        voxel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        full_shape_px_zyx=(16, 16, 16),
        crop_shape_px_zyx=(8, 8, 8),
    )

    assert scale == 1.0
    assert R is not None and t is not None
    assert bbox is not None
    assert seen["margin_um"] == 7.5
