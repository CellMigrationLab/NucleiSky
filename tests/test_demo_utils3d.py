import numpy as np
import pytest

from nucleisky3d.demo_utils import generate_random_subvolume_3d


def test_generate_random_subvolume_uses_voxel_geometry_for_sampling():
    full = np.ones((64, 64, 64), dtype=np.float32)
    rng = np.random.default_rng(0)

    crop_a, vox_a, gt_a = generate_random_subvolume_3d(
        full,
        crop_shape_zyx=(16, 16, 16),
        scale_range=(1.0, 1.01),
        voxel_size_um=(1.0, 1.0, 1.0),
        rng=rng,
    )

    rng = np.random.default_rng(0)
    crop_b, vox_b, gt_b = generate_random_subvolume_3d(
        full,
        crop_shape_zyx=(16, 16, 16),
        scale_range=(1.0, 1.01),
        voxel_size_um=(2.0, 0.5, 0.25),
        rng=rng,
    )

    # Same RNG seed + same array/crop/scale params => same sampled geometry in voxel space.
    np.testing.assert_allclose(crop_a, crop_b)
    np.testing.assert_allclose(gt_a["R"], gt_b["R"])
    np.testing.assert_allclose(gt_a["t"] / np.array([1.0, 1.0, 1.0]), gt_b["t"] / np.array([2.0, 0.5, 0.25]))

    # Physical outputs still depend on voxel size.
    assert not np.allclose(vox_a, vox_b)


def test_generate_random_subvolume_small_full_volume_warns_but_succeeds():
    full = np.zeros((8, 8, 8), dtype=np.uint16)

    with pytest.warns(RuntimeWarning, match="interpreted in voxels"):
        crop, crop_vox, gt = generate_random_subvolume_3d(
            full,
            crop_shape_zyx=(8, 8, 8),
            scale_range=(0.5, 0.6),
            voxel_size_um=1.0,
            rng=np.random.default_rng(1),
        )

    assert crop.shape == (8, 8, 8)
    assert crop_vox.shape == (3,)
    assert set(gt.keys()) == {"scale", "R", "t"}
