import numpy as np

from nucleisky3d.preprocess import (
    choose_common_target_um_per_voxel,
    rescale_to_target_um_per_voxel,
    scale_normalize_pair_for_segmentation,
)


def test_choose_common_target_um_per_voxel_coarsest():
    target = choose_common_target_um_per_voxel((2.0, 0.5, 0.25), (1.0, 0.75, 0.2), strategy="coarsest")
    np.testing.assert_allclose(target, np.array([2.0, 0.75, 0.25]))


def test_rescale_to_target_um_per_voxel_returns_expected_scale_and_shape():
    vol = np.zeros((5, 6, 7), dtype=np.float32)

    vol_rs, sf = rescale_to_target_um_per_voxel(
        vol,
        current_um_per_voxel_zyx=(2.0, 1.0, 1.0),
        target_um_per_voxel_zyx=(1.0, 1.0, 2.0),
        order=0,
    )

    np.testing.assert_allclose(sf, np.array([2.0, 1.0, 0.5]))
    assert vol_rs.shape == (10, 6, 4)


def test_scale_normalize_pair_for_segmentation_aligns_effective_voxel_sizes():
    full = np.ones((12, 12, 12), dtype=np.float32)
    crop = np.ones((10, 14, 16), dtype=np.float32)

    (
        full_seg,
        crop_seg,
        vox_full_seg,
        vox_crop_seg,
        scale_full,
        scale_crop,
        target_um,
    ) = scale_normalize_pair_for_segmentation(
        full,
        crop,
        voxel_size_full_um_zyx=(2.0, 1.0, 0.5),
        voxel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        strategy="coarsest",
        order=0,
    )

    np.testing.assert_allclose(np.asarray(target_um), np.array([2.0, 1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(scale_full), np.array([1.0, 1.0, 0.5]))
    np.testing.assert_allclose(np.asarray(scale_crop), np.array([0.5, 1.0, 1.0]))

    np.testing.assert_allclose(np.asarray(vox_full_seg), np.array(target_um))
    np.testing.assert_allclose(np.asarray(vox_crop_seg), np.array(target_um))

    assert full_seg.shape == (12, 12, 6)
    assert crop_seg.shape == (5, 14, 16)
