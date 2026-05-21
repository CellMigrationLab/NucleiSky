import numpy as np
import pytest
from tifffile import imwrite

from nucleisky3d.io import require_voxel_size_um_zyx


def test_require_voxel_size_um_zyx_returns_metadata_zyx(tmp_path):
    arr = np.zeros((4, 8, 10), dtype=np.uint16)
    path = tmp_path / "with_meta.tif"

    # X/Y from resolution tags + ImageJ unit hint, Z from ImageJ spacing.
    imwrite(
        path,
        arr,
        imagej=True,
        resolution=(4.0, 5.0),  # px per µm => x=0.25 µm/px, y=0.2 µm/px
        metadata={"unit": "um", "spacing": 1.5},
    )

    vox = require_voxel_size_um_zyx(path)
    np.testing.assert_allclose(vox, (1.5, 0.2, 0.25), rtol=1e-6, atol=1e-9)


def test_require_voxel_size_um_zyx_raises_when_metadata_missing_and_no_fallback(tmp_path):
    arr = np.zeros((5, 6, 7), dtype=np.uint16)
    path = tmp_path / "no_meta.tif"
    imwrite(path, arr)

    with pytest.raises(ValueError, match=r"Provide fallback=\(z_um, y_um, x_um\)"):
        require_voxel_size_um_zyx(path)


def test_require_voxel_size_um_zyx_uses_fallback_when_metadata_missing(tmp_path):
    arr = np.zeros((5, 6, 7), dtype=np.uint16)
    path = tmp_path / "no_meta_fallback.tif"
    imwrite(path, arr)

    vox = require_voxel_size_um_zyx(path, fallback=(2.0, 1.0, 0.5))
    assert vox == (2.0, 1.0, 0.5)


def test_require_voxel_size_um_zyx_allow_missing_z_does_not_fabricate_z(tmp_path):
    arr = np.zeros((4, 8, 10), dtype=np.uint16)
    path = tmp_path / "xy_only_meta.tif"

    # X/Y are present but Z spacing is absent.
    imwrite(
        path,
        arr,
        imagej=True,
        resolution=(4.0, 5.0),
        metadata={"unit": "um"},
    )

    with pytest.raises(ValueError, match=r"missing or incomplete"):
        require_voxel_size_um_zyx(path, allow_missing_z=True)
