import numpy as np
import pytest
from tifffile import imwrite

from nucleisky2d.io import get_pixel_size_um_from_tiff
from nucleisky3d.io import require_voxel_size_um_zyx
from nucleisky3d.matching.geometry import bbox_full_px_from_similarity_um_3d


@pytest.mark.geometry
def test_missing_voxel_metadata_requires_fallback_or_errors(tmp_path):
    vol = np.zeros((4, 6, 8), dtype=np.uint16)
    path = tmp_path / "missing_meta.tif"
    imwrite(path, vol, photometric="minisblack")

    with pytest.raises(ValueError):
        require_voxel_size_um_zyx(path)

    got = require_voxel_size_um_zyx(path, fallback=(1.5, 0.7, 0.7))
    assert got == (1.5, 0.7, 0.7)


@pytest.mark.geometry
def test_wrong_voxel_size_metadata_changes_bbox_mapping():
    t_um = np.array([12.0, 8.0, 6.0], dtype=float)
    R = np.eye(3, dtype=float)
    crop_shape = (10, 20, 30)

    bbox_true = bbox_full_px_from_similarity_um_3d(
        crop_shape_px=crop_shape,
        pixel_size_full_um_zyx=(2.0, 0.5, 0.5),
        pixel_size_crop_um_zyx=(2.0, 0.5, 0.5),
        scale=1.0,
        R_zyx=R,
        t_um_zyx=t_um,
        full_shape_px=(200, 200, 200),
    )
    bbox_wrong = bbox_full_px_from_similarity_um_3d(
        crop_shape_px=crop_shape,
        pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
        pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        scale=1.0,
        R_zyx=R,
        t_um_zyx=t_um,
        full_shape_px=(200, 200, 200),
    )
    assert tuple(int(v) for v in bbox_true) != tuple(int(v) for v in bbox_wrong)


@pytest.mark.geometry
def test_2d_tiff_pixel_size_roundtrip_header_parser(tmp_path):
    img = np.zeros((6, 6), dtype=np.uint16)
    path = tmp_path / "meta.ome.tif"
    imwrite(
        path,
        img,
        ome=True,
        metadata={"axes": "YX", "PhysicalSizeX": 0.4, "PhysicalSizeY": 0.4, "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"},
    )
    px = get_pixel_size_um_from_tiff(path)
    assert px == pytest.approx(0.4, abs=1e-9)
