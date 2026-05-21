import numpy as np
from tifffile import TiffFile

from nucleisky3d.io import save_tiff_zyx


def test_save_tiff_zyx_roundtrip_shape_dtype_int_labels(tmp_path):
    arr = (np.arange(4 * 5 * 6, dtype=np.uint32) % 17).reshape(4, 5, 6)
    path = tmp_path / "labels.tif"

    save_tiff_zyx(path, arr)

    with TiffFile(path) as tif:
        out = tif.asarray()

    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


def test_save_tiff_zyx_writes_imagej_spacing_and_resolution(tmp_path):
    arr = np.ones((3, 4, 5), dtype=np.float32)
    path = tmp_path / "with_voxel.tif"

    save_tiff_zyx(path, arr, voxel_size_um_zyx=(1.5, 0.4, 0.25))

    with TiffFile(path) as tif:
        out = tif.asarray()
        assert out.shape == arr.shape
        assert out.dtype == arr.dtype

        ij = tif.imagej_metadata or {}
        assert float(ij["spacing"]) == 1.5
        assert ij["unit"] in ("um", "µm", "μm")

        xres = tif.pages[0].tags["XResolution"].value
        yres = tif.pages[0].tags["YResolution"].value
        xres = float(xres[0]) / float(xres[1])
        yres = float(yres[0]) / float(yres[1])
        assert abs(xres - 4.0) < 1e-6
        assert abs(yres - 2.5) < 1e-6
