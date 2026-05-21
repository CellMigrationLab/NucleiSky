from pathlib import Path

import numpy as np
import pytest
from tifffile import imwrite

from nucleisky2d.export import inspect_image_header
from nucleisky2d.io import get_pixel_size_um_from_tiff


def test_inspect_image_header_tiff_uses_shared_pixel_size_parser(tmp_path: Path):
    path = tmp_path / "sample.ome.tiff"
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)

    imwrite(
        path,
        data,
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.42,
            "PhysicalSizeY": 0.42,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )

    header = inspect_image_header(str(path))
    expected = get_pixel_size_um_from_tiff(str(path), return_details=False)

    assert header["pixel_size_um"] is not None
    assert header["pixel_size_um"] == pytest.approx(expected)
