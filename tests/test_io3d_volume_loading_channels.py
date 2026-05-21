import numpy as np
import pytest

from nucleisky3d.io import inspect_volume_header, load_volume


def test_inspect_volume_header_reads_npy_shape_and_dtype(tmp_path):
    arr = np.zeros((2, 5, 6, 7), dtype=np.uint16)
    p = tmp_path / "vol.npy"
    np.save(p, arr)

    hdr = inspect_volume_header(str(p))

    assert tuple(hdr["shape"]) == (2, 5, 6, 7)
    assert hdr["dtype"] == "uint16"


def test_load_volume_uses_explicit_channel_axis_and_index(tmp_path):
    arr = np.arange(2 * 4 * 5 * 6, dtype=np.int32).reshape(2, 4, 5, 6)
    p = tmp_path / "vol.npy"
    np.save(p, arr)

    vol = load_volume(str(p), channel_axis=0, channel_index=1)

    np.testing.assert_array_equal(vol, arr[1])
    assert vol.shape == (4, 5, 6)


def test_load_volume_rejects_invalid_channel_index(tmp_path):
    arr = np.zeros((2, 4, 5, 6), dtype=np.uint8)
    p = tmp_path / "vol.npy"
    np.save(p, arr)

    with pytest.raises(ValueError, match="channel_index=2 is out of bounds"):
        load_volume(str(p), channel_axis=0, channel_index=2)
