from pathlib import Path
import types
import sys

import numpy as np

import nucleisky2d.io as io2d


def _install_fake_zarr(monkeypatch, store):
    fake = types.SimpleNamespace(
        Group=dict,
        open=lambda *args, **kwargs: store,
    )
    monkeypatch.setitem(sys.modules, 'zarr', fake)


def test_load_image_probes_directory_without_markers_2d(tmp_path: Path, monkeypatch):
    p = tmp_path / 'folder_input'
    p.mkdir()

    arr = np.arange(4, dtype=np.uint16).reshape(2, 2)
    _install_fake_zarr(monkeypatch, arr)

    out = io2d.load_image(str(p))
    assert np.array_equal(out, arr)
