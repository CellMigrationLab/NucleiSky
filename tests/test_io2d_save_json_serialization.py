from pathlib import Path
import json

import numpy as np

from nucleisky2d.io import save_json


def test_save_json_handles_numpy_and_path(tmp_path: Path):
    out_path = tmp_path / "out.json"
    payload = {
        "n": np.int64(3),
        "x": np.float32(1.25),
        "ok": np.bool_(True),
        "arr": np.array([1, 2, 3], dtype=np.int16),
        "p": tmp_path / "image.tif",
    }

    save_json(out_path, payload)

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data == {
        "n": 3,
        "x": 1.25,
        "ok": True,
        "arr": [1, 2, 3],
        "p": str(tmp_path / "image.tif"),
    }
