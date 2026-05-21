from __future__ import annotations

import numpy as np
from tifffile import imread


def assert_2d_record_and_artifact_consistent(record: dict, artifact_path, *, expect_bbox_contains: tuple[int, int, int, int] | None = None):
    assert "bbox_full_px_y0y1x0x1" in record, "2D record missing bbox_full_px_y0y1x0x1"
    y0, y1, x0, x1 = [int(v) for v in record["bbox_full_px_y0y1x0x1"]]
    assert y1 > y0 and x1 > x0, f"Invalid 2D bbox ordering: {(y0,y1,x0,x1)}"

    arr = imread(artifact_path)
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    assert h == (y1 - y0), f"2D artifact height {h} != bbox height {y1-y0}"
    assert w == (x1 - x0), f"2D artifact width {w} != bbox width {x1-x0}"

    for k in ("scale", "R_yx", "t_um_yx"):
        assert k in record, f"2D record missing {k}"
    R = np.asarray(record["R_yx"], dtype=float)
    t = np.asarray(record["t_um_yx"], dtype=float)
    assert R.shape == (2, 2), f"R_yx must be 2x2, got {R.shape}"
    assert t.shape == (2,), f"t_um_yx must be len-2, got {t.shape}"

    if "A_px" in record and "b_px" in record:
        A = np.asarray(record["A_px"], dtype=float)
        b = np.asarray(record["b_px"], dtype=float)
        assert A.shape == (2, 2), f"A_px must be 2x2, got {A.shape}"
        assert b.shape == (2,), f"b_px must be len-2, got {b.shape}"

    if expect_bbox_contains is not None:
        ey0, ey1, ex0, ex1 = expect_bbox_contains
        assert y0 <= ey0 <= ey1 <= y1
        assert x0 <= ex0 <= ex1 <= x1


def assert_3d_record_and_artifact_consistent(record: dict, artifact_path):
    assert "bbox_full_px_z0z1y0y1x0x1" in record, "3D record missing bbox_full_px_z0z1y0y1x0x1"
    z0, z1, y0, y1, x0, x1 = [int(v) for v in record["bbox_full_px_z0z1y0y1x0x1"]]
    assert z1 > z0 and y1 > y0 and x1 > x0, f"Invalid 3D bbox ordering: {(z0,z1,y0,y1,x0,x1)}"

    arr = imread(artifact_path)
    d, h, w = map(int, arr.shape[-3:])
    assert d == (z1 - z0), f"3D artifact depth {d} != bbox depth {z1-z0}"
    assert h == (y1 - y0), f"3D artifact height {h} != bbox height {y1-y0}"
    assert w == (x1 - x0), f"3D artifact width {w} != bbox width {x1-x0}"

    for k in ("scale", "R_zyx", "t_um_zyx", "pixel_size_full_um_zyx", "pixel_size_crop_um_zyx"):
        assert k in record, f"3D record missing {k}"
    R = np.asarray(record["R_zyx"], dtype=float)
    t = np.asarray(record["t_um_zyx"], dtype=float)
    vf = np.asarray(record["pixel_size_full_um_zyx"], dtype=float)
    vc = np.asarray(record["pixel_size_crop_um_zyx"], dtype=float)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert vf.shape == (3,) and np.all(vf > 0), f"Invalid full voxel spacing: {vf}"
    assert vc.shape == (3,) and np.all(vc > 0), f"Invalid crop voxel spacing: {vc}"
