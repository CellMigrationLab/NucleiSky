import json

import numpy as np

from nucleisky3d.io import load_transforms_any_3d
from nucleisky3d.export import similarity_um_to_affine_px_3d


def test_load_transforms_any_3d_normalizes_legacy_adaptive_jsonl(tmp_path):
    legacy = {
        "matcher": "pyramid",
        "success": True,
        "best_scale": 1.25,
        "best_R": np.eye(3, dtype=float).tolist(),
        "best_t": [4.0, -1.0, 2.0],
        "best_bbox": [0, 5, 10, 20, 30, 40],
        "pixel_size_full_orig_um_zyx": [0.4, 0.5, 0.6],
        "pixel_size_crop_orig_um_zyx": [0.7, 0.8, 0.9],
        "match_quality": {"success": True, "frac_inliers": 0.8, "mean_error_um": 0.6},
    }

    p = tmp_path / "transforms.jsonl"
    with p.open("w", encoding="utf-8") as f:
        json.dump(legacy, f)
        f.write("\n")

    recs = load_transforms_any_3d(str(p))
    assert len(recs) == 1
    rec = recs[0]

    assert rec["scale"] == legacy["best_scale"]
    assert rec["R_zyx"] == legacy["best_R"]
    assert rec["t_um_zyx"] == legacy["best_t"]
    assert rec["pixel_size_full_um_zyx"] == legacy["pixel_size_full_orig_um_zyx"]
    assert rec["pixel_size_crop_um_zyx"] == legacy["pixel_size_crop_orig_um_zyx"]
    assert rec["bbox_full_px_z0z1y0y1x0x1"] == legacy["best_bbox"]

    A_px, b_px = similarity_um_to_affine_px_3d(
        best_scale=legacy["best_scale"],
        best_R=np.asarray(legacy["best_R"], dtype=float),
        best_t=np.asarray(legacy["best_t"], dtype=float),
        pixel_size_full_um=np.asarray(legacy["pixel_size_full_orig_um_zyx"], dtype=float),
        pixel_size_crop_um=np.asarray(legacy["pixel_size_crop_orig_um_zyx"], dtype=float),
    )
    np.testing.assert_allclose(np.asarray(rec["A_px"], dtype=float), A_px)
    np.testing.assert_allclose(np.asarray(rec["b_px"], dtype=float), b_px)

    assert rec["_source_kind"] == "jsonl"
    assert rec["_line"] == 1
