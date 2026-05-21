import pytest

pytest.importorskip("SimpleITK")

import json

import numpy as np
import pandas as pd

from nucleisky3d import pipeline
from nucleisky3d.export import similarity_um_to_affine_px_3d


def test_run_adaptive_matching_exports_canonical_transform_fields(monkeypatch, tmp_path):
    df_full = pd.DataFrame({"centroid_z_px": [1.0], "centroid_y_px": [2.0], "centroid_x_px": [3.0]})
    df_crop = pd.DataFrame({"centroid_z_px": [1.5], "centroid_y_px": [2.5], "centroid_x_px": [3.5]})

    best_out = {
        "success": True,
        "matcher": "pyramid",
        "best_scale": 1.2,
        "best_R": np.eye(3, dtype=float),
        "best_t": np.array([5.0, -2.0, 1.0], dtype=float),
        "best_bbox": (1, 4, 2, 5, 3, 6),
        "match_quality": {"frac_inliers": 0.9, "mean_error_um": 0.4, "success": True},
    }

    def _fake_run_adaptive(**kwargs):
        return best_out, []

    monkeypatch.setattr(pipeline, "run_adaptive_nucleisky_3d", _fake_run_adaptive)

    result_dir = tmp_path / "results"
    pipeline.run_adaptive_matching_and_export_3d(
        df_full=df_full,
        df_crop=df_crop,
        pixel_size_full_orig_um_zyx=(0.5, 0.6, 0.7),
        pixel_size_crop_orig_um_zyx=(0.8, 0.9, 1.0),
        labels_full=np.zeros((8, 8, 8), dtype=np.int32),
        labels_crop=np.zeros((4, 4, 4), dtype=np.int32),
        result_dir=str(result_dir),
    )

    transforms_path = result_dir / "matching" / "adaptive_3d" / "exports_adaptive" / "transforms.jsonl"
    line = transforms_path.read_text(encoding="utf-8").strip()
    rec = json.loads(line)

    assert rec["scale"] == best_out["best_scale"]
    assert rec["R_zyx"] == best_out["best_R"].tolist()
    assert rec["t_um_zyx"] == best_out["best_t"].tolist()
    assert rec["pixel_size_full_um_zyx"] == [0.5, 0.6, 0.7]
    assert rec["pixel_size_crop_um_zyx"] == [0.8, 0.9, 1.0]
    assert rec["bbox_full_px_z0z1y0y1x0x1"] == [1, 4, 2, 5, 3, 6]
    assert rec["match_quality"] == best_out["match_quality"]
    assert "best_scale" not in rec
    assert "best_R" not in rec
    assert "best_t" not in rec
    assert "best_bbox" not in rec

    A_px, b_px = similarity_um_to_affine_px_3d(
        best_scale=best_out["best_scale"],
        best_R=best_out["best_R"],
        best_t=best_out["best_t"],
        pixel_size_full_um=(0.5, 0.6, 0.7),
        pixel_size_crop_um=(0.8, 0.9, 1.0),
    )
    np.testing.assert_allclose(np.asarray(rec["A_px"], dtype=float), A_px)
    np.testing.assert_allclose(np.asarray(rec["b_px"], dtype=float), b_px)
