import json

import numpy as np
import pandas as pd

from nucleisky3d import pipeline
from nucleisky3d.matching.geometry import bbox_full_px_from_similarity_um_3d


def test_run_adaptive_matching_recomputes_bbox_on_orig_grid(monkeypatch, tmp_path):
    df_full = pd.DataFrame({"centroid_z_px": [1.0], "centroid_y_px": [2.0], "centroid_x_px": [3.0]})
    df_crop = pd.DataFrame({"centroid_z_px": [1.5], "centroid_y_px": [2.5], "centroid_x_px": [3.5]})

    img_full_orig = np.zeros((60, 80, 100), dtype=np.uint16)
    img_crop_orig = np.zeros((12, 16, 20), dtype=np.uint16)
    labels_full = np.zeros((30, 40, 50), dtype=np.int32)
    labels_crop = np.zeros((6, 8, 10), dtype=np.int32)

    best_scale = 1.0
    best_R = np.eye(3, dtype=float)
    best_t = np.array([8.0, 10.0, 12.0], dtype=float)
    vox_full_orig = (0.5, 0.5, 0.5)
    vox_crop_orig = (1.0, 1.0, 1.0)
    vox_full_seg = (1.0, 1.0, 1.0)
    vox_crop_seg = (2.0, 2.0, 2.0)

    seg_bbox = tuple(
        int(v)
        for v in bbox_full_px_from_similarity_um_3d(
            crop_shape_px=labels_crop.shape,
            pixel_size_full_um_zyx=vox_full_seg,
            pixel_size_crop_um_zyx=vox_crop_seg,
            scale=best_scale,
            R_zyx=best_R,
            t_um_zyx=best_t,
            full_shape_px=labels_full.shape,
        )
    )
    orig_bbox = tuple(
        int(v)
        for v in bbox_full_px_from_similarity_um_3d(
            crop_shape_px=img_crop_orig.shape,
            pixel_size_full_um_zyx=vox_full_orig,
            pixel_size_crop_um_zyx=vox_crop_orig,
            scale=best_scale,
            R_zyx=best_R,
            t_um_zyx=best_t,
            full_shape_px=img_full_orig.shape,
        )
    )
    assert seg_bbox != orig_bbox

    captured = {}

    def _fake_run_adaptive(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "success": True,
            "matcher": "pyramid",
            "best_scale": best_scale,
            "best_R": best_R,
            "best_t": best_t,
            "best_bbox": seg_bbox,
            "match_quality": {"frac_inliers": 0.9, "mean_error_um": 0.4, "success": True},
        }, []

    monkeypatch.setattr(pipeline, "run_adaptive_nucleisky_3d", _fake_run_adaptive)

    result_dir = tmp_path / "results"
    pipeline.run_adaptive_matching_and_export_3d(
        df_full=df_full,
        df_crop=df_crop,
        img_full_orig=img_full_orig,
        img_crop_orig=img_crop_orig,
        pixel_size_full_orig_um_zyx=vox_full_orig,
        pixel_size_crop_orig_um_zyx=vox_crop_orig,
        img_full_seg=np.zeros(labels_full.shape, dtype=np.uint16),
        img_crop_seg=np.zeros(labels_crop.shape, dtype=np.uint16),
        pixel_size_full_seg_um_zyx=vox_full_seg,
        pixel_size_crop_seg_um_zyx=vox_crop_seg,
        labels_full=labels_full,
        labels_crop=labels_crop,
        result_dir=str(result_dir),
        save_segmentation_masks=False,
    )

    assert captured["kwargs"]["full_shape_px_zyx"] == img_full_orig.shape
    assert captured["kwargs"]["crop_shape_px_zyx"] == img_crop_orig.shape
    assert tuple(captured["kwargs"]["pixel_size_full_um_zyx"]) == vox_full_orig
    assert tuple(captured["kwargs"]["pixel_size_crop_um_zyx"]) == vox_crop_orig

    transforms_path = result_dir / "matching" / "adaptive_3d" / "exports_adaptive" / "transforms.jsonl"
    line = transforms_path.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["bbox_full_px_z0z1y0y1x0x1"] == list(orig_bbox)
    assert rec["bbox_grid"] == "orig"
