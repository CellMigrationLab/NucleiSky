import json

import numpy as np
from tifffile import TiffFile, imread

from nucleisky3d.export import (
    export_aligned_crop_tiff,
    export_bbox_pair_tiffs_3d,
    warp_crop_to_full_bbox_volume,
    warp_crop_to_full_volume,
)


def _identity_res_with_offset(offset_zyx=(0.0, 0.0, 0.0), bbox=None):
    return {
        "best_scale": 1.0,
        "best_R": np.eye(3, dtype=float),
        "best_t": np.asarray(offset_zyx, dtype=float),
        "best_bbox": bbox,
    }


def test_warp_crop_to_full_bbox_matches_full_volume_subregion():
    img_crop = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    full_shape = (6, 7, 5)
    bbox = (1, 3, 2, 5, 1, 3)

    res = _identity_res_with_offset(offset_zyx=(1.0, 2.0, 1.0), bbox=bbox)

    full = warp_crop_to_full_volume(
        img_crop,
        full_shape_zyx=full_shape,
        pixel_size_full_um=(1.0, 1.0, 1.0),
        pixel_size_crop_um=(1.0, 1.0, 1.0),
        res=res,
        order=0,
    )
    roi, roi_bbox = warp_crop_to_full_bbox_volume(
        img_crop,
        full_shape_zyx=full_shape,
        bbox_zyx=bbox,
        pixel_size_full_um=(1.0, 1.0, 1.0),
        pixel_size_crop_um=(1.0, 1.0, 1.0),
        res=res,
        order=0,
    )

    assert roi_bbox == bbox
    z0, z1, y0, y1, x0, x1 = bbox
    np.testing.assert_allclose(roi, full[z0:z1, y0:y1, x0:x1])


def test_export_aligned_crop_tiff_bbox_writes_roi_and_sidecar(tmp_path):
    img_full = np.zeros((10, 10, 10), dtype=np.float32)
    img_crop = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    bbox = (3, 5, 4, 6, 2, 4)
    res = _identity_res_with_offset(offset_zyx=(3.0, 4.0, 2.0), bbox=bbox)

    out_path = tmp_path / "aligned_bbox.tif"
    export_aligned_crop_tiff(
        img_full=img_full,
        img_crop=img_crop,
        output_path=out_path,
        pixel_size_full_um=(1.0, 1.0, 1.0),
        pixel_size_crop_um=(1.0, 1.0, 1.0),
        res=res,
        export_region="bbox",
        order=0,
    )

    got = imread(out_path)
    assert got.shape == (2, 2, 2)
    np.testing.assert_allclose(got, img_crop)

    meta = json.loads((tmp_path / "aligned_bbox.tif.json").read_text(encoding="utf-8"))
    assert meta["export_region"] == "bbox"
    assert meta["bbox_full_px_z0z1y0y1x0x1"] == list(bbox)
    assert meta["bbox_origin_full_px_zyx"] == [3, 4, 2]

    with TiffFile(out_path) as tif:
        ij_meta = tif.imagej_metadata
        assert ij_meta["export_region"] == "bbox"
        bbox_origin = ij_meta["bbox_origin_full_px_zyx"]
        if isinstance(bbox_origin, str):
            bbox_origin = json.loads(bbox_origin)
        assert bbox_origin == [3, 4, 2]


def test_export_aligned_crop_tiff_bbox_requires_best_bbox(tmp_path):
    img_full = np.zeros((5, 5, 5), dtype=np.float32)
    img_crop = np.ones((2, 2, 2), dtype=np.float32)
    res = _identity_res_with_offset(offset_zyx=(0.0, 0.0, 0.0), bbox=None)

    try:
        export_aligned_crop_tiff(
            img_full=img_full,
            img_crop=img_crop,
            output_path=tmp_path / "bad.tif",
            pixel_size_full_um=(1.0, 1.0, 1.0),
            pixel_size_crop_um=(1.0, 1.0, 1.0),
            res=res,
            export_region="bbox",
        )
    except ValueError as e:
        assert "best_bbox" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_export_bbox_pair_tiffs_3d_alignment_with_synthetic_cube(tmp_path):
    img_full = np.zeros((20, 20, 20), dtype=np.float32)
    img_full[6:12, 7:13, 8:14] = 5.0
    img_full[8:10, 9:11, 10:12] = 12.0

    img_crop = np.asarray(img_full[6:12, 7:13, 8:14])
    record = {
        "best_scale": 1.0,
        "best_R": np.eye(3, dtype=float),
        "best_t": np.array([6.0, 3.5, 4.0], dtype=float),
        "bbox_full_px_z0z1y0y1x0x1": [6, 12, 7, 13, 8, 14],
    }

    out = export_bbox_pair_tiffs_3d(
        img_full_zyx=img_full,
        img_crop_zyx=img_crop,
        record_or_result=record,
        voxel_full_um_zyx=(1.0, 0.5, 0.5),
        voxel_crop_um_zyx=(1.0, 0.5, 0.5),
        out_dir=tmp_path,
        margin_px_zyx=(1, 1, 1),
        prefix="syn_",
    )

    full_roi = imread(out["full_bbox_tif"]).astype(np.float32)
    aligned_roi = imread(out["aligned_crop_bbox_tif"]).astype(np.float32)
    assert full_roi.shape == aligned_roi.shape

    corr = np.corrcoef(full_roi.ravel(), aligned_roi.ravel())[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.99

    with TiffFile(out["full_bbox_tif"]) as tif_full, TiffFile(out["aligned_crop_bbox_tif"]) as tif_aligned:
        assert tif_full.series[0].axes == "ZYX"
        assert tif_aligned.series[0].axes == "ZYX"
        np.testing.assert_allclose(
            [
                tif_full.imagej_metadata["spacing"],
                1.0 / tif_full.pages[0].tags["YResolution"].value[0],
                1.0 / tif_full.pages[0].tags["XResolution"].value[0],
            ],
            [
                tif_aligned.imagej_metadata["spacing"],
                1.0 / tif_aligned.pages[0].tags["YResolution"].value[0],
                1.0 / tif_aligned.pages[0].tags["XResolution"].value[0],
            ],
            rtol=1e-6,
            atol=1e-6,
        )
