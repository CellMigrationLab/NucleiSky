import json

import numpy as np
import pytest

from nucleisky2d.io import load_nucleisky_transform, load_transforms_any


def _valid_2d_record():
    return {
        "scale": 1.0,
        "R_yx": [[1.0, 0.0], [0.0, 1.0]],
        "t_um_yx": [0.0, 0.0],
        "pixel_size_full_um": 1.0,
        "pixel_size_crop_um": 1.0,
        "A_px": [[1.0, 0.0], [0.0, 1.0]],
        "b_px": [0.0, 0.0],
        "bbox_full_px_y0y1x0x1": [0, 10, 0, 10],
    }


@pytest.mark.geometry
def test_default_jsonl_loading_remains_permissive(tmp_path):
    r0 = _valid_2d_record()
    r1 = {"future": 1, **r0}
    r2 = {**r0, "bbox_full_px_y0y1x0x1": [5, 4, 0, 1]}  # invalid ordering, permissive mode keeps it
    p = tmp_path / "mix.jsonl"
    p.write_text("\n".join([json.dumps(r0), json.dumps(r1), json.dumps(r2)]) + "\n", encoding="utf-8")

    recs = load_transforms_any(str(p))
    assert len(recs) == 3
    assert recs[1]["future"] == 1
    assert recs[2]["bbox_full_px_y0y1x0x1"] == [5, 4, 0, 1]


@pytest.mark.geometry
def test_strict_jsonl_accepts_valid_and_extra_fields(tmp_path):
    r0 = _valid_2d_record()
    r1 = {"future": {"v": 2}, **r0}
    p = tmp_path / "ok.jsonl"
    p.write_text("\n".join([json.dumps(r0), json.dumps(r1)]) + "\n", encoding="utf-8")

    recs = load_transforms_any(str(p), strict=True)
    assert len(recs) == 2
    assert recs[1]["future"]["v"] == 2


@pytest.mark.geometry
@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"bbox_full_px_y0y1x0x1": [5, 4, 0, 1]}, "bbox_full_px_y0y1x0x1"),
        ({"bbox_full_px_y0y1x0x1": [0, 1, 2]}, "bbox_full_px_y0y1x0x1"),
        ({"pixel_size_full_um": 0.0}, "pixel_size_full_um"),
        ({"pixel_size_crop_um": -1.0}, "pixel_size_crop_um"),
        ({"scale": float("nan")}, "scale"),
        ({"t_um_yx": [float("inf"), 0.0]}, "t_um_yx"),
        ({"R_yx": [[1, 0, 0], [0, 1, 0]]}, "R_yx"),
        ({"A_px": [[1, 0, 0], [0, 1, 0]]}, "A_px"),
        ({"b_px": [0.0, 1.0, 2.0]}, "b_px"),
    ],
)
def test_strict_jsonl_rejects_invalid_records_with_line_context(tmp_path, patch, msg):
    good = _valid_2d_record()
    bad = {**good, **patch}
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2"):
        load_transforms_any(str(p), strict=True)
    with pytest.raises(ValueError, match=msg):
        load_transforms_any(str(p), strict=True)


@pytest.mark.geometry
def test_strict_jsonl_malformed_line_reports_file_and_line(tmp_path):
    p = tmp_path / "badsyntax.jsonl"
    p.write_text(json.dumps(_valid_2d_record()) + "\n{" + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2"):
        load_transforms_any(str(p), strict=True)
    with pytest.raises(ValueError, match=r"Invalid JSONL"):
        load_transforms_any(str(p), strict=True)


@pytest.mark.geometry
def test_strict_jsonl_parity_with_single_record_loader(tmp_path):
    good = _valid_2d_record()
    bad = {**good, "bbox_full_px_y0y1x0x1": [9, 2, 0, 1]}

    p_json = tmp_path / "one.json"
    p_json.write_text(json.dumps(good), encoding="utf-8")
    one = load_nucleisky_transform(p_json)

    p_jsonl = tmp_path / "one.jsonl"
    p_jsonl.write_text(json.dumps(good) + "\n", encoding="utf-8")
    many = load_transforms_any(str(p_jsonl), strict=True)
    assert many[0]["scale"] == pytest.approx(one["scale"])

    p_bad_json = tmp_path / "bad.json"
    p_bad_json.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="bbox_full_px_y0y1x0x1"):
        load_nucleisky_transform(p_bad_json)

    p_bad_jsonl = tmp_path / "bad.jsonl"
    p_bad_jsonl.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bbox_full_px_y0y1x0x1"):
        load_transforms_any(str(p_bad_jsonl), strict=True)
