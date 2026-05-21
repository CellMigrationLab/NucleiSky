import json
from pathlib import Path

import numpy as np
import pytest

from nucleisky2d.io import load_nucleisky_transform, load_transforms_any, validate_transform_record
from nucleisky3d.io import load_nucleisky_transform_3d, load_transforms_any_3d


def _map_2d_um(points, scale, R_yx, t_um_yx):
    pts = np.asarray(points, dtype=float)
    R = np.asarray(R_yx, dtype=float).reshape(2, 2)
    t = np.asarray(t_um_yx, dtype=float).reshape(2)
    return (scale * (R @ pts.T)).T + t


def _map_3d_um(points, scale, R_zyx, t_um_zyx):
    pts = np.asarray(points, dtype=float)
    R = np.asarray(R_zyx, dtype=float).reshape(3, 3)
    t = np.asarray(t_um_zyx, dtype=float).reshape(3)
    return (scale * (R @ pts.T)).T + t


@pytest.mark.geometry
def test_2d_canonical_and_extra_field_records_load_and_replay(tmp_path):
    rec = {
        "scale": 1.1,
        "R_yx": [[0.0, -1.0], [1.0, 0.0]],
        "t_um_yx": [4.0, -2.0],
        "pixel_size_full_um": 0.8,
        "pixel_size_crop_um": 1.2,
        "A_px": [[0.0, -1.65], [1.65, 0.0]],
        "b_px": [5.0, -2.5],
        "bbox_full_px_y0y1x0x1": [2, 12, 3, 16],
        "future_field": {"v": 1},
    }
    p = tmp_path / "r2d.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    loaded = load_nucleisky_transform(p)
    ok, problems = validate_transform_record(loaded)
    assert ok, problems
    assert loaded["scale"] == pytest.approx(rec["scale"])
    assert loaded["t_um_yx"] == pytest.approx(rec["t_um_yx"])
    assert loaded["bbox_full_px_y0y1x0x1"] == [2, 12, 3, 16]

    src = np.array([[1.0, 2.0], [3.0, 4.5], [0.0, -1.0]], dtype=float)
    mapped_ref = _map_2d_um(src, rec["scale"], rec["R_yx"], rec["t_um_yx"])
    mapped_loaded = _map_2d_um(src, loaded["scale"], loaded["R_yx"], loaded["t_um_yx"])
    np.testing.assert_allclose(mapped_ref, mapped_loaded, atol=1e-12)


@pytest.mark.geometry
def test_2d_schema_guard_and_malformed_records(tmp_path):
    canonical = {
        "scale": 1.0,
        "R_yx": [[1.0, 0.0], [0.0, 1.0]],
        "t_um_yx": [0.0, 0.0],
        "pixel_size_full_um": 1.0,
        "pixel_size_crop_um": 1.0,
        "A_px": [[1.0, 0.0], [0.0, 1.0]],
        "b_px": [0.0, 0.0],
        "bbox_full_px_y0y1x0x1": [0, 10, 0, 11],
    }
    assert {"A_px", "b_px", "pixel_size_full_um", "pixel_size_crop_um"}.issubset(canonical)

    bads = [
        {"A_px": [[1, 0], [0, 1]], "pixel_size_full_um": 1.0, "pixel_size_crop_um": 1.0},  # missing b_px
        {**canonical, "scale": "nanstr"},
        {**canonical, "R_yx": [[1, 0, 0], [0, 1, 0]]},
        {**canonical, "pixel_size_full_um": 0.0},
        {**canonical, "pixel_size_crop_um": -1.0},
    ]
    for i, b in enumerate(bads):
        p = tmp_path / f"bad2d_{i}.json"
        p.write_text(json.dumps(b), encoding="utf-8")
        with pytest.raises(ValueError):
            load_nucleisky_transform(p)


@pytest.mark.geometry
def test_2d_invalid_json_jsonl_and_malformed_line_fails(tmp_path):
    (tmp_path / "empty.json").write_text("", encoding="utf-8")
    with pytest.raises(Exception):
        load_nucleisky_transform(tmp_path / "empty.json")

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"ok":1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(Exception):
        load_transforms_any(str(bad_jsonl))


@pytest.mark.geometry
def test_3d_json_and_jsonl_compatibility_with_legacy_aliases(tmp_path):
    legacy = {
        "best_scale": 1.0,
        "best_R": np.eye(3).tolist(),
        "best_t": [1.0, -2.0, 3.0],
        "voxel_size_full_um_zyx": [2.0, 1.0, 0.5],
        "voxel_size_crop_um_zyx": [2.0, 1.0, 0.5],
        "best_bbox": [1, 6, 2, 8, 3, 9],
    }
    p_jsonl = tmp_path / "mix3d.jsonl"
    p_jsonl.write_text("\n".join([json.dumps(legacy), json.dumps({**legacy, "extra": 9})]) + "\n", encoding="utf-8")
    recs = load_transforms_any_3d(str(p_jsonl))
    assert len(recs) == 2
    for rec in recs:
        for k in ("scale", "R_zyx", "t_um_zyx", "pixel_size_full_um_zyx", "pixel_size_crop_um_zyx", "A_px", "b_px"):
            assert k in rec
        assert rec["bbox_full_px_z0z1y0y1x0x1"] == [1, 6, 2, 8, 3, 9]

    p_json = tmp_path / "legacy3d.json"
    p_json.write_text(json.dumps({
        "A_px": np.eye(3).tolist(),
        "b_px": [0, 0, 0],
        "pixel_size_full_um_zyx": [2.0, 1.0, 0.5],
        "pixel_size_crop_um_zyx": [2.0, 1.0, 0.5],
        **legacy,
    }), encoding="utf-8")
    rec_json = load_nucleisky_transform_3d(p_json)
    assert rec_json["pixel_size_full_um_zyx"] == [2.0, 1.0, 0.5]


@pytest.mark.geometry
def test_3d_replay_equivalence_and_schema_guard(tmp_path):
    rec = {
        "scale": 0.9,
        "R_zyx": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "t_um_zyx": [2.0, 3.0, -4.0],
        "pixel_size_full_um_zyx": [2.5, 0.8, 0.4],
        "pixel_size_crop_um_zyx": [2.5, 0.8, 0.4],
        "A_px": np.eye(3).tolist(),
        "b_px": [0.0, 0.0, 0.0],
        "bbox_full_px_z0z1y0y1x0x1": [0, 7, 2, 9, 3, 11],
    }
    p = tmp_path / "r3d.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    loaded = load_nucleisky_transform_3d(p)
    stable = {"scale", "R_zyx", "t_um_zyx", "pixel_size_full_um_zyx", "pixel_size_crop_um_zyx", "A_px", "b_px"}
    assert stable.issubset(loaded.keys())

    src = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]], dtype=float)
    np.testing.assert_allclose(
        _map_3d_um(src, rec["scale"], rec["R_zyx"], rec["t_um_zyx"]),
        _map_3d_um(src, loaded["scale"], loaded["R_zyx"], loaded["t_um_zyx"]),
        atol=1e-12,
    )


@pytest.mark.geometry
def test_3d_malformed_records_fail_or_validate_bad(tmp_path):
    bad_records = [
        {"pixel_size_full_um_zyx": [1, 1, 1], "pixel_size_crop_um_zyx": [1, 1, 1], "b_px": [0, 0, 0]},
    ]
    for i, rec in enumerate(bad_records):
        p = tmp_path / f"bad3d_{i}.json"
        p.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(ValueError):
            load_nucleisky_transform_3d(p)

    strict_bad_cases = [
        {"A_px": np.eye(3).tolist(), "b_px": [0, 0, 0], "pixel_size_full_um_zyx": [0, 1, 1], "pixel_size_crop_um_zyx": [1, 1, 1]},
        {"A_px": np.eye(2).tolist(), "b_px": [0, 0], "pixel_size_full_um_zyx": [1, 1, 1], "pixel_size_crop_um_zyx": [1, 1, 1]},
        {"A_px": np.eye(3).tolist(), "b_px": [0, 0, 0], "pixel_size_full_um_zyx": [1, 1], "pixel_size_crop_um_zyx": [1, 1, 1]},
    ]
    for i, rec in enumerate(strict_bad_cases):
        p = tmp_path / f"strict_bad3d_{i}.json"
        p.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(ValueError):
            load_transforms_any_3d(str(p))

    bad_mix = tmp_path / "bad_mix3d.jsonl"
    bad_mix.write_text(json.dumps({"ok": 1}) + "\n{" + "\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_transforms_any_3d(str(bad_mix))


@pytest.mark.geometry
@pytest.mark.parametrize(
    ("bad_patch", "msg"),
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
def test_2d_parametrized_malformed_record_errors(tmp_path, bad_patch, msg):
    base = {
        "scale": 1.0,
        "R_yx": [[1.0, 0.0], [0.0, 1.0]],
        "t_um_yx": [0.0, 0.0],
        "pixel_size_full_um": 1.0,
        "pixel_size_crop_um": 1.0,
        "A_px": [[1.0, 0.0], [0.0, 1.0]],
        "b_px": [0.0, 0.0],
        "bbox_full_px_y0y1x0x1": [0, 10, 0, 10],
    }
    rec = {**base, **bad_patch}
    p = tmp_path / "bad2d_param.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        load_nucleisky_transform(p)


@pytest.mark.geometry
@pytest.mark.parametrize(
    ("bad_patch", "msg"),
    [
        ({"bbox_full_px_z0z1y0y1x0x1": [5, 4, 0, 1, 0, 1]}, "bbox_full_px_z0z1y0y1x0x1"),
        ({"bbox_full_px_z0z1y0y1x0x1": [0, 1, 2]}, "bbox_full_px_z0z1y0y1x0x1"),
        ({"pixel_size_full_um_zyx": [0.0, 1.0, 1.0]}, "pixel_size_full_um_zyx"),
        ({"pixel_size_crop_um_zyx": [1.0, -1.0, 1.0]}, "pixel_size_crop_um_zyx"),
        ({"pixel_size_full_um_zyx": [1.0, 1.0]}, "pixel_size_full_um_zyx"),
        ({"scale": float("nan")}, "scale"),
        ({"t_um_zyx": [float("inf"), 0.0, 0.0]}, "t_um_zyx"),
        ({"R_zyx": [[1, 0], [0, 1]]}, "R_zyx"),
        ({"A_px": [[1, 0], [0, 1]]}, "A_px"),
        ({"b_px": [0.0, 1.0]}, "b_px"),
    ],
)
def test_3d_parametrized_malformed_record_errors(tmp_path, bad_patch, msg):
    base = {
        "scale": 1.0,
        "R_zyx": np.eye(3).tolist(),
        "t_um_zyx": [0.0, 0.0, 0.0],
        "pixel_size_full_um_zyx": [1.0, 1.0, 1.0],
        "pixel_size_crop_um_zyx": [1.0, 1.0, 1.0],
        "A_px": np.eye(3).tolist(),
        "b_px": [0.0, 0.0, 0.0],
        "bbox_full_px_z0z1y0y1x0x1": [0, 3, 0, 4, 0, 5],
    }
    rec = {**base, **bad_patch}
    p = tmp_path / "bad3d_param.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        load_nucleisky_transform_3d(p)
