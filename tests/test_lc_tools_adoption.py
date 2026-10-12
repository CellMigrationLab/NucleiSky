"""The tool settings that mirror the notebook (segmentation settings, outlines in original pixels, region filter)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("labconstrictor_tools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labconstrictor_tools import ToolError  # noqa: E402
from nucleisky_lc_tools import _inside_region, _outlines, _segmentation_settings  # noqa: E402

DEFAULTS = dict(
    device="auto", threshold_method="otsu", blur_sigma=1.0, min_area_px=5, watershed_split=True, peak_distance_px=5,
    instanseg_model="brightfield_nuclei", instanseg_target="nuclei", instanseg_cleanup_fragments=True, instanseg_mode="auto",
    instanseg_pixel_size_um=None, cellpose_diameter_px=None, cellpose_flow_threshold=0.4, cellpose_cellprob_threshold=0.0,
    cellpose_min_size_px=15, cellpose_batch_size=1, cellpose_tile_size_px=None, cellpose_tile_overlap=None,
    cellpose_normalize=True, cellpose_invert=False,
)  # fmt: skip


def test_only_what_was_set_reaches_cellpose():
    settings = _segmentation_settings(**DEFAULTS)
    assert settings["cellpose"] == {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "min_size": 15, "batch_size": 1, "normalize": True, "invert": False}
    custom = _segmentation_settings(**{**DEFAULTS, "cellpose_diameter_px": 30.0, "cellpose_tile_size_px": 256, "cellpose_tile_overlap": 0.2, "cellpose_invert": True})
    assert custom["cellpose"]["diameter"] == 30.0 and custom["cellpose"]["tile_size"] == 256 and custom["cellpose"]["overlap"] == 0.2
    assert custom["cellpose"]["invert"] is True


def test_the_settings_use_the_names_the_back_ends_take():
    from nucleisky.nucleisky2d.segmentation import Segmentor
    import inspect

    cellpose = set(inspect.signature(Segmentor.segment_cellpose).parameters) - {"self", "img2d", "pretrained_model"}
    assert set(_segmentation_settings(**{**DEFAULTS, "cellpose_diameter_px": 1.0, "cellpose_tile_size_px": 64, "cellpose_tile_overlap": 0.1})["cellpose"]) <= cellpose
    instanseg = set(inspect.signature(Segmentor.segment_instanseg).parameters) - {"self", "img"}
    assert set(_segmentation_settings(**{**DEFAULTS, "instanseg_pixel_size_um": 0.5})["instanseg"]) - {"model_name"} <= instanseg | {"pixel_size_um"}
    assert _segmentation_settings(**{**DEFAULTS, "instanseg_pixel_size_um": 0.5})["instanseg"]["pixel_size_um"] == 0.5
    assert "pixel_size_um" not in _segmentation_settings(**DEFAULTS)["instanseg"]
    assert _segmentation_settings(**{**DEFAULTS, "instanseg_mode": "medium"})["instanseg"]["mode"] == "medium"


def test_outlines_are_in_original_pixels_and_only_the_kept_nuclei():
    pytest.importorskip("skimage")
    labels = np.zeros((40, 40), int)
    labels[10:20, 10:20] = 1
    labels[25:35, 25:35] = 2
    shapes = _outlines(labels, scale=2.0, kept_labels=[2])  # segmented on a copy twice as big: divide by the scale
    assert [f["properties"]["label"] for f in shapes["features"]] == [2]
    xs = [x for ring in shapes["features"][0]["geometry"]["coordinates"] for x, _ in ring]
    assert 12 < min(xs) < 13.5 and 17 < max(xs) < 18.5  # 25..35 / 2


def test_region_keeps_the_nuclei_inside_and_refuses_an_empty_result():
    frame = pd.DataFrame({"label": [1, 2, 3], "centroid_y_px_orig": [5.0, 50.0, 90.0], "centroid_x_px_orig": [5.0, 50.0, 90.0]})
    image = np.zeros((100, 100))
    region = np.zeros((100, 100), int)
    region[40:60, 40:60] = 1
    assert list(_inside_region("reference", frame, region, image)["label"]) == [2]
    other = np.zeros((100, 100), int)
    other[70:75, 10:15] = 1
    with pytest.raises(ToolError, match="inside the selected region"):
        _inside_region("reference", frame, other, image)
    with pytest.raises(ToolError, match="empty"):
        _inside_region("reference", frame, np.zeros((100, 100), int), image)
    with pytest.raises(ToolError, match="size of the image"):
        _inside_region("reference", frame, np.ones((10, 10), int), image)
