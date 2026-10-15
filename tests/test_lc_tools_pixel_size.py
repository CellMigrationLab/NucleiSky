"""Missing pixel-size calibration is a refusal in the tools, never a silent default (shared rule: unknown calibration blocks
the run, a bare scale is unknown). The notebook pipeline is not touched by this."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("labconstrictor_tools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labconstrictor_tools import ToolError  # noqa: E402
from nucleisky_lc_tools import relocalize  # noqa: E402

IMAGE = np.zeros((16, 16), dtype=np.uint16)


def _run(**sizes):
    return relocalize(IMAGE, IMAGE, **sizes)


def test_pixel_sizes_have_no_numeric_default_and_stay_linked_to_the_images():
    import importlib

    from labconstrictor_tools.introspection import describe_tools

    importlib.import_module("nucleisky_lc_tools")
    tool = {t["id"]: t for t in describe_tools("nucleisky_lc_tools")["tools"]}["relocalize"]
    inputs = {p["name"]: p for p in tool["inputs"]}
    for name, image in (("reference_pixel_size_um", "reference"), ("query_pixel_size_um", "query")):
        assert inputs[name].get("default") is None, "%s must not have a numeric default (0.65 / 0.325 were an author's example)" % name
        assert inputs[name]["pixel_size_of"] == image  # the host prefills it from the image calibration
        assert inputs[name]["unit"] == "um/px"


@pytest.mark.parametrize(
    "sizes, code, words",
    [
        ({"query_pixel_size_um": 0.325}, "missing_parameter", ["reference image has no calibration", "reference_pixel_size_um"]),
        ({"reference_pixel_size_um": 0.65}, "missing_parameter", ["query image has no calibration", "query_pixel_size_um"]),
        ({}, "missing_parameter", ["reference image and the query image have no calibration", "reference_pixel_size_um", "query_pixel_size_um"]),
        ({"reference_pixel_size_um": None, "query_pixel_size_um": None}, "missing_parameter", ["no calibration"]),
        ({"reference_pixel_size_um": 0.0, "query_pixel_size_um": 0.325}, "invalid_parameter", ["reference pixel size must be a number above 0"]),
        ({"reference_pixel_size_um": 0.65, "query_pixel_size_um": -1.0}, "invalid_parameter", ["query pixel size must be a number above 0"]),
        ({"reference_pixel_size_um": float("nan"), "query_pixel_size_um": 0.325}, "invalid_parameter", ["reference pixel size"]),
        ({"reference_pixel_size_um": 0.65, "query_pixel_size_um": float("inf")}, "invalid_parameter", ["query pixel size"]),
    ],
)
def test_missing_or_invalid_pixel_size_is_refused_before_any_work(sizes, code, words):
    with pytest.raises(ToolError) as caught:
        _run(**sizes)
    assert caught.value.code == code
    message = str(caught.value)
    for word in words:
        assert word in message
    assert "type it" in message and "Image > Properties" in message  # says how to supply it


def test_explicit_pixel_sizes_pass_the_calibration_check():
    from nucleisky_lc_tools import _require_pixel_sizes

    assert _require_pixel_sizes(0.65, 0.325) == (0.65, 0.325)
    assert _require_pixel_sizes(1, 2) == (1.0, 2.0)
    # with valid sizes the run goes on to the images: 2D check passes, an all-zero image has no nuclei (not a calibration error)
    with pytest.raises(ToolError) as caught:
        _run(reference_pixel_size_um=0.65, query_pixel_size_um=0.325)
    assert caught.value.code != "missing_parameter" and "calibration" not in str(caught.value)


@pytest.mark.parametrize("which", ["reference", "query"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_an_image_with_nan_or_infinite_pixels_is_refused_not_silently_repaired(which, bad):
    damaged = np.zeros((16, 16), dtype=np.float32)
    damaged[3, 4] = bad
    clean = np.zeros((16, 16), dtype=np.float32)
    reference, query = (damaged, clean) if which == "reference" else (clean, damaged)
    with pytest.raises(ToolError) as caught:
        relocalize(reference, query, reference_pixel_size_um=0.65, query_pixel_size_um=0.325)
    assert caught.value.code == "non_finite_pixels" and which in str(caught.value)


def test_the_3d_notebook_imports_every_name_it_calls():
    import json
    import re

    path = Path(__file__).resolve().parents[1] / "notebooks" / "NucleiSky3DApp" / "NucleiSky3DApp.ipynb"
    code = "\n".join("".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code")
    imported = set(re.findall(r"import\s+[^\n]*", code)) and " ".join(re.findall(r"from nucleisky[^\n]*import[^\n]*", code))
    for name in set(re.findall(r"\b(plot_warp_overlay3[dD])\(", code)):
        assert name in imported, "the notebook calls %s but does not import it (NameError at run time)" % name
