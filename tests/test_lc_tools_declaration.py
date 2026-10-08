"""The tool declarations must stay valid and cheap to import (they are read to list tools in Napari/Fiji)."""

import sys

import pytest

pytest.importorskip("labconstrictor_tools")


def test_declarations_are_valid_and_light():
    import importlib

    from labconstrictor_tools.introspection import describe_tools

    importlib.import_module("nucleisky_lc_tools")
    schema = describe_tools("nucleisky_lc_tools")
    tool = {t["id"]: t for t in schema["tools"]}["relocalize"]
    assert [p["name"] for p in tool["inputs"][:2]] == ["reference", "query"]
    assert {o["type"] for o in tool["outputs"]} == {"affine", "image", "values", "points", "shapes"}
    inputs = {p["name"]: p for p in tool["inputs"]}
    assert inputs["blur_sigma"]["widget"] == "slider"  # a FloatSlider (0 to 5) in the notebook too
    assert inputs["reference_region"]["region_of"] == "reference" and inputs["query_region"]["region_of"] == "query"
    assert inputs["device"]["choices_from"]["tool"] == "list_devices"
    assert {"cellpose_diameter_px", "cellpose_flow_threshold", "instanseg_mode", "instanseg_pixel_size_um"} <= set(inputs)
    assert "nucleisky" not in sys.modules, "importing the declarations must not import the (slow) NucleiSky package"
