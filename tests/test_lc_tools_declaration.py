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
    assert {o["type"] for o in tool["outputs"]} == {"affine", "image", "values"}
    assert "nucleisky" not in sys.modules, "importing the declarations must not import the (slow) NucleiSky package"
