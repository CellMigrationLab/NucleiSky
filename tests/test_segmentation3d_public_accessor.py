from pathlib import Path
import importlib

def test_3d_segmentation_uses_public_segmentor_accessor() -> None:
    source = Path("src/nucleisky/nucleisky3d/segmentation.py").read_text()

    assert "from nucleisky2d.segmentation import get_global_segmentor" in source
    assert "_GLOBAL_SEGMENTOR" not in source

def test_3d_segmentation_public_module_imports() -> None:
    mod = importlib.import_module("nucleisky3d.segmentation")
    assert hasattr(mod, "segment_nuclei_2p5d")