from pathlib import Path


def test_3d_segmentation_uses_public_segmentor_accessor() -> None:
    source = Path("src/nucleisky3d/segmentation.py").read_text()

    assert "from nucleisky2d.segmentation import get_global_segmentor" in source
    assert "_GLOBAL_SEGMENTOR" not in source
