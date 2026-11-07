"""The 2D API workflow example notebook tells the user what it needs and fails with clear messages (#42).

Round alpha.4 (LabConstrictor-Tools#141, MAC-04): the Configuration cell held placeholder paths, so the first run ended in a
bare FileNotFoundError, and the "Helper utilities" cell printed nothing, so it was easy to skip and the next cell raised
"NameError: load_tiff is not defined". The cells are run here with the notebook's own imports, without Jupyter.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import tifffile

matplotlib.use("Agg")

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "NucleiSky2D_API_Workflow_Example"
    / "NucleiSky2D_API_Workflow_Example.ipynb"
)
PLACEHOLDER_LINE = "base_path = PLACEHOLDER_FOLDER\n"


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _code_cell(title: str) -> str:
    found = [
        "".join(c["source"]) for c in _cells() if c["cell_type"] == "code" and "".join(c["source"]).startswith(title)
    ]
    assert len(found) == 1, f"expected one cell starting with {title!r}"
    return found[0]


def _namespace():
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    from nucleisky.nucleisky2d.io import get_pixel_size_um_from_tiff
    from nucleisky.nucleisky2d.preprocess import ij_percentile_normalize

    return {
        "np": np,
        "plt": plt,
        "imread": tifffile.imread,
        "Path": Path,
        "find_boundaries": find_boundaries,
        "get_pixel_size_um_from_tiff": get_pixel_size_um_from_tiff,
        "ij_percentile_normalize": ij_percentile_normalize,
    }


def _run(source: str, namespace: dict, name: str) -> None:
    exec(compile(source, name, "exec"), namespace)


def _configuration(base: Path, reference: str = "reference.tif", query: str = "query.tif", pixel_sizes=None) -> str:
    source = _code_cell("# --- 1. Configuration ---")
    assert source.count(PLACEHOLDER_LINE) == 1
    source = source.replace(PLACEHOLDER_LINE, f"base_path = Path({str(base)!r})\n")
    source = source.replace('"reference_image.tif"', repr(reference)).replace('"crop_image.tif"', repr(query))
    if pixel_sizes:
        source = source.replace("pixel_size_full_um = None\n", f"pixel_size_full_um = {pixel_sizes[0]}\n", 1)
        source = source.replace("pixel_size_crop_um = None\n", f"pixel_size_crop_um = {pixel_sizes[1]}\n", 1)
    return source


def _write_uncalibrated_pair(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tifffile.imwrite(folder / "reference.tif", rng.integers(0, 4000, (64, 64), dtype=np.uint16))
    tifffile.imwrite(folder / "query.tif", rng.integers(0, 4000, (32, 32), dtype=np.uint16))


def test_intro_says_what_the_notebook_needs():
    intro = "".join(_cells()[0]["source"])
    assert "## Before you start" in intro
    for needed in ("lc_tests/make_fixtures.py", "reference.tif", "query.tif", "0.65", "0.325", "Run All Cells", "restart"):
        assert needed in intro, needed


def test_helper_cell_confirms_it_ran(capsys):
    namespace = _namespace()
    _run(_code_cell("# --- Helper utilities"), namespace, "Helper utilities")
    assert "load_tiff" in namespace
    assert "Helper functions defined" in capsys.readouterr().out


def test_unedited_configuration_names_the_placeholder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="placeholder 'path/to/your/data'"):
        _run(_code_cell("# --- 1. Configuration ---"), _namespace(), "Configuration")
    assert not (tmp_path / "nucleisky_output").exists()


def test_missing_image_is_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"Reference image not found: .*reference\.tif"):
        _run(_configuration(tmp_path / "ns"), _namespace(), "Configuration")


def test_missing_pixel_size_is_asked_for_not_guessed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_uncalibrated_pair(tmp_path / "ns")
    with pytest.raises(ValueError, match="type it as pixel_size_full_um"):
        _run(_configuration(tmp_path / "ns"), _namespace(), "Configuration")


def test_configuration_and_load_run_with_typed_pixel_sizes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_uncalibrated_pair(tmp_path / "ns")
    namespace = _namespace()
    _run(_code_cell("# --- Helper utilities"), namespace, "Helper utilities")
    _run(_configuration(tmp_path / "ns", pixel_sizes=(0.65, 0.325)), namespace, "Configuration")
    assert (namespace["pixel_size_full_um"], namespace["pixel_size_crop_um"]) == (0.65, 0.325)
    assert (tmp_path / "nucleisky_output").is_dir()
    _run(_code_cell("# --- 2. Load & Display ---"), namespace, "Load & Display")
    assert namespace["img_full"].shape == (64, 64)
    assert namespace["img_crop"].shape == (32, 32)


def test_load_without_helper_cell_says_what_to_run(tmp_path):
    namespace = _namespace()
    namespace.update(full_image_path=tmp_path / "reference.tif", crop_image_path=tmp_path / "query.tif")
    with pytest.raises(NameError, match="Not defined yet: load_tiff, show_side_by_side. Run every cell above this one"):
        _run(_code_cell("# --- 2. Load & Display ---"), namespace, "Load & Display")
