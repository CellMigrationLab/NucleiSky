from pathlib import Path

from nucleisky2d.io import _is_zarr_store_path


def test_detects_zarr_extension_without_markers(tmp_path: Path):
    zarr_dir = tmp_path / "sample.zarr"
    zarr_dir.mkdir()

    assert _is_zarr_store_path(zarr_dir)


def test_detects_marker_based_directory_without_zarr_suffix(tmp_path: Path):
    zarr_like_dir = tmp_path / "image_folder"
    zarr_like_dir.mkdir()
    (zarr_like_dir / "zarr.json").write_text("{}")

    assert _is_zarr_store_path(zarr_like_dir)


def test_non_zarr_directory_not_detected(tmp_path: Path):
    plain_dir = tmp_path / "plain_folder"
    plain_dir.mkdir()

    assert not _is_zarr_store_path(plain_dir)
