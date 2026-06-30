import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.geometry
def test_public_api_import_stability_and_all_exports():
    import nucleisky
    import nucleisky2d
    import nucleisky3d

    # Top-level package exports subpackages without importing optional backends eagerly.
    assert hasattr(nucleisky, "nucleisky2d")
    assert hasattr(nucleisky, "nucleisky3d")
    assert set(nucleisky.__all__) == {"nucleisky2d", "nucleisky3d"}

    # 2D documented workflow/public helpers.
    assert hasattr(nucleisky2d, "NucleiSky")
    assert hasattr(nucleisky2d, "run_adaptive_nucleisky")
    assert hasattr(nucleisky2d, "run_adaptive_matching_and_export")
    assert hasattr(nucleisky2d, "get_pixel_size_um_from_tiff")

    # 3D documented workflow/public helpers.
    required_3d = {
        "NucleiSky3D",
        "run_adaptive_nucleisky_3d",
        "run_adaptive_matching_and_export_3d",
        "save_nucleisky_transform_3d",
        "load_nucleisky_transform_3d",
        "load_transforms_any_3d",
        "get_voxel_size_um_from_tiff",
        "require_voxel_size_um_zyx",
        "run_geometric_hashing_matching_3d_um",
        "run_pyramid_based_matching_um",
    }
    for name in required_3d:
        assert hasattr(nucleisky3d, name), f"missing public symbol {name}"
    assert set(required_3d).issubset(set(nucleisky3d.__all__))


@pytest.mark.geometry
def test_minimal_documented_config_construction_and_invalid_values():
    from nucleisky2d.pipeline import NucleiSky
    from nucleisky3d.pipeline import NucleiSky3D

    pts2 = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 2.0], [5.0, 5.0]], dtype=float)
    out2 = NucleiSky(
        centroids_crop_um=pts2,
        centroids_full_um=pts2.copy(),
        img_full=np.zeros((16, 16), dtype=np.float32),
        img_crop=np.zeros((16, 16), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=1.0,
        pixel_size_crop_um=1.0,
        matcher="hashing",
        matcher_kwargs={"n_iters": 200, "random_state": 1},
    )
    assert isinstance(out2, dict)
    assert "success" in out2

    with pytest.raises(ValueError):
        NucleiSky(
            centroids_crop_um=pts2,
            centroids_full_um=pts2,
            img_full=np.zeros((16, 16), dtype=np.float32),
            img_crop=np.zeros((16, 16), dtype=np.float32),
            ij_percentile_normalize=False,
            pixel_size_full_um=0.0,
            pixel_size_crop_um=1.0,
            matcher="hashing",
        )

    pts3 = np.array([[1.0, 2.0, 3.0], [2.5, 1.5, 4.0], [4.0, 3.0, 2.0], [6.0, 4.0, 5.0]], dtype=float)
    out3 = NucleiSky3D(
        centroids_crop_um=pts3,
        centroids_full_um=pts3.copy(),
        full_shape_px_zyx=(12, 14, 16),
        crop_shape_px_zyx=(12, 14, 16),
        pixel_size_full_um_zyx=(2.0, 1.0, 0.5),
        pixel_size_crop_um_zyx=(2.0, 1.0, 0.5),
        matcher="hashing",
        matcher_kwargs={"n_iters": 200, "random_state": 2},
    )
    assert isinstance(out3, dict)
    assert "success" in out3


@pytest.mark.integration
def test_docs_style_2d_and_3d_smoke_workflows(tmp_path):
    from nucleisky2d.export import export_aligned_dataset
    from nucleisky2d.io import load_nucleisky_transform, save_nucleisky_transform
    from nucleisky2d.pipeline import NucleiSky
    from nucleisky3d.export import export_aligned_crop_tiff
    from nucleisky3d.io import load_nucleisky_transform_3d, save_nucleisky_transform_3d
    from nucleisky3d.pipeline import NucleiSky3D

    full2 = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 12.0], [8.0, 9.0], [13.0, 4.0]], dtype=float)
    crop2 = full2[:4] + np.array([2.0, -1.0])
    out2 = NucleiSky(
        centroids_crop_um=crop2,
        centroids_full_um=full2,
        img_full=np.zeros((64, 64), dtype=np.float32),
        img_crop=np.zeros((40, 40), dtype=np.float32),
        ij_percentile_normalize=False,
        pixel_size_full_um=0.7,
        pixel_size_crop_um=0.7,
        matcher="hashing",
        matcher_kwargs={"n_iters": 1200, "random_state": 7},
    )
    rec2 = save_nucleisky_transform(out2, tmp_path / "t2d.json", pixel_size_full_um=0.7, pixel_size_crop_um=0.7, require_success=True)
    rec2_loaded = load_nucleisky_transform(tmp_path / "t2d.json")
    assert rec2_loaded["pixel_size_full_um"] == pytest.approx(0.7)
    paths2 = export_aligned_dataset(
        rec2,
        out_dir=tmp_path / "exp2d",
        img_full=np.zeros((64, 64), np.float32),
        img_crop=np.zeros((40, 40), np.float32),
        pixel_size_full_um=0.7,
        pixel_size_crop_um=0.7,
        export_region="roi",
    )
    assert Path(paths2["aligned_on_full_px"]).exists()

    full3 = np.array([[0.0, 0.0, 0.0], [3.0, 6.0, 2.0], [6.0, 1.0, 5.0], [8.0, 7.0, 4.0], [10.0, 3.0, 9.0]], dtype=float)
    crop3 = full3[:4] + np.array([1.0, -2.0, 3.0])
    out3 = NucleiSky3D(
        centroids_crop_um=crop3,
        centroids_full_um=full3,
        full_shape_px_zyx=(24, 30, 34),
        crop_shape_px_zyx=(20, 22, 24),
        pixel_size_full_um_zyx=(2.2, 0.9, 0.4),
        pixel_size_crop_um_zyx=(2.2, 0.9, 0.4),
        matcher="hashing",
        matcher_kwargs={"n_iters": 1400, "random_state": 9},
    )
    rec3 = save_nucleisky_transform_3d(out3, tmp_path / "t3d.json", pixel_size_full_um_zyx=(2.2, 0.9, 0.4), pixel_size_crop_um_zyx=(2.2, 0.9, 0.4), require_success=True)
    rec3_loaded = load_nucleisky_transform_3d(tmp_path / "t3d.json")
    assert tuple(rec3_loaded["pixel_size_full_um_zyx"]) == pytest.approx((2.2, 0.9, 0.4))
    p3 = export_aligned_crop_tiff(
        img_full=np.zeros((24, 30, 34), dtype=np.float32),
        img_crop=np.zeros((20, 22, 24), dtype=np.float32),
        output_path=tmp_path / "roi3d.tif",
        pixel_size_full_um=(2.2, 0.9, 0.4),
        pixel_size_crop_um=(2.2, 0.9, 0.4),
        res=out3,
        export_region="bbox",
    )
    assert Path(p3).exists()


@pytest.mark.geometry
def test_notebooks_are_parseable_and_reference_public_api():
    nb_dir = Path("notebooks")
    notebooks = sorted(nb_dir.glob("*/*.ipynb"))
    assert notebooks, "expected at least one notebook under notebooks/"

    api_tokens = ["nucleisky", "nucleisky2d", "nucleisky3d", "NucleiSky", "NucleiSky3D"]
    for nb in notebooks:
        data = json.loads(nb.read_text(encoding="utf-8"))
        assert isinstance(data.get("cells"), list)
        assert isinstance(data.get("metadata"), dict)
        assert data.get("nbformat", 0) >= 4

        cell_sources = []
        for c in data["cells"]:
            src = c.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            cell_sources.append(str(src))
        joined = "\n".join(cell_sources)
        assert any(tok in joined for tok in api_tokens), f"{nb.name} does not reference public API"


@pytest.mark.geometry
def test_no_documented_console_entrypoints_in_pyproject():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" not in text


@pytest.mark.integration
def test_user_error_contracts_dimensionality_and_missing_artifacts(tmp_path):
    from nucleisky2d.io import load_nucleisky_transform
    from nucleisky2d.pipeline import NucleiSky
    from nucleisky3d.pipeline import NucleiSky3D

    pts2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    out_bad_3d = NucleiSky3D(
            centroids_crop_um=pts2,
            centroids_full_um=pts2,
            full_shape_px_zyx=(10, 10, 10),
            crop_shape_px_zyx=(10, 10, 10),
            pixel_size_full_um_zyx=(1.0, 1.0, 1.0),
            pixel_size_crop_um_zyx=(1.0, 1.0, 1.0),
        )
    assert out_bad_3d.get("success") is False

    pts3 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=float)
    with pytest.raises(Exception):
        NucleiSky(
            centroids_crop_um=pts3,
            centroids_full_um=pts3,
            img_full=np.zeros((20, 20), dtype=np.float32),
            img_crop=np.zeros((20, 20), dtype=np.float32),
            ij_percentile_normalize=False,
            pixel_size_full_um=1.0,
            pixel_size_crop_um=1.0,
        )

    with pytest.raises(FileNotFoundError):
        load_nucleisky_transform(tmp_path / "missing_transform.json")
