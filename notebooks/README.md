# Notebooks

These notebooks are retained as runnable examples and manuscript-support workflows. Prefer the package API documented under `docs/` for reusable scripts, and treat benchmark notebooks as analysis workflows that may require substantial runtime and manual parameter review.

## Notebook groups

| Folder | Notebook | Intended use | Notes |
| --- | --- | --- | --- |
| `NucleiSky2DApp` | `NucleiSky2DApp.ipynb` | Interactive 2D no-code app | Uses widgets and optional segmentation backends. |
| `NucleiSky3DApp` | `NucleiSky3DApp.ipynb` | Interactive 3D no-code app | 3D support should be manually QC-reviewed before downstream analysis. |
| `NucleiSky2D_API_Workflow_Example` | `NucleiSky2D_API_Workflow_Example.ipynb` | Script-like 2D API example | Good starting point for external users converting a workflow to Python. |
| `NucleiSky3D_API_Workflow_Example` | `NucleiSky3D_API_Workflow_Example.ipynb` | Script-like 3D API example | Demonstrates synthetic fallback data and 3D exports. |
| `NucleiSky2DBenchmarking` | `NucleiSky2DBenchmarking.ipynb` | Manuscript/benchmark analysis | Expects calibrated 2D image inputs and can checkpoint/reload result CSVs. |
| `NucleiSky3DBenchmarking` | `NucleiSky3DBenchmarking.ipynb` | Manuscript/benchmark analysis | Reference-only 3D subvolume-localisation benchmark; not a full arbitrary-rotation validation. |

## Environment

For exact notebook dependencies, use each folder's `requirements.yaml` or the merged repository-level `requirements.txt`. For package installs, `pip install "nucleisky[notebooks]"` installs the common interactive/plotting dependencies, while `pip install "nucleisky[all]"` also installs optional segmentation and volumetric I/O backends.

## Inputs and paths

Replace placeholder paths such as `path/to/your/data` with your local files or mounted cloud paths. Avoid committing private absolute paths in executed notebook cells.

## Benchmark reproducibility notes

Benchmark notebooks should record:

* input image/volume path and calibration (`pixel_size_um` for 2D, `voxel_size_um_zyx` for 3D);
* segmentation method/settings or precomputed label/centroid tables;
* matcher configuration and random seeds;
* checkpoint/result CSV locations;
* success definition used by the plot (matcher-reported inlier success versus image-level validation such as SSIM);
* output plots and any reload cells used to regenerate figures.

See `docs/DOCUMENTATION_AUDIT.md` for unresolved notebook-publication checks identified during the documentation audit.
