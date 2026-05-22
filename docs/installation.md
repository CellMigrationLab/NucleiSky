# Installation

[:arrow_left: Documentation index](index.md)

## Requirements

* **Python 3.10+**
* Core scientific stack including `numpy>=2.0.2`, `scipy>=1.16.3`, `scikit-image>=0.26.0`, `pandas>=2.2.2`, `networkx>=3.1`, `tifffile>=2026.1.28`, `matplotlib>=3.10.0`, and `numba`.

## Performance and Acceleration Notes

* **GPU acceleration is optional.** NucleiSky's core matching and analysis pipeline runs on CPU.
* **GPU is only used by optional segmentation backends** (for example, Cellpose / InstanSeg via their deep-learning runtimes).
* **`numba` is used for CPU-side acceleration** in performance-critical parts of the pipeline.

## Standard Installation

For a lightweight setup (ideal if you already have segmented label images or rely on simple built-in thresholds), install the core package:

```bash
pip install nucleisky

```

## Optional Extras (By Use Case)

NucleiSky provides several optional dependency groups to tailor the installation to your specific workflow. These extras map directly to `pyproject.toml`: `segmentation`, `instanseg`, `simpleitk`, `zarr`, and `all`.

*Tip: It is highly recommended to use quotes around the package name with extras (e.g., `"nucleisky[all]"`) to prevent shell parsing errors in environments like Zsh.*

### 1. Complete Segmentation Suite

Installs both the **Cellpose** (`cellpose[all]`) and **InstanSeg** (`instanseg-torch==0.1.1`) backends for deep learning-based segmentation.

```bash
pip install "nucleisky[segmentation]"

```

### 2. InstanSeg-Only Backend

If you only need InstanSeg, this installs `instanseg-torch==0.1.1` and the necessary `torch` runtime.

```bash
pip install "nucleisky[instanseg]"

```

### 3. 3D Volumetric Superpowers

Adds **SimpleITK** (`SimpleITK`), required for specific 3D volumetric feature extraction paths.

```bash
pip install "nucleisky[simpleitk]"

```

### 4. Large-Scale Data (Zarr / OME-Zarr)

Adds `zarr` and `numcodecs` for handling chunked array I/O and OME-Zarr workflows.

```bash
pip install "nucleisky[zarr]"

```

### 5. Everything Bundle (Max Power Mode)

A convenience installation that bundles all of the optional backends above (`cellpose[all]`, `zarr`, `numcodecs`, `torch`, `SimpleITK`, and `instanseg-torch==0.1.1`).

```bash
pip install "nucleisky[all]"

```

## Developer Install

For software developers or contributors wanting to run tests and modify the source code, clone the repository and install the package in editable mode (`-e`) with all dependencies:

```bash
git clone https://github.com/cellmigrationlab/NucleiSky.git
cd NucleiSky
pip install -e ".[all]"

```

## Troubleshooting

* **GPU Compatibility:** Ensure your CUDA environment aligns with your installed version of `torch` (for InstanSeg) or your deep learning environment (for Cellpose) if you plan to use GPU-backed segmenters.
* **No GPU available?** You can still run NucleiSky fully on CPU. GPU support is optional and only impacts the deep-learning segmentation backends.
