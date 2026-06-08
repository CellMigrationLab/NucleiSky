![NucleiSky Banner](docs/assets/banner.jpg)

<div align="center">

# NucleiSky ✨🔬

**Constellation-based point-set registration for microscopy in 2D and 3D** *Align partial views to whole-slide images or thick tissue volumes — scale, rotation, and origin don’t matter.*

[![PyPI](https://img.shields.io/pypi/v/nucleisky)](https://pypi.org/project/nucleisky/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/nucleisky)](https://pypi.org/project/nucleisky/)
[![Open 2D App in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cellmigrationlab/NucleiSky/blob/main/notebooks/NucleiSky2DApp.ipynb)
[![Open 3D App in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cellmigrationlab/NucleiSky/blob/main/notebooks/NucleiSky3DApp.ipynb)

</div>

---

## ✨ What is NucleiSky?

Microscopy alignment gets messy when images are rotated, scaled, or captured on completely different platforms. NucleiSky solves this by treating your nuclei like **stars in a constellation**. 

Instead of relying on pixel intensity similarity, NucleiSky matches the *geometry* of the cells. Whether your data is a 2D field of view or a thick 3D tissue stack, NucleiSky asks: *Which mathematical transform makes these two constellations overlap perfectly?*

### 👩‍🔬 For the Biologist: The "Telescope" Setup
Imagine looking through a small telescope at a random patch of the night sky. Even without seeing the whole galaxy, you can figure out exactly where you are by matching your small star pattern against a full sky map. NucleiSky does this for your tissue: it anchors your high-magnification ROIs into whole-slide scans, perfectly recovering the rotation, zoom, and shift.

### 👨‍💻 For the Developer: The Constellation Engine
NucleiSky is a robust, extensible point-set registration pipeline built for real-world microscopy noise. We provide modular feature extraction, dynamically scaled RANSAC, geometric hashing, and tetrahedral pyramid matchers, ready to be dropped into your automated spatial pipelines.

---

## ⚡ Quickstart

Choose your launchpad and get aligning in minutes:

### 🌟 Try it in the Browser (Fastest)
No installation required. Run our interactive apps directly in Google Colab:
* [**Launch 2D App**](https://colab.research.google.com/github/cellmigrationlab/NucleiSky/blob/main/notebooks/NucleiSky2DApp.ipynb)
* [**Launch 3D App**](https://colab.research.google.com/github/cellmigrationlab/NucleiSky/blob/main/notebooks/NucleiSky3DApp.ipynb)

### 🖥️ No-Code Desktop GUI
Prefer a local app? Use our standalone [desktop installer](.tools/docs/download_executable.md).

### 🐍 Python Workflows (Local Installation)
To install the full toolkit for your own scripts: `pip install "nucleisky[all]"`

**2D Pipeline (ROI → Whole Slide)**
1. [2D Data Preparation](docs/2D/data_preparation.md)
2. [2D End-to-End Workflow](docs/2D/workflow.md)
3. [Quality Control](docs/qc.md)

**3D Pipeline (Subvolume → Thick Tissue)**
1. [3D Data Preparation](docs/3D/data_preparation.md)
2. [3D End-to-End Workflow](docs/3D/workflow.md)
3. [Quality Control](docs/qc.md)

*(Need a leaner installation? See our [Installation Guide](docs/installation.md) for modular backend options).*

---

## 🔑 Key Capabilities

* **2D & 3D Registration:** Unified logic for flat slides and thick tissue volumes.
* **Scale-Invariant:** Matches a 10x overview to a 60x confocal crop effortlessly.
* **Rotation-Invariant:** Completely robust across full 0 to 360-degree rotations.
* **Modality-Agnostic:** If you can segment the nuclei (or spots/cells), NucleiSky can match them.
* **Export-Ready:** Saves transforms, warped hyperstacks, and QC overlays for downstream analysis.

---

## ⚠️ The Two Golden Rules (Make or Break)

1. **Pixel and Voxel size matter immensely.** NucleiSky measures physical distances in micrometres. If your scale metadata (µm/px) is missing or wrong, the search bounds break, and alignment fails. Always verify your metadata! See [Data Preparation](docs/2D/data_preparation.md).
2. **Segmentation quality drives match quality.** NucleiSky registers point sets. Missing, massively merged, or noisy nuclei obscure the constellation and make matching much harder. See [Segmentation](docs/segmentation.md).

---

## 📚 Documentation Index

**Start Here**
* [Main Documentation Index](docs/index.md)
* [Installation & Extras](docs/installation.md)

**Quality Control & Troubleshooting**
* [Visualization & Quality Control](docs/qc.md)
* [Exports & Output Artifacts](docs/exports.md)
* [Troubleshooting Guide](docs/troubleshooting.md)
* [Known Limitations](docs/limitations.md)

**Developer Deep Dives**
* [Segmentation (2D + 3D)](docs/segmentation.md)
* [Matchers (2D)](docs/2D/matchers.md) | [Matchers (3D)](docs/3D/matchers.md)
* [API Reference (2D)](docs/2D/api.md) | [API Reference (3D)](docs/3D/api.md)

---


## 🤝 Contributing

We welcome PRs, feature ideas, and bug reports! For the fastest triage, please use our GitHub issue forms.

If you’re using NucleiSky in a new workflow, drop an issue and tell us:
* What you’re aligning (restains, multi-round, ROI-to-reference, etc.)
* What success looks like in your pipeline
* What would make adoption easier for your lab

*Need troubleshooting help? Please include a QC overlay from your `save_dir/original/` folder when you open a bug report!*

---

## 📢 Citation

If you find this tool useful in your research, please cite:

```bibtex
@misc{nucleisky,
  title        = {NucleiSky: Constellation-based Point-Set Registration for Microscopy},
  author       = {CellMigrationLab},
  year         = {2026},
  howpublished = {\url{[https://github.com/cellmigrationlab/NucleiSky](https://github.com/cellmigrationlab/NucleiSky)}},
  note         = {Version X.Y.Z}
}
```
