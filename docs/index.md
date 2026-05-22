# NucleiSky Documentation 🌤️

Welcome! Whether you are analyzing classic 2D slides or exploring full 3D volumes, you are in the right place.

## 🧭 Choose Your Adventure

### 🔬 I’m an image analyst

#### 🖼️ I have 2D slides
Start here for a smooth, reliable 2D pipeline:

1. **Set up your input data:** [Preparing Your Data](2D/data_preparation.md)
2. **Run end-to-end matching:** [Workflow](2D/workflow.md)
3. **Review quality:** [Quality Control](qc.md)

Need segmentation help? Jump to [Segmentation](segmentation.md) or [Troubleshooting](troubleshooting.md).

#### 🧊 I have 3D volumes
Great choice — the 3D path is designed to feel just as approachable as 2D, with a familiar workflow and clear checkpoints.

1. **Prepare volumes and metadata:** [3D Data Preparation](3D/data_preparation.md)
2. **Run the full 3D pipeline:** [3D Workflow](3D/workflow.md)
3. **Review quality:** [Quality Control](qc.md)

> ⚠️ **Experimental note:** The `nucleisky3d` module is experimental. Please manually verify results before downstream analysis.

---

### 🧑‍💻 I’m a developer

## 🧰 Developer Toolkit

Everything you need to integrate, extend, and reason about both 2D and 3D behavior:

- **Core APIs**
  - [2D API Reference](2D/api.md)
  - [3D API Reference](3D/api.md)
- **Matching internals**
  - [2D Matchers](2D/matchers.md)
  - [3D Matchers](3D/matchers.md)
- **Limits and failure modes**
  - [2D + 3D Limitations](limitations.md)

---

## 🚀 Quick starts

- **Fastest way to try 2D:** [Workflow](2D/workflow.md)
- **Fastest way to try 3D:** [3D Workflow](3D/workflow.md)
- **Prefer a GUI for 2D workflows:** [NucleiSkyApp](https://github.com/cellmigrationlab/NucleiSkyApp)

---

## 📚 Full documentation map

### Shared docs
- [Installation](installation.md)
- [Segmentation (2D + 3D)](segmentation.md)

### 2D docs
- [Preparing Your Data](2D/data_preparation.md)
- [Workflow](2D/workflow.md)
- [Quality Control](qc.md)
- [Exports](exports.md)
- [Matchers](2D/matchers.md)
- [API Reference](2D/api.md)
- [Troubleshooting](troubleshooting.md)
- [Limitations](limitations.md)

### 3D docs
- [3D Data Preparation](3D/data_preparation.md)
- [3D Workflow](3D/workflow.md)
- [Quality Control](qc.md)
- [3D Matchers](3D/matchers.md)
- [3D API Reference](3D/api.md)
- [Troubleshooting (2D + 3D)](troubleshooting.md)
- [Limitations (shared 2D + 3D)](limitations.md)
