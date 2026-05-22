# End-to-End 2D Workflow & Quick Start

[:arrow_left: Documentation index](../index.md)

> **Scope:** This page is for **2D image registration** workflows.
> Looking for the volumetric workflow? See the **3D equivalent**: [3D workflow](../3D/workflow.md).

Welcome to the NucleiSky2D “constellation tour”.

The big idea is simple: segment nuclei, convert them to point clouds, then match those constellations to recover scale, rotation, and translation. Because the matching is geometry-based, it’s a great fit for cross-modality and cross-magnification alignment.

Tip: This guide mirrors our interactive demo notebook:
[NucleiSky2D_API_Workflow_Example.ipynb](../../notebooks/NucleiSky2D_API_Workflow_Example.ipynb)

Demo note: You can synthesize ROI crops by sampling random patches from the full reference image using `nucleisky2d.demo_utils.generate_random_crop`. This is useful when you want to test the matching pipeline without collecting a separate ROI image.

Minimal patch demo:

```python
import numpy as np
from nucleisky2d.demo_utils import generate_random_crop

rng = np.random.default_rng(0)

crop, crop_pixel_size_um, (cy, cx, angle_deg, zoom_factor) = generate_random_crop(
    img_full,
    patch_h=512,
    patch_w=512,
    zoom_range=(0.6, 1.2),
    max_angle_deg=15,
    pixel_size_um=px_full,
    rng=rng,
)

```

---

## Choose your route (fast)

GUI (no coding)

NucleiSkyApp → [https://github.com/cellmigrationlab/NucleiSkyApp](https://github.com/cellmigrationlab/NucleiSkyApp)

Try in the browser (quickest)

Colab demo → [https://colab.research.google.com/github/cellmigrationlab/NucleiSky/blob/main/notebooks/NucleiSky2DApp.ipynb](https://colab.research.google.com/github/cellmigrationlab/NucleiSky/blob/main/notebooks/NucleiSky2DApp.ipynb)

Python pipeline (batch / reproducible)

You are in the right place. Keep reading.

Developer/contributor

Start with: [API Reference](api.md) and [Matchers](matchers.md)

---

## The workflow, at a glance

1. Load images (+ pixel size)
2. (Optional) Normalise scale for segmentation
3. Segment nuclei (or load your own label masks)
4. Extract nuclei features + centroids
5. Match constellations (single matcher or adaptive)
6. Export aligned outputs
7. QC: overlays + error maps
8. Reuse the saved transform on multi-dimensional data (and coordinates)

Each step below points you to deeper docs when you want more control.

---

## Step 1 — Load images (+ pixel size)

You need:

* A reference image (the “sky map”)
* A partial view / ROI image (the “telescope snapshot”)

Pixel size (µm/px) is essential for cross-scale matching. For formats + pixel size guidance, see:
[Preparing Your Data](data_preparation.md)

Minimal example:

```python
from pathlib import Path
from nucleisky2d.io import load_image, get_pixel_size_um_from_tiff, make_result_dir

base = Path("path/to/your/data")
full_path = base / "reference_image.tif"
roi_path  = base / "partial_view.tif"

out_dir = make_result_dir(root_dir="nucleisky_output")

img_full = load_image(full_path)
img_roi  = load_image(roi_path)

px_full = get_pixel_size_um_from_tiff(full_path) or 0.65
px_roi  = get_pixel_size_um_from_tiff(roi_path)  or 0.65

# Warning: 0.65 is a placeholder fallback used only for this example.
# Replace with acquisition-validated µm/px metadata before production use.

```

A note about dimensions:

* NucleiSky2D registration is driven by 2D nuclei centroids (a single YX plane).
* RGB images are accepted, but converted to a single 2D grayscale plane internally.
* If your file is a stack (Z/T), choose a plane or projection upstream before running matching.

If you want help deciding what to feed in (projection, channel choice, etc.), jump to:
[Preparing Your Data](data_preparation.md)

---

## Step 2 — (Optional) Normalise scale for segmentation

If your reference and ROI have very different pixel sizes, it often helps to segment at a common “segmentation scale” and map the centroids back to the original pixel space.

Helper:

* `scale_normalize_pair_for_segmentation` (see: [Segmentation](../segmentation.md))

Example:

```python
from nucleisky2d.preprocess import scale_normalize_pair_for_segmentation

(
    img_full_seg,
    img_roi_seg,
    px_full_seg,
    px_roi_seg,
    scale_full_to_orig,
    scale_roi_to_orig,
    target_um,
) = scale_normalize_pair_for_segmentation(img_full, img_roi, px_full, px_roi, strategy="coarsest")

```

If you already know your segmentation will be fine “as is”, you can skip this step and segment `img_full` / `img_roi` directly.

---

## Step 3 — Segment nuclei (or bring your own label masks)

You have two valid options:

**Option A: Use NucleiSky2D segmentation dispatch** - `segment_nuclei_dispatch`

* Backends include thresholding and model-based options (details here: [Segmentation](../segmentation.md))

Example:

```python
from nucleisky2d.segmentation import segment_nuclei_dispatch

seg_method = "threshold"  # see segmentation.md for available methods
seg_settings = {
    "threshold": {
        "threshold_method": "otsu",
        "min_object_size": 80,
        "do_watershed": True,
    }
}

labels_full = segment_nuclei_dispatch(img_full_seg, seg_method, px_full_seg, seg_settings)
labels_roi  = segment_nuclei_dispatch(img_roi_seg,  seg_method, px_roi_seg,  seg_settings)

```

Cellpose example (model-based segmentation):

```python
from nucleisky2d.segmentation import segment_nuclei_dispatch

seg_method = "cellpose"
seg_settings = {
    "cellpose": {
        "pretrained_model": "cpsam",
        "diameter": None,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "min_size": 15,
    }
}

labels_full = segment_nuclei_dispatch(img_full_seg, seg_method, px_full_seg, seg_settings)
labels_roi  = segment_nuclei_dispatch(img_roi_seg,  seg_method, px_roi_seg,  seg_settings)

```

**Option B: Bring your own segmentation label images** If you already have masks (from Fiji, Cellpose, Ilastik, Stardist, etc.), load them as 2D integer label images:

* background = 0
* each nucleus has an integer id: 1..N

Then proceed directly to Step 4.

---

## Step 4 — Extract nuclei features + centroids

This step turns label masks into tables (DataFrames) and centroids.

Key functions:

* `extract_nuclear_features`
* `add_centroids_orig_px_columns` (if you used Step 2)
* `extract_centroids_um`

More detail:

* [Segmentation](../segmentation.md)
* [Matchers](matchers.md) (explains what the matchers expect)

Example:

```python
from nucleisky2d.features import (
    extract_nuclear_features,
    add_centroids_orig_px_columns,
    extract_centroids_um,
)

df_full = extract_nuclear_features(labels_full, None, px_full_seg)
df_roi  = extract_nuclear_features(labels_roi,  None, px_roi_seg)

# If you used Step 2, map centroids back to original pixel coordinates
df_full = add_centroids_orig_px_columns(df_full, scale_full_to_orig)
df_roi  = add_centroids_orig_px_columns(df_roi,  scale_roi_to_orig)

centroids_full_um = extract_centroids_um(df_full, name="df_full")
centroids_roi_um  = extract_centroids_um(df_roi, name="df_roi")

```

---

## Step 5 — Match constellations

You can run a single matcher or let NucleiSky2D choose adaptively.

Matcher details and tradeoffs:

* [Matchers](matchers.md)

### Option A: Single matcher (more control)

Use:

* `NucleiSky(...)`

Example:

```python
from nucleisky2d.pipeline import NucleiSky
from nucleisky2d.preprocess import ij_percentile_normalize

res = NucleiSky(
    centroids_crop_um=centroids_roi_um,
    centroids_full_um=centroids_full_um,
    img_full=img_full,
    img_crop=img_roi,
    ij_percentile_normalize=ij_percentile_normalize,
    pixel_size_full_um=px_full,
    pixel_size_crop_um=px_roi,
    matcher="quad",   # "graph", "quad", "triangles", "hashing"
    df_full=df_full,
    df_crop=df_roi,
    labels_full=labels_full,
    labels_crop=labels_roi,
)

```

### Option B: Adaptive matching (recommended for most users)

Use:

* `run_adaptive_matching_and_export`

This tries a sensible matcher order, keeps track of attempts, and writes exports and transforms for you.

Example:

```python
from nucleisky2d.pipeline import run_adaptive_matching_and_export
from nucleisky2d.preprocess import ij_percentile_normalize

best_res, history = run_adaptive_matching_and_export(
    df_full=df_full,
    df_crop=df_roi,
    img_full=img_full,
    img_crop=img_roi,
    pixel_size_full_um=px_full,
    pixel_size_crop_um=px_roi,
    result_dir=str(out_dir),
    store_full_out=True,
    labels_full=labels_full,
    labels_crop=labels_roi,
    ij_percentile_normalize=ij_percentile_normalize, # Required if you want automatic QC overlays saved
)

```

If matching fails, the two most common levers are:

* pixel size correctness (see: [Preparing Your Data](data_preparation.md))
* segmentation quality (see: [Segmentation](../segmentation.md) and [Troubleshooting](../troubleshooting.md))

---

## Step 6 — Export aligned outputs

If you used adaptive mode, exports are already written under your `result_dir` (e.g., `matching/adaptive/exports_adaptive/`).

If you ran a single matcher, export manually with:

* `export_aligned_dataset`

Export formats and folder structure:

* [Exports](../exports.md)

Example:

```python
from nucleisky2d.export import export_aligned_dataset

export_aligned_dataset(
    res,
    out_dir=out_dir / "aligned",
    img_full=img_full,
    img_crop=img_roi,
    pixel_size_full_um=px_full,
    pixel_size_crop_um=px_roi,
    export_region="roi",  # "roi" or "full"
    axes_full="YX",
    axes_crop="YX",
)

```

---

## Step 7 — QC (highly recommended)

Constellation matching can produce something that is “mathematically valid” but biologically wrong if segmentation is poor or the ROI doesn’t truly overlap.

For quick visual validation, use:

* `show_alignment_original_and_rescaled`

See the full QC guide here:

* [Quality Control](../qc.md)

Example:

```python
from nucleisky2d.visualization import show_alignment_original_and_rescaled

show_alignment_original_and_rescaled(
    res,
    img_full_orig=img_full,
    img_crop_orig=img_roi,
    pixel_size_full_orig_um=px_full,
    pixel_size_crop_orig_um=px_roi,
    save_dir=out_dir / "qc",
)

```

---

## Step 8 — Reuse a successful alignment on multi-dimensional data (and coordinates)

This is the “make it pay off” step.

A good match produces a transform JSON (e.g., `adaptive_best_transform_original.json`). That JSON is meant to be reused: apply the same registration to other channels, other derived images, and even non-image coordinates.

Details on what’s inside the JSON:

* [Exports](../exports.md)

### 8A) Load the transform JSON

```python
from nucleisky2d.io import load_nucleisky_transform

tr = load_nucleisky_transform(out_dir / "matching/adaptive/exports_adaptive/adaptive_best_transform_original.json")

# Convenient: reuse the stored pixel sizes
px_full = float(tr["pixel_size_full_um"])
px_roi  = float(tr["pixel_size_crop_um"])

# Canonical transform schema reference:
# docs/exports.md → "Canonical transform schema (2D and 3D)"

```

### 8B) Apply it to multi-channel / Z / time stacks during export (recommended)

If you computed the transform on DAPI, you can apply it to a multi-channel crop stack (e.g., DAPI+GFP+RFP) by exporting with axes that describe your array layout.

The exporter understands common axis conventions (`YX`, `CYX`, `ZYX`, `ZCYX`, `TYX`, `TCYX`, `TZYX`, `TZCYX`). See:

* [Exports](../exports.md)
* [API Reference](api.md)

Example: apply the same transform to multi-channel stacks

```python
from nucleisky2d.io import load_image
from nucleisky2d.export import export_aligned_dataset

img_full_mc = load_image("reference_multichannel.tif")
img_roi_mc  = load_image("roi_multichannel.tif")

export_aligned_dataset(
    tr,                              # saved transform record works here
    out_dir=out_dir / "aligned_mc",
    img_full=img_full_mc,
    img_crop=img_roi_mc,
    pixel_size_full_um=px_full,
    pixel_size_crop_um=px_roi,
    axes_full="CYX",                 # change to match your data (e.g., "ZCYX", "TCYX", "TZCYX")
    axes_crop="CYX",
    export_region="roi",
)

```

Practical note:

* Registration is estimated from 2D nuclei centroids, but the same affine can be applied across Z/T/C by warping each plane consistently.

### 8C) Warp in memory (when you want an array, not files)

If you want to warp a dataset in memory (e.g., to feed into your own pipeline), use:

```python
from nucleisky2d.export import warp_dataset_with_transform

warped_roi_mc = warp_dataset_with_transform(
    moving=img_roi_mc,
    A_px=tr["A_px"],
    b_px=tr["b_px"],
    out_shape_yx=img_full_mc.shape[-2:],   # (Y, X) of the reference grid
    axes="CYX",                             # same axes you used above
)

```

### 8D) Apply the transform to coordinates (spots, ROIs, annotations)

The same affine mapping (`A_px` and `b_px`) used internally for images can be applied to 2D coordinate sets. This is useful for aligning “invisible layers” (spatial transcriptomics spots, manual ROI points, etc.).

To manually map `(y, x)` crop points to the full reference frame using numpy: `pts_full = pts_crop @ np.array(tr["A_px"]).T + np.array(tr["b_px"])`.

---

## Advanced (short, optional): tuning matcher settings

If you need to adjust scale bounds, inlier radius, or matcher-specific parameters, prefer `matcher_config` (structured, reproducible) over ad-hoc tweaks.

Deep dive:

* [Matchers](matchers.md)

Minimal example:

```python
my_config = {
    "_common": {
        "scale_max": 4.0,
    },
    "graph": {
        "n_iters": 100_000,
    },
}

res = NucleiSky(
    centroids_crop_um=centroids_roi_um,
    centroids_full_um=centroids_full_um,
    img_full=img_full,
    img_crop=img_roi,
    ij_percentile_normalize=ij_percentile_normalize,
    pixel_size_full_um=px_full,
    pixel_size_crop_um=px_roi,
    matcher="graph",
    matcher_config=my_config,
    df_full=df_full,
    df_crop=df_roi,
)

```

---

## Where to go next

* If you’re choosing or tuning segmentation: [Segmentation](../segmentation.md)
* If you’re choosing matchers or tuning parameters: [Matchers](matchers.md)
* If you’re checking overlays/error maps: [Quality Control](../qc.md)
* If exports (including transform reuse) are your main goal: [Exports](../exports.md)
* If something looks “almost right” but not quite: [Troubleshooting](../troubleshooting.md)
