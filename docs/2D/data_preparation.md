# Preparing Your Data for NucleiSky2D

[:arrow_left: Documentation index](../index.md)

> **Scope:** This page is for **2D image registration** workflows.
> Looking for the volumetric workflow? See the **3D equivalent**: [3D data preparation](../3D/data_preparation.md).

NucleiSky2D aligns a *partial view* (ROI / field-of-view / “crop”) to a *reference* image by matching **nuclei constellations**.

Because the alignment is **geometry-based**, it can work beautifully across **modalities** and **magnifications**—the main trick is making sure your data arrives in a form NucleiSky2D expects.

If you remember one thing from this page, make it this:

**Pixel size (µm/px) matters a lot.** It tells NucleiSky2D how “big” your nuclei constellation is in real physical units, so it can search for the right scale during geometric matching.

---

## What you need (inputs)

You need two 2D images:

1. **Reference image** (the target coordinate system)
2. **Partial view / ROI image** (the one you want to place into the reference)

Terminology note: the code uses `img_full` and `img_crop` as internal variable names. NucleiSky2D is not limited to literal crops—any partial field-of-view / ROI works perfectly.

---

## The Golden Rule: Pixel size (µm/px)

Pixel size is the key that converts “helpful pixels” into “real-world geometry.”

* If both images are truly 0.65 µm/px, they are comparable in physical scale.
* If one image is 0.65 µm/px but is incorrectly passed to the algorithm as 1.30 µm/px, the same nuclei constellation looks twice as large to the matcher, and matching will struggle.

A common source of confusion: when pixel-size metadata is missing, many workflows fall back to an assumed value (often 1.0 µm/px). That can easily push the true match outside the typical scale search window.

### How to find pixel size

In **Fiji/ImageJ**:

* Open the image
* Go to `Image` → `Properties`
* Look for Pixel Width/Height (µm)

You can also find it in acquisition metadata, imaging facility reports, or OME metadata (for OME-TIFF / OME-Zarr).

---

## Image formats (and what to check)

### TIFF / OME-TIFF (recommended)

Usually, the physical pixel size is stored directly in the file metadata and can be read automatically by the NucleiSky I/O helpers.

### Zarr / OME-Zarr

Often great for large data, but metadata consistency depends heavily on how the Zarr was originally written. Expect pixel size to be readable when using standard OME-NGFF specs, but be ready to provide it manually when missing.

### NPY

Handy for developer pipelines and testing. Pixel size is not embedded in standard NumPy arrays, so you must store pixel size alongside the file (e.g., in a config, JSON, or dataframe) and pass it explicitly to the pipeline.

### PNG / JPEG

Convenient for screenshots, but they typically do not preserve physical-pixel-size metadata. They can still be used if you explicitly provide pixel sizes as floats in your Python workflow.

---

## Loading images (Python): `load_image`

If you want a single, robust loader that “does the right thing” for common formats, use:

```python
from nucleisky2d.io import load_image

img_full = load_image("reference_image.ome.tif")
img_crop = load_image("partial_view.tif")

```

What `load_image` supports:

* TIFF / OME-TIFF
* NPY
* Zarr / OME-Zarr
* A fallback loader for standard image formats (via scikit-image)

A note about Zarr:

* If you point to an OME-Zarr group, `load_image` attempts to return the highest-resolution array (typically level "0").
* If it cannot safely guess which array you want, it returns the Zarr Group object so you can choose manually.
* *Tip for big images:* Zarr arrays load lazily. You can inspect `.shape` and `.dtype` before loading everything into memory.

---

## Displaying images safely

Microscopy images can be enormous. If you want a quick visual check in a Jupyter Notebook without accidentally allocating massive amounts of RAM, use the visualization helpers:

```python
import matplotlib.pyplot as plt
from nucleisky2d.visualization import imshow_safe

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
imshow_safe(ax[0], img_full, title="Reference")
imshow_safe(ax[1], img_crop, title="Partial view / ROI")
plt.tight_layout()
plt.show()

```

`imshow_safe` is explicitly designed for microscopy checks:

* It automatically downsamples for display when needed (keeping notebooks responsive).
* It applies a stable percentile normalization for contrast.
* It safely handles both grayscale `(H, W)` and RGB `(H, W, 3/4)`.

---

## 2D Requirements & Dimension Handling

NucleiSky2D strictly requires **2D arrays** for matching operations (a single `Y, X` plane). The pipeline utilizes a strict `require_2d_image` preprocessor that enforces this.

Here is exactly how inputs are handled:

* **Grayscale `(Y, X)`:** Perfect. Passed through directly as `float32`.
* **RGB/RGBA `(Y, X, 3)` or `(Y, X, 4)`:** Converted automatically to a single 2D grayscale plane. The pipeline applies standard luma weights (`R*0.2126 + G*0.7152 + B*0.0722`) and safely ignores the alpha channel if present.
* **Any non-2D stack layout (for example `(Z, Y, X)`, `(T, Y, X)`, or ndim > 3):** the pipeline blocks these and raises a `ValueError` because matching expects a single 2D plane (or RGB/RGBA image). Slice or project upstream before passing data in.

Practical choices for multi-dimensional data:

* **Z-stack:** Pick a representative slice, or explicitly compute a max intensity projection.
* **Multichannel:** Choose the channel where nuclei are clearest (often DAPI or Hoechst) for the registration step, then reuse the resulting JSON transform to warp your other channels during export.

---

## Segmentation: Built-in backends or Bring Your Own Masks (BYOM)

NucleiSky2D registers **nuclei point sets**. Therefore, you need nuclei label masks. You have two options:

### Option A: Segment nuclei with built-in methods

Use `segment_nuclei_dispatch` to run a segmentation backend and return a **2D label image**.

Supported backends include:

* `threshold` (fast, scikit-image based)
* `cellpose` (deep learning)
* `instanseg` (deep learning)

See details and settings here:

* [Segmentation](../segmentation.md)

### Option B: Bring Your Own Masks (label images)

If you already have nuclei masks from your favorite tool (e.g., QuPath, ImageJ, StarDist), you can load them and use them directly.

To be accepted by the pipeline's internal `require_2d_label_mask` validator, your mask must be:

* A 2D array (matching the `Y, X` dimensions of the image).
* An integer data type (`int32` is preferred; other types will trigger a safe cast and a warning).
* Strictly non-negative (no labels `< 0`).
* Non-empty (the maximum label value must be `> 0`).

Loading masks looks exactly the same as loading images:

```python
from nucleisky2d.io import load_image

labels_full = load_image("reference_labels.tif")
labels_crop = load_image("partial_labels.tif")

```

---

## Input quality: What helps matching succeed

### Nuclei count (practical guidance)

NucleiSky2D needs enough nuclei points to form a distinctive “constellation.”

* Aim for **10+ nuclei** in the partial view when possible.
* Fewer can work, but results become increasingly sensitive to segmentation noise or biological symmetry.

### Overlap

The partial view must genuinely overlap the reference image. If the two images come from entirely different physical regions (or different biological samples), the matching will fail—often in a way that looks mathematically "plausible" but biologically nonsensical.

### Segmentation quality

If nuclei are missing, merged, or dominated by debris artifacts, the geometric point sets will diverge and matching becomes exponentially harder.

If segmentation is the bottleneck for your data, consult these pages:

* [Segmentation](../segmentation.md)
* [Quality Control](../qc.md)
* [Troubleshooting](../troubleshooting.md)

---

## Sanity checks

### Inspect headers + pixel size

This is a quick, memory-efficient way to catch missing metadata before running heavy computations.

```python
from nucleisky2d.export import inspect_image_header
from nucleisky2d.io import get_pixel_size_um_from_tiff

# Fast header inspection (does not load pixel data into RAM)
print(inspect_image_header("reference_image.tif"))
print(inspect_image_header("partial_view.tif"))

# Explicit physical pixel size extraction
px_ref = get_pixel_size_um_from_tiff("reference_image.tif")
px_roi = get_pixel_size_um_from_tiff("partial_view.tif")

print("Pixel sizes (um/px):", px_ref, px_roi)

```

If a pixel size returns `None`, you must fetch the correct value from your microscope's acquisition metadata and provide it explicitly as a float to the pipeline.

---

## Next step

If your pixel sizes are known and your nuclei labels look reasonable (either from built-in segmentation or your own masks), you’re ready for:

* [Workflow](workflow.md) for an end-to-end example
* [Exports](../exports.md) and [Quality Control](../qc.md) to inspect and trust the results
