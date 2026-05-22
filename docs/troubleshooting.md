# Troubleshooting (2D and 3D)

[:arrow_left: Documentation index](index.md)

This page combines troubleshooting guidance for both **NucleiSky2D** and **NucleiSky3D** workflows.

## Jump to section

* [2D troubleshooting](#2d-troubleshooting)
* [3D troubleshooting](#3d-troubleshooting)

---

## 2D Troubleshooting

When NucleiSky2D misbehaves, it almost always falls into one of four buckets:

1. **Environment / optional backends** (segmentation, Zarr, GPU).
2. **Input shape / axes** (not truly 2D; channel/time/z-stack confusion).
3. **Pixel size / physical scaling** (wrong or missing µm/px metadata).
4. **Matching quality** (too few nuclei, unrelated images, poor segmentation).

This page is designed for fast diagnosis. Each section uses **Symptom → Likely cause → Fix → Verify**.

---

## Quick Triage Checklist

Before going deep, check these three facts:

* **Are you passing a 2D array (Y, X)?** NucleiSky2D's internal `require_2d_image` strictly expects 2D images and raises a `ValueError` if it receives anything else. *(Note: RGB/RGBA inputs are automatically converted to 2D grayscale, but 3D Z-stacks will crash)*.
* **Are your pixel sizes correct and positive?** Matching and rescaling assume valid physical units; non-finite or negative values will be rejected.
* **Do you have enough nuclei in *both* images?** Matchers dynamically compute a required `min_inliers` based on your crop size, but it has an absolute hard floor of `3` points to compute an affine transform.

If those look good, go to **Registration failures (`success=False`)**.

---

## Installation & Optional Dependencies

### Symptom

* `ModuleNotFoundError` / `ImportError` for `cellpose`, `instanseg`, or `torch`.
* `ValueError: Unknown segmentation backend...`

### Likely cause

Segmentation backends are **optional**. The project defines a `[segmentation]` installation extra, and the `segment_nuclei_dispatch` function will error if you request a backend without the underlying library installed.

### Fix

Install optional segmentation extras:

```bash
pip install "nucleisky[segmentation]"

```

*(Tip: Always use quotes around bracketed extras in zsh/macOS terminals).*

---

## Zarr / OME-Zarr Loading Errors

### Symptom

`ImportError: Loading .zarr files requires the 'zarr' library.`

### Likely cause

You are trying to load a `.zarr` store without Zarr dependencies installed. The internal `_is_zarr_store_path` detector found Zarr metadata (like `.zgroup` or `.zarray`), but Python lacks the `zarr` module.

### Fix

```bash
pip install "nucleisky[zarr]"
# OR
pip install zarr ome-zarr

```

---

## Input Shape & Dimension Errors

### Symptom

`ValueError: Input '<name>' must be a 2D image (YX)...`

### Likely cause

A multi-dimensional array (Z-stack, time series) is being passed where NucleiSky2D expects a single 2D plane.

### Fix

Slice your data explicitly **before** calling NucleiSky2D.

* Z-stack → choose one Z plane (e.g., `img[10, :, :]`)
* Time-lapse → choose one timepoint (e.g., `img[0, :, :]`)

---

## Label Mask Shape / Dtype Issues

### Symptom

* `ValueError: Label mask <name> must be 2D...`
* `ValueError: Label mask <name> must not contain negative values.`
* `ValueError: Label mask <name> is empty (max value is 0).`

### Likely cause

The pipeline's strict `require_2d_label_mask` validator enforces that masks are 2D, strictly non-negative integers (0 = background), and actually contain at least one labeled object.

### Fix

* Ensure mask is `(Y, X)` and integer-labeled (e.g., `mask.astype(np.int32)`).
* If you generated a binary/boolean mask, relabel it into connected components (`skimage.measure.label(mask)`) before passing it in.

---

## Pixel Size & Metadata Pitfalls

### Symptom A: Invalid Value

`ValueError: pixel_size_*_um must be a positive, finite float. Got ...`

### Likely cause

Scaling utilities require a valid physical pixel size and will reject `None`, `0`, negative values, `NaN`, or `inf`.

### Fix

Provide correct pixel sizes manually if your TIFF headers are missing them.

---

### Symptom B: Anisotropy note in returned metadata details

`Anisotropic XY pixel size: x=... µm, y=... µm (rel=...)`

### Likely cause

When reading TIFF metadata, `get_pixel_size_um_from_tiff` detects an X/Y pixel-size mismatch exceeding the threshold and stores this message in the returned `details["note"]` when `return_details=True`. The function still falls back to the mean of X and Y, which can distort rigid geometry.

### Fix

* Correct the acquisition metadata (preferred).
* Explicitly pass a single, correct `pixel_size_um` float to override the metadata-derived value.

---

## Registration Failures (`success=False`)

### Symptom

* The pipeline runs but reports `success=False`.
* Saving transform fails with: `ValueError: Match '...' is not successful (success=False).`.
* If transform JSON/JSONL parsing or validation fails, review canonical fields and loader compatibility notes in [Exports → Canonical transform schema](exports.md#5-canonical-transform-schema-2d-and-3d).

### Likely cause

1. **No transform was found** (the algorithms could not find enough geometric consensus).
2. **Transform exists, but the quality is too low.** Success is mathematically determined by the gate: `frac_inliers >= frac_inliers_thresh` (Default is 0.6 for 2D, 0.45 for 3D).

### Fix

1. **Confirm Biological Overlap:** If the crop and full image do not physically overlap, the inlier fraction will be near zero.
2. **Verify Pixel Sizes:** Incorrect µm/px causes a scaling mismatch, rendering identical tissues invisible to the matcher.
3. **Check Segmentation:** Too few nuclei or heavy fragmentation destroys point-cloud geometry.
4. **Use Adaptive / Hashing:** If the dataset is massive, rely on the `hashing` or `adaptive` modes instead of `triangles`.

---

## Export, Memory, and Performance Issues

### Symptom A: Full-grid TIFF is missing (or you only see Zarr)

### Likely cause

To prevent devastating Out-Of-Memory (OOM) crashes, the `export_aligned_dataset` function explicitly aborts TIFF generation if the canvas exceeds **400,000,000 pixels**, warning you to use Zarr instead.

### Fix

* Use `format="zarr"` for massive whole-slide outputs.
* Or, restrict the export area using `export_region="roi"`.

---

### Symptom B: Axes errors during export

`ValueError: Output axes '...' must include exactly 'Y' and 'X' ...`

### Likely cause

The exporter validates the shape against the axes string (e.g., `YX`, `CYX`, `ZYX`). If your string contains duplicates or omits Y and X, it fails safely.

### Fix

Provide correct `axes_full` / `axes_crop` strings that exactly match the dimensionality of your input arrays.

---

## 3D Troubleshooting

When `NucleiSky3D` fails, the most common issues are:

1. **Input shape dimension/channel confusion.**
2. **3D rescaling memory blowups** during preprocessing.
3. **Slice stitching identity splits** during 2.5D segmentation.
4. **Missing Z-spacing** causing incorrect physical scaling.

---

## Quick Triage Checklist (3D)

* **Confirm ZYX shape:** Arrays must be strictly `(Z, Y, X)`. If your array is 4D, `load_volume` attempts to slice `channel_index=0` across the identified `channel_axis` to preserve a 3D lazy-load.
* **Check Voxel Sizes:** Ensure `pixel_size_um_zyx` is a tuple of 3 positive floats.
* **Is Z-spacing missing?** The helper `require_voxel_size_um_zyx` explicitly refuses to auto-fill missing Z-spacing from XY spacing (even if legacy flags are used) to prevent physical distortion. You *must* provide a fallback if the TIFF lacks Z metadata.

---

## Rescaling causes OOM (Out Of Memory)

### Symptom

* Python process is killed by the OS.
* You see `MemoryError` during `scale_normalize_pair_for_segmentation(...)`.

### Likely cause

`skimage.transform.rescale` is used to warp the full 3D array. Because volumetric data scales cubically, requesting a 2x upsample in Z, Y, and X simultaneously requires **8x** more RAM.

### Fix

1. **Use a coarser target voxel size** (larger µm/voxel).
2. **Lower `max_upsample**` (default is `4.0`, reduce it to `2.0` or `1.5`).
3. **Prefer `strategy="coarsest"**` to avoid forcing both datasets to the finest spacing.
4. **Reduce dtype footprint** by keeping `dtype_out=np.float32` (avoiding `float64`).

---

## Nuclei fragment across Z during stitching

### Symptom

* Single nuclei appear as multiple stacked IDs in neighboring slices.
* 3D objects look like disconnected fragments.

### Likely cause

The 2.5D segmenter links slices using Intersection-over-Union (IoU). If the overlap between a cell in Slice 1 and Slice 2 is lower than `min_iou`, it assigns a new ID. Coarse Z-spacing or segmentation noise reduces this overlap.

### Fix

1. **Lower `min_iou` gradually** (e.g., from `0.30` to `0.15`).
2. **Do not set `min_iou` to 0**, as this will cause distinct neighboring cells in dense tissues to merge incorrectly.
3. Stabilize per-slice segmentation quality (e.g., by adjusting contrast or `min_object_size`) so that masks remain physically contiguous across depth.
