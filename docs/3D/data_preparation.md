# Preparing Your Data for NucleiSky3D

[:arrow_left: Documentation index](../index.md)

NucleiSky3D aligns nuclei constellations in **3D**, so input geometry has to be explicit and consistent.

If you remember one thing from this page, make it this:

**Axis order and voxel size must match reality.**

In 3D, a swapped axis or wrong Z spacing doesn't just change the picture—it distorts the physical distance between stars in the constellation, causing geometric matching to fail entirely.

---

## What you need (inputs)

You need two 3D image volumes:

1. **Reference/full volume** (the target coordinate system)
2. **Partial/crop volume** (the one you want to place in reference space)

**Do I need labels?**
No! You can provide **raw intensity images** (like a 3D DAPI stack or brightfield scan). NucleiSky3D has built-in 2.5D slice-and-stitch segmentation to find the nuclei for you.

*Optional (Bring Your Own Masks):* If you already have segmentation masks from another tool, you can use those directly. The pipeline's internal `require_3d_label_mask` validator strictly requires that they are 3D integer label arrays, safely casts them to `int32`, ensures background is `0` (no negative labels allowed), confirms the array is not empty (max label > 0), and verifies the shape is strictly `(Z, Y, X)`.

---

## The golden rule in 3D: strict `ZYX` ordering

NucleiSky3D rigorously assumes volumes are ordered `(z, y, x)`.

This is enforced when using the `load_volume(...)` helper via the internal `_ensure_zyx(...)` function:

* If the array is 3D, it is accepted **as-is**.
* If the array is 4D (e.g., it contains a channel axis), NucleiSky3D attempts to drop the channel axis.
* If you don't specify the `channel_axis`, it will try to guess it by looking for a dimension size  at the start (axis 0) or end (axis -1).
* By default, it keeps **Channel 0** (`channel_index=0`).
* *Developer Note:* Channel reduction is performed using slicing rather than `np.take()` specifically to preserve lazy-loading behavior for out-of-core formats like OME-Zarr and Dask.
* If the 4D shape is ambiguous and no channel axis can be inferred, it raises a `ValueError`.

Why this matters:

* 3D transforms (`R_zyx`, `t_um_zyx`) are defined in ZYX coordinates.
* Feature extraction and geometry use physical units along those same axes.
* If your data is actually XYZ or YXZ but treated as ZYX, the tissue is effectively rotated in the math, and nuclei will appear in the wrong places.

Practical checks:

* Confirm that axis 0 really indexes optical slices/depth (Z).
* If loading multi-channel data where your nuclear stain is not the first channel, explicitly provide `channel_axis` and `channel_index`.

---

## Defining voxel size correctly: use `(z, y, x)` for anisotropic data

In 2D, a single pixel size is often enough.
In 3D microscopy, Z spacing is frequently much coarser than XY voxel spacing.

For that reason, NucleiSky3D supports voxel size as:

* **Isotropic:** single float (e.g., `1.0` means 1.0 µm in all directions)
* **Anisotropic:** tuple `(z, y, x)` in micrometers

**Always use tuples** whenever your volume is anisotropic (common in confocal/light-sheet stacks):

```python
voxel_size_full_um_zyx = (2.0, 0.325, 0.325)
voxel_size_crop_um_zyx = (2.0, 0.325, 0.325)

```

This keeps distances and alignment scoring physically meaningful. If your Z-step is 2.0 µm but you default to 1.0 µm, your spherical nuclei will be mathematically squashed into pancakes!

---

## Reading voxel size safely from metadata

The safest way to load metadata is using `require_voxel_size_um_zyx(...)`.

This helper prioritizes OME-XML physical sizes, then TIFF resolution tags for XY, then ImageJ spacing for Z. Crucially, it strictly enforces a 3D physical scale and allows you to provide a fallback tuple if the metadata is missing (which is very common in standard TIFFs).

*Developer Warning:* `require_voxel_size_um_zyx` no longer auto-fabricates missing Z-spacing from in-plane XY spacing (even if `allow_missing_z=True` is passed). If Z-spacing is missing from the header, you **must** provide it via the `fallback` argument to prevent physically invalid alignments.

```python
from nucleisky3d.io import require_voxel_size_um_zyx

# Returns a (z, y, x) float tuple, or raises an error if metadata is missing
voxel_zyx = require_voxel_size_um_zyx("full_stack.ome.tif", fallback=(2.0, 0.5, 0.5))
print(voxel_zyx)  # e.g., (2.0, 0.5, 0.5)

```

---

## Sanity Check: Inspect before loading

3D microscopy volumes are huge. You can use `inspect_volume_header` to check dimensions, axes, and metadata without loading the actual voxel data into memory. This works for TIFF, NPY, and OME-Zarr formats.

```python
from nucleisky3d.io import inspect_volume_header

info = inspect_volume_header("massive_image.ome.tif")
print(f"Shape: {info['shape']}")     # Should be (Z, Y, X)
print(f"Axes:  {info['axes']}")      # e.g., "ZYX" or "CZYX"
print(f"Voxel: {info['voxel_size_um_zyx']}")

```

---

## Common failure modes (and fixes)

* **Looks mirrored/rotated or fails to match:**
Verify true axis order is ZYX and not XYZ/TZYX/etc.
* **Alignment scale is clearly wrong:**
Check your Z-spacing. Avoid using a single scalar if the data is anisotropic.
* **4D input error from `_ensure_zyx`:**
Provide `channel_axis` and `channel_index` when loading the volume so the correct channel (the nuclear stain) is selected.

---

## Minimal loading pattern

This is the recommended pattern to start your pipeline. It ensures your data is in the correct format and fails fast if the physical metadata is bad, rather than producing silent garbage results downstream.

```python
from nucleisky3d.io import load_volume, require_voxel_size_um_zyx

# 1. Load Data (Ensures ZYX ordering)
# If dealing with a multichannel stack where DAPI is channel 2 (index 1):
# full_vol = load_volume("full_stack.ome.tif", channel_axis=0, channel_index=1)
full_vol = load_volume("full_stack.ome.tif")
crop_vol = load_volume("crop_stack.ome.tif")

# 2. Load Metadata (Forces physical units, uses fallback if Z is missing)
voxel_size_full_um_zyx = require_voxel_size_um_zyx("full_stack.ome.tif", fallback=(2.0, 0.3, 0.3))
voxel_size_crop_um_zyx = require_voxel_size_um_zyx("crop_stack.ome.tif", fallback=(2.0, 0.3, 0.3))

```

When in doubt: enforce `ZYX`, define voxel size explicitly as `(z, y, x)`, and prefer explicit values over guessed metadata.
