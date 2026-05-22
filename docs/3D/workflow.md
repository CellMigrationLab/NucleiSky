# Galactic 3D Alignment: End-to-End Workflow

[:arrow_left: Documentation index](../index.md)

Welcome to the NucleiSky3D constellation tour.

Aligning massive 3D tissue volumes can be terrifying for your RAM, but the core idea here is elegant and highly memory-efficient: we segment the nuclei, convert those massive label volumes into lightweight 3D point clouds (centroids in physical µm), and then match the *geometry* to recover the scale, rotation, and translation.

---

## Choose your route (fast)

**Notebook / Python API (recommended)**
You are in the right place. Use this page for end-to-end 3D usage.

**Need details on axis order and voxel metadata first?**
Start with: [Data preparation](data_preparation.md)

**Need segmentation choices and caveats?**
See: [Segmentation](../segmentation.md)

**Need to understand the underlying math?**
See: [3D Matcher Guide](matchers.md)

---

## The workflow, at a glance

1. **Load volumes** (+ enforce physical voxel sizes)
2. **Normalize scale** (optional, but highly recommended for multi-scale microscopy)
3. **Segment nuclei** (or load your own label volumes)
4. **Extract features** (turn voxels into coordinates)
5. **Match constellations** (choose a specific algorithm or use Auto-Pilot)
6. **Export aligned outputs** (warp the tissue)
7. **QC** (inspect match quality and visual overlays)
8. **Reuse saved transforms** (apply the math to other channels!)

---

## Step 1 — Load volumes (+ voxel size)

You need a **reference volume** (the "sky map") and a **crop/ROI volume** (the "telescope snapshot" you are trying to place).

For 3D, voxel size in **`(Z, Y, X)`** µm/voxel is strictly required for the math to be physically meaningful.


> ⚠️ **Scientific metadata warning:** voxel size is a scientific input, not display metadata.
> Wrong `voxel_*_um_zyx` values can produce scientifically incorrect `R_zyx`, `t_um_zyx`, and `bbox_full_px_z0z1y0y1x0x1` fields even when code execution succeeds.
> All 3D spacing in NucleiSky is ordered **(Z, Y, X)**.


```python
from pathlib import Path
from nucleisky3d.io import load_volume, require_voxel_size_um_zyx

base = Path("path/to/your/data")
full_path = base / "full_volume.tif"
crop_path = base / "crop_volume.tif"

# Load volumes, forcing ZYX order
img_full = load_volume(str(full_path))
img_crop = load_volume(str(crop_path))

# Extract physical metadata. If the TIFF header is missing Z-spacing, 
# it will gracefully fall back to the provided tuple to prevent scaling errors.
voxel_full_um_zyx = require_voxel_size_um_zyx(str(full_path), fallback=(1.0, 0.5, 0.5))
voxel_crop_um_zyx = require_voxel_size_um_zyx(str(crop_path), fallback=(1.0, 0.5, 0.5))

# Warning: fallback tuples are placeholders for demonstration only.
# Replace with acquisition-validated voxel metadata (Z,Y,X) before production use.

```

---

## Step 2 — (Optional) Normalize scale for segmentation

**🔬 For the Biologist:** You wouldn't use a magnifying glass to look at an elephant. If your reference and crop were taken at totally different magnifications, our AI models might struggle to recognize the cells. This step rescales both images to a common "target" size before segmentation.

**💻 For the Developer:** This step normalizes the arrays to an effective common voxel spacing (by default, the `coarsest` available to save memory). Watch your RAM here—upsampling thick Z-stacks can cause memory explosions if `max_upsample` bounds aren't respected!

```python
from nucleisky3d.preprocess import scale_normalize_pair_for_segmentation

(
    img_full_seg,
    img_crop_seg,
    voxel_full_seg_um_zyx,
    voxel_crop_seg_um_zyx,
    scale_full_to_orig_zyx,
    scale_crop_to_orig_zyx,
    target_um_zyx,
) = scale_normalize_pair_for_segmentation(
    img_full,
    img_crop,
    voxel_size_full_um_zyx=voxel_full_um_zyx,
    voxel_size_crop_um_zyx=voxel_crop_um_zyx,
)

```

*(If you skip this step, just set `img_full_seg = img_full` and `voxel_full_seg_um_zyx = voxel_full_um_zyx`, etc.)*

---

## Step 3 — Segment nuclei (or bring your own labels)

You have two valid options to find the "stars":

### Option A: Run built-in 2.5D segmentation

**🔬 For the Biologist:** True 3D deep learning requires massive amounts of VRAM. Instead, we use an optimized "Slice-and-Stitch" approach: we slice the stack like a loaf of bread, segment the cells in 2D, and then mathematically glue them back together across the Z-axis.

```python
from nucleisky3d.segmentation import segment_nuclei_2p5d

seg_method = "threshold" # Or "cellpose", "instanseg"
seg_settings = {
    "threshold": {
        "threshold_method": "otsu",
        "min_object_size": 10,
        "do_watershed": True,
    }
}

labels_full = segment_nuclei_2p5d(
    volume_zyx=img_full_seg,
    method=seg_method,
    pixel_size_um_zyx=voxel_full_seg_um_zyx,
    settings=seg_settings,
)
labels_crop = segment_nuclei_2p5d(
    volume_zyx=img_crop_seg,
    method=seg_method,
    pixel_size_um_zyx=voxel_crop_seg_um_zyx,
    settings=seg_settings,
)

```

### Option B: Bring your own 3D label volumes

If labels were generated elsewhere (Cellpose 3D, Fiji, ilastik), load them directly. Just ensure the array is strictly **ZYX** integer labels, with background as `0`.

---

## Step 4 — Extract features + centroids (µm)

Convert those heavy label masks into lightweight pandas DataFrames using SimpleITK shape statistics, then pull the matcher-ready 3D centroid coordinates.

```python
from nucleisky3d.features import extract_nuclear_features_3d, centroids_from_df_3d

# Extract morphological features (volume, sphericity)
df_full = extract_nuclear_features_3d(labels_full, pixel_size_um=voxel_full_seg_um_zyx)
df_crop = extract_nuclear_features_3d(labels_crop, pixel_size_um=voxel_crop_seg_um_zyx)

# Pull just the (Z, Y, X) coordinates in physical micrometers
centroids_full_um = centroids_from_df_3d(df_full, use_um=True)
centroids_crop_um = centroids_from_df_3d(df_crop, use_um=True)

```

---

## Step 5 — Match constellations

You can control the specific mathematical algorithm, or let NucleiSky test them automatically.

### Option A: Single Matcher (More Control)

Use this if you know exactly what algorithm you want (e.g., `"pyramid"` for small crops, `"hashing"` for massive clouds).

```python
from nucleisky3d.pipeline import NucleiSky3D

best_res = NucleiSky3D(
    centroids_crop_um=centroids_crop_um,
    centroids_full_um=centroids_full_um,
    full_shape_px_zyx=img_full.shape,
    crop_shape_px_zyx=img_crop.shape,
    pixel_size_full_um_zyx=voxel_full_um_zyx,
    pixel_size_crop_um_zyx=voxel_crop_um_zyx,
    matcher="pyramid",  
    df_full=df_full,
    df_crop=df_crop,
)

```

### Option B: Adaptive Matcher "Auto-Pilot" (Recommended)

This runs a dynamic sequence, testing different matchers based on how many nuclei are present, and keeps a detailed history of the attempts. **Note:** This function conveniently handles the metadata tracking (`transforms.jsonl`) and TIFF exporting automatically!
Canonical transform fields and legacy alias mapping are documented in [Exports → Canonical transform schema](../exports.md#5-canonical-transform-schema-2d-and-3d).

```python
from nucleisky3d.pipeline import run_adaptive_matching_and_export_3d

best_res, history = run_adaptive_matching_and_export_3d(
    df_full=df_full,
    df_crop=df_crop,
    img_full_orig=img_full,
    img_crop_orig=img_crop,
    pixel_size_full_orig_um_zyx=voxel_full_um_zyx,
    pixel_size_crop_orig_um_zyx=voxel_crop_um_zyx,
    result_dir="./nucleisky_3d_output",
    # Pass segmentation data if you used Step 2 scaling:
    labels_full=labels_full,
    labels_crop=labels_crop,
    img_full_seg=img_full_seg,
    img_crop_seg=img_crop_seg,
    pixel_size_full_seg_um_zyx=voxel_full_seg_um_zyx,
    pixel_size_crop_seg_um_zyx=voxel_crop_seg_um_zyx,
)

```

*(If you used Option B, your aligned TIFFs are already waiting for you in `./nucleisky_3d_output/matching/adaptive_3d/exports_adaptive`. You can skip to Step 7!)*

---

## Step 6 — Export aligned outputs (If using Option A)

If you ran a single matcher manually, use `export_aligned_crop_tiff` to warp the crop pixels into the full-volume space. *Developer Note:* Notice that `pixel_size_full_um` omits the `_zyx` suffix in the parameter name for this specific function, though it still expects ZYX tuple order.

```python
from nucleisky3d.export import export_aligned_crop_tiff

aligned_path = export_aligned_crop_tiff(
    img_full=img_full,
    img_crop=img_crop,
    output_path="./aligned_crop_zyx.tif",
    pixel_size_full_um=voxel_full_um_zyx,  # values remain ordered (Z, Y, X)
    pixel_size_crop_um=voxel_crop_um_zyx,  # values remain ordered (Z, Y, X)
    res=best_res,
    export_region="bbox",  # Use "bbox" for ROI only, "full" for whole grid
)

```

---

## Step 7 — QC: match quality + overlays

For a full QC checklist and interpretation guide, see [Quality Control](../qc.md).

Did it work? Never trust an algorithm blindly.
At a minimum, check the console output for `best_res["success"]` and `best_res["match_quality"]`.

To generate visual Maximum Intensity Projection (MIP) overlays and slice checks for quick inspection:

```python
from nucleisky3d.visualization import plot_warp_overlay3D

fig = plot_warp_overlay3D(
    img_full_zyx=img_full,
    img_crop_zyx=img_crop,
    record_or_result=best_res,
    pixel_size_full_um_zyx=voxel_full_um_zyx,
    pixel_size_crop_um_zyx=voxel_crop_um_zyx,
    save_path="./qc_overlay_3d.png",
    show=False,
)

```

---

## Step 8 — Reuse saved transforms (The ultimate payoff)

**🔬 For the Biologist:** You matched your tissues using the DAPI channel. Now what? You can load the saved `.json` transform file and apply that exact same mathematical warp to your GFP channel, your RFP channel, or your spatial transcriptomics spots, perfectly aligning the whole multiplexed experiment.

**💻 For the Developer:** The `save_nucleisky_transform_3d` function uses an internal JSON sanitizer to ensure that NumPy arrays are safely serialized. You can load this file into Anndata or SpatialData ecosystems without ever needing to touch the original TIFFs again.

```python
from nucleisky3d.io import save_nucleisky_transform_3d, load_nucleisky_transform_3d
from nucleisky3d.export import warp_crop_to_full_volume

# 1. Save the math
record = save_nucleisky_transform_3d(
    res=best_res,
    out_path="./best_transform_3d.json",
    pixel_size_full_um_zyx=voxel_full_um_zyx,
    pixel_size_crop_um_zyx=voxel_crop_um_zyx,
    matcher_name="adaptive_choice",
)

# 2. Later, load the math...
loaded_transform = load_nucleisky_transform_3d("./best_transform_3d.json")

# 3. ...and apply it to a completely different channel!
gfp_crop = load_volume("gfp_crop_volume.tif")

aligned_gfp_array = warp_crop_to_full_volume(
    img_crop=gfp_crop,
    full_shape_zyx=img_full.shape,
    pixel_size_full_um=voxel_full_um_zyx,  # values remain ordered (Z, Y, X)
    pixel_size_crop_um=voxel_crop_um_zyx,  # values remain ordered (Z, Y, X)
    res=loaded_transform
)

```
