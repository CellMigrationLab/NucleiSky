# Understanding Outputs & Exports

[:arrow_left: Documentation index](index.md)

> **Scope:** This page covers both **2D and 3D** export outputs.
> Use it as a quick guide for where files are written and how to inspect them.

NucleiSky exports are meant to be:

* easy to open (Fiji/ImageJ, napari),
* easy to QC (multi-panel overlay and error figures),
* easy to reuse (a transform JSON you can apply to images, points, and downstream pipelines).

This page explains what gets written to disk, what each file is for, and where to start depending on what you’re trying to do.

> **Export reliability note (all workflows):** matching and transform persistence can succeed even if image export later fails (for example, due to I/O/output constraints). In those cases, saved transform records remain the reproducibility source of truth.

## If you’re in a hurry

**I just want to confirm the alignment worked**

* Open `*_overlay.png` first (fast visual sanity check).
* Look at the "Error (1-SSIM)" or "|diff| error" panels within that image to see exactly where structural mismatches are located.

**I want an aligned image I can analyze**

* Navigate into the export subdirectories and open `aligned_on_full_px.tif` (common, Fiji-friendly) or `aligned_on_full_px.zarr` (best for very large images).
* Turn on “Composite” in Fiji (Channels tool) or use the layer controls in napari.

**I want to apply the same registration to other channels / a Z-stack**

* Use the transform JSON (`*_transform_*.json` or `transforms.jsonl`) as the source of truth.
* Export/warp your multichannel or Z-stack with the saved transform (see “Reusing the transform”).

---

## 1) Where exports are written

### A) Adaptive pipeline output folder (2D)

If you run the 2D adaptive pipeline, exports go into:

`<result_dir>/matching/adaptive/exports_adaptive/`

Inside you will typically see:

* `transforms.jsonl` — a line-by-line log of candidate transforms.
* `history.jsonl` — matcher attempt history and diagnostics for each adaptive trial.
* `adaptive_best_transform_original.json` — the “best” transform mapped to the original image coordinates.
* `adaptive_best_transform_segscale.json` — the transform mapped to the downsampled/segmentation-scale image.
* `matcher_config_used.json` — the effective matcher configuration used for the run (defaults deep-merged with user overrides).
* Image export subfolders (e.g., `adaptive_best_images_original_roi/`) containing the warped `aligned_on_full_px.tif` or `.zarr` files.
* `segmentation_masks/` — optional TIFF masks (if segmentation was used/produced in the run).

Example structure:

```text
results/
└── matching/
    └── adaptive/
        └── exports_adaptive/
            ├── transforms.jsonl
            ├── history.jsonl
            ├── matcher_config_used.json
            ├── adaptive_best_transform_original.json
            ├── adaptive_best_transform_segscale.json
            ├── adaptive_best_images_original_roi/
            │   └── aligned_on_full_px.tif
            ├── adaptive_best_images_original_full/
            │   └── aligned_on_full_px.zarr
            └── segmentation_masks/
                ├── labels_full.tif
                └── labels_crop.tif

```

*(Note: Visual QC images like `_overlay.png` are generated separately by calling the visualization utilities and are often saved under an `original/` subfolder based on your `save_dir` arguments.)*

### B) Adaptive pipeline output folder (3D)

If you run the 3D adaptive pipeline, exports go into:

`<result_dir>/matching/adaptive_3d/exports_adaptive/`

Inside you will typically see:

* `transforms.jsonl` — best transform record saved for downstream reuse.
* `history.jsonl` — candidate matcher attempts, scoring metrics, and execution times.
* `best_summary.json` — quick, lightweight run summary.
* `matcher_config_used.json` — the effective matcher configuration used for the run (defaults deep-merged with user overrides).
* `aligned_crop_<matcher>.tif` — aligned crop exported in full-volume space (`ZYX`).
* `segmentation_masks/labels_full.tif` and `segmentation_masks/labels_crop.tif` — optional segmentation masks.

### C) Direct export (when you call the exporter yourself)

If you call `export_aligned_dataset(out_dir=...)` directly, outputs go into whatever `out_dir` you provide.

By default, the aligned image filenames are:

* `aligned_on_full_px.tif` (TIFF), or
* `aligned_on_full_px.zarr` (OME-Zarr)

---

## 2) The aligned image stack (what you actually open)

### What is “aligned” here?

NucleiSky aligns a **moving** crop/patch onto a **fixed** reference image or volume. Exports are written so you can directly compare the reference and the warped crop.

### ROI vs full-grid export

There are two common export modes (`export_region`):

**ROI export (`"roi"`) (smaller, faster)**

* The output covers a bounding region (in full-image coordinates) that contains the mapped crop (with an optional pixel margin).
* This is usually the most convenient format for quick QC and downstream quantification around the matched region.

**Full-grid export (`"full"`) (largest, most interoperable)**

* The output canvas exactly matches the full reference image grid dimensions.
* This is the easiest format when you need to overlay other full-image annotations or run whole-image downstream tools.
* *Developer Note:* To prevent Memory/OOM errors on massive images (e.g., area > 400,000,000 pixels), the pipeline automatically abandons TIFF generation in favor of chunked OME-Zarr generation.

### Axes and ordering (why things look “stacked”)

**TIFF exports are written as an ImageJ hyperstack strictly following `TZCYX` dimension ordering**. In practice:

* If your data is 2D single-channel, you’ll typically see an output mapped as `T=1, Z=1, C=2, Y, X`.

**Channel convention (very important):**

* Arrays are concatenated along the Channel axis (`axis=2`).
* First come the reference (fixed) channels,
* then the warped crop (moving) channels.

So if the reference has `Cf` channels and the crop has `Cc` channels, the export will have `Cf + Cc` channels total.

### How to open aligned exports

**Open TIFF in Fiji/ImageJ**

1. File → Open… and select `aligned_on_full_px.tif`
2. Open the Channels tool (Image → Color → Channels Tool…)
3. Switch to “Composite” and toggle channels on/off to compare fixed vs moving content

Fiji: [https://fiji.sc/](https://fiji.sc/)

**Open OME-Zarr in napari**

* File → Open… and select `aligned_on_full_px.zarr`
* Use the layer controls to inspect channels, and zoom smoothly across scales

napari: [https://napari.org/](https://napari.org/)

OME-NGFF / OME-Zarr overview: [https://ngff.openmicroscopy.org/latest/](https://ngff.openmicroscopy.org/latest/)

---

## 3) Transform JSON (the source of truth)

Every alignment can be represented as an affine-style transform that maps **crop coordinates into full-image/volume coordinates** (2D or 3D).

The transform JSON is the file you keep if you want to:

* reproduce the alignment later,
* apply the same registration to other channels,
* warp non-image data (spots, ROIs, point sets) into the full-image coordinate system.

### What’s inside the JSON?

Transform records contain **physical-space similarity fields** and **pixel/voxel-space affine replay fields**. Physical fields are the scientific source-of-truth; affine fields are convenience fields for replay/export in array coordinates.

#### Canonical 2D transform record fields

| Field | Shape / Type | Units | Coordinate order | Canonical export presence | Replay role |
|---|---|---|---|---|---|
| `scale` | scalar float | unitless | n/a | required | global similarity scale |
| `R_yx` | `2x2` float matrix | unitless | `(y, x)` | required | physical-space rotation |
| `t_um_yx` | length-2 float | µm | `(y, x)` | required | physical-space translation |
| `pixel_size_full_um` | scalar float | µm/px | n/a | required | links physical↔pixel spaces |
| `pixel_size_crop_um` | scalar float | µm/px | n/a | required | links physical↔pixel spaces |
| `A_px` | `2x2` float matrix | px/px | `(y, x)` | required | affine replay matrix |
| `b_px` | length-2 float | px | `(y, x)` | required | affine replay offset |
| `bbox_full_px_y0y1x0x1` | length-4 int array or `null` | px | `[y0,y1,x0,x1]` | key always present; value may be `null` | mapped crop ROI in full frame |
| `match_quality` | dict | mixed | n/a | recommended | contains `frac_inliers`, `mean_error_um` |
| `success` | bool | n/a | n/a | recommended | pipeline success gate |

#### Canonical 3D transform record fields

| Field | Shape / Type | Units | Coordinate order | Canonical export presence | Replay role |
|---|---|---|---|---|---|
| `scale` | scalar float | unitless | n/a | required | global similarity scale |
| `R_zyx` | `3x3` float matrix | unitless | `(z, y, x)` | required | physical-space rotation |
| `t_um_zyx` | length-3 float | µm | `(z, y, x)` | required | physical-space translation |
| `pixel_size_full_um_zyx` | length-3 float | µm/voxel | `(z, y, x)` | required | links physical↔voxel spaces |
| `pixel_size_crop_um_zyx` | length-3 float | µm/voxel | `(z, y, x)` | required | links physical↔voxel spaces |
| `A_px` | `3x3` float matrix | voxel/voxel | `(z, y, x)` | required | affine replay matrix |
| `b_px` | length-3 float | voxels | `(z, y, x)` | required | affine replay offset |
| `bbox_full_px_z0z1y0y1x0x1` | length-6 int array or `null` | voxels | `[z0,z1,y0,y1,x0,x1]` | key always present; value may be `null` | mapped crop ROI in full volume |
| `match_quality` | dict | mixed | n/a | recommended | contains `frac_inliers`, `mean_error_um` |
| `success` | bool | n/a | n/a | recommended | pipeline success gate |

New exports should be treated as **canonical-schema records** using the fields listed in the 2D/3D tables above.

Validation/compatibility note:

* **2D strict validation** accepts either similarity fields (`scale`, `R_yx`, `t_um_yx`) or affine replay fields (`A_px`, `b_px`).
* **Strict validation and bbox fields:** bbox keys are optional in strict loaders; if provided, they may be `null` or a finite ordered bbox vector.
* **2D loaders do not auto-normalize legacy `best_*` aliases**; prefer canonical keys in saved records and analysis code.
* **3D `load_nucleisky_transform_3d`** normalizes supported legacy similarity aliases (`best_scale`, `best_R`, `best_t`, `best_bbox`) before validation.
* **3D `load_transforms_any_3d`** additionally normalizes supported voxel-size aliases (for example `voxel_size_*_um_zyx`) before strict checks.

### Reusing the transform (images and points)

**Apply to points (conceptually)**
The affine mapping used internally to translate crop bounding boxes and points to the full reference frame is:

x_full = A_px · x_crop + b_px

If you have crop-space coordinates (e.g., detected spots in the crop), applying the transform moves them into the full-image coordinate system.

Example (plain Python-style pseudocode; adapt to your own scripts):

```python
import json
import numpy as np

with open("adaptive_best_transform_original.json", "r") as f:
    payload = json.load(f)

A = np.array(payload["A_px"], dtype=float)   # shape (2, 2)
b = np.array(payload["b_px"], dtype=float)   # shape (2,)

# points in (y, x) crop coordinates, shape (N, 2)
pts_crop = np.array([[10, 20], [30.5, 40.2]], dtype=float)

# map into full-image coordinates using transpose matrix multiplication
pts_full = pts_crop @ A.T + b

```

**Apply to other channels / a Z-stack**
When applying transforms to arrays, you must explicitly declare the data axes. The pipeline rigorously validates the `axes` string, which must consist of un-duplicated letters from the `TZCYX` set and must always include `Y` and `X`.

Examples of valid axes inputs:

* `"YX"` (2D single-channel)
* `"CYX"` (2D multi-channel)
* `"ZCYX"` (Z-stack with channels)
* `"TZCYX"` (Fully defined hyperstack)

If axes are wrong, exports will fail validation.

---

## 4) Transform history (`transforms.jsonl`)

When using the adaptive pipeline, you’ll often see:

`transforms.jsonl`

This is a JSON Lines file (one JSON object per line) containing transform records. Records include a `run_id` so multiple runs in the same folder remain traceable. It is useful when:

* you want to inspect why the “best” match was chosen,
* you want to compare metrics across candidates,
* you want to debug difficult samples.

How to inspect quickly:

* open it in any text editor (each line is valid JSON), or
* parse it line-by-line in your analysis code.

### JSONL validation modes (recommended for reproducibility)

- **2D single-record JSON (`load_nucleisky_transform`)** validates strictly.
- **2D JSONL (`load_transforms_any`) defaults to permissive mode**: `strict=False` preserves backward compatibility and can keep extra/future fields.
- **2D JSONL strict mode**: `load_transforms_any(path, strict=True)` validates each record and raises line-aware diagnostics (for example, malformed JSON at line N or invalid bbox ordering).
- **3D loaders** validate strictly; `load_nucleisky_transform_3d` normalizes supported `best_*` similarity aliases, and `load_transforms_any_3d` additionally normalizes supported voxel-size aliases.
- Invalid spacing, NaN/inf values, malformed `R_*`/`t_*`/`A_px`/`b_px`, or invalid bbox ordering fail validation in strict pathways.

```python
from nucleisky2d.io import load_transforms_any
records = load_transforms_any("transforms.jsonl", strict=True)  # recommended in production
```

## 5) Canonical transform schema (2D and 3D)

Use the following fields as the canonical schema for persisted transforms.

| Dimension | Canonical fields |
|---|---|
| **2D** | `scale`, `R_yx`, `t_um_yx`, `A_px`, `b_px`, `pixel_size_full_um`, `pixel_size_crop_um`, `bbox_full_px_y0y1x0x1`, `match_quality`, `success` |
| **3D** | `scale`, `R_zyx`, `t_um_zyx`, `A_px`, `b_px`, `pixel_size_full_um_zyx`, `pixel_size_crop_um_zyx`, `bbox_full_px_z0z1y0y1x0x1`, `match_quality`, `success` |

Legacy aliases accepted by loaders for compatibility:

* Similarity aliases: `best_scale` → `scale`, `best_R` → `R_yx`/`R_zyx`, `best_t` → `t_um_yx`/`t_um_zyx`, `best_bbox` → canonical bbox field.
* 3D voxel-size aliases: supported `voxel_size_*_um_zyx` variants are normalized to canonical `pixel_size_*_um_zyx` fields.

### 3D JSON/JSONL loading behavior (explicit)

* `load_nucleisky_transform_3d(path.json)` loads one record, normalizes supported legacy aliases, then validates required canonical structure.
* `load_transforms_any_3d(path)` accepts `.json` or `.jsonl`:
  * `.json`: same strict validation behavior as single-record loader.
  * `.jsonl`: each non-empty line is parsed, normalized, and validated; malformed JSON lines or invalid records raise `ValueError`.
* Malformed JSONL lines may surface directly as decoder exceptions (typically `json.JSONDecodeError`, a `ValueError` subclass). Unlike 2D strict JSONL mode, 3D loaders do not currently guarantee wrapped path/line-aware diagnostics for malformed JSON parse failures.
* Failure classes are distinct: malformed JSON text fails during decode (e.g., `json.JSONDecodeError`), while structurally valid JSON that violates canonical transform schema fails during semantic validation (validator-raised `ValueError`).

### Minimal canonical examples

2D canonical transform record (example):

```json
{
  "scale": 1.0,
  "R_yx": [[1.0, 0.0], [0.0, 1.0]],
  "t_um_yx": [0.0, 0.0],
  "A_px": [[1.0, 0.0], [0.0, 1.0]],
  "b_px": [0.0, 0.0],
  "pixel_size_full_um": 0.65,
  "pixel_size_crop_um": 0.65,
  "bbox_full_px_y0y1x0x1": [100, 200, 300, 400],
  "match_quality": {"frac_inliers": 0.8, "mean_error_um": 1.2},
  "success": true
}
```

3D canonical transform record (example):

```json
{
  "scale": 1.0,
  "R_zyx": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "t_um_zyx": [0.0, 0.0, 0.0],
  "A_px": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "b_px": [0.0, 0.0, 0.0],
  "pixel_size_full_um_zyx": [2.0, 0.5, 0.5],
  "pixel_size_crop_um_zyx": [2.0, 0.5, 0.5],
  "bbox_full_px_z0z1y0y1x0x1": [10, 30, 100, 220, 200, 340],
  "match_quality": {"frac_inliers": 0.75, "mean_error_um": 1.8},
  "success": true
}
```

---

## 6) QC images (what to look at)

QC visualization scripts generate multi-panel figures:

**Overlay image (`*_overlay.png`)**

* A composite figure featuring panels that display the Crop, the Reference ROI, the Overlay, and the Error map.
* In the Overlay panel, Green = target/reference (fixed) and Magenta = source/crop after warping (moving). White-ish boundaries indicate good structural overlap.

**Error map (The "1-SSIM" or "|diff|" panel)**

* This is included as a panel inside the `_overlay.png` image itself; it is not saved as a standalone file.
* For 2D, it displays `1 - SSIM` with a "magma" colormap (bright = high error).
* Bright regions indicate local mismatch (useful for spotting partial failures).

### Interpreting common outcomes

**Looks mostly white (in overlay)**

* Typically good alignment.

**Large green-only or magenta-only regions**

* The crop is not landing on the expected tissue region.
* Common causes: wrong pixel sizes, wrong channel used for matching, or a poor initial match.

**Error concentrated at edges**

* Often normal if margins include background, padding, or strongly different context.
* If error is high over the biologically relevant region, treat as suspect.

---

## 7) Segmentation masks (optional)

If segmentation is enabled/available in your workflow, you may find:

`segmentation_masks/`

* `labels_full.tif`
* `labels_crop.tif`

These are the integer label masks produced by algorithms like Cellpose or InstanSeg, saved securely as TIFFs. These are typically intermediate artifacts used for matching or downstream diagnostics. If you do not use segmentation-based steps, this folder will be absent.

---

## Troubleshooting

**I expected a full-grid TIFF but it’s missing**

* Full-grid TIFF is intentionally skipped on very large canvases to prevent memory crashes.
* Look for `aligned_on_full_px.zarr` instead, which handles large data chunks effectively.

**My channels look “swapped”**

* Remember: export channels are strictly `[fixed channels..., moving channels...]`.
* Confirm how many channels your full image and crop image each had at export time.

**My Z-stack looks wrong**

* Check the `axes` string you provided (e.g., `ZCYX` vs `CYX`).
* If axes do not exactly match the array's layout logic, the exporter will raise an error or broadcast axes incorrectly.

**The overlay is strongly offset**

* Verify pixel sizes for crop and full reference.
* Try matching on a different channel (higher-contrast structural marker).
* Check the transform metrics in the JSON / JSONL history.

---

## Next steps

* If your question is about *inputs* (axes, pixel sizes, channel choices), see **Data preparation**.
* If your question is about *how matching works end-to-end*, see **Workflow**.
* If your question is about *visual QC and plotting*, see **Visualization**.

## 8) 3D-specific data + export notes

* 3D volumes are expected in `(Z, Y, X)` order for export helpers.
* 3D aligned TIFF exports are written with `axes="ZYX"` metadata.
* Adaptive 3D runs save transform records in `transforms.jsonl`; these can be reused to export full-grid (`export_region="full"`) or ROI/bbox (`export_region="bbox"` / `"roi"`) results.
* When 3D segmentation labels are provided and `save_segmentation_masks=True`, masks are exported as TIFFs under `segmentation_masks/`.
