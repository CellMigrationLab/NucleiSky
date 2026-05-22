# NucleiSky2D API Reference

[:arrow_left: Documentation index](../index.md)

> **Scope:** This page is for **2D image registration** workflows.
> Looking for the volumetric workflow? See the **3D equivalent**: [3D API Reference](../3D/api.md).

This reference documents the public API for the NucleiSky2D registration pipeline. Signatures and behavior are taken directly from the library source to ensure accuracy.

## Core Pipeline

### `NucleiSky`

**Signature**

```python
def NucleiSky(
    centroids_crop_um,
    centroids_full_um,
    img_full,
    img_crop,
    ij_percentile_normalize,
    pixel_size_full_um,
    pixel_size_crop_um,
    matcher="graph",
    features_crop=None,
    features_full=None,
    df_full=None,
    df_crop=None,
    labels_full=None,
    labels_crop=None,
    matcher_config=None,
    matcher_kwargs=None,
    save_dir=None,
    save_prefix="match",
):

```

**Description**
Config-driven wrapper that runs the selected geometric matcher, validates inputs, and returns the best-fit similarity transform and match quality metrics.

**Arguments**

* **centroids_crop_um** (`array-like`): (N, 2) crop centroids in `(y, x)` order, in microns.
* **centroids_full_um** (`array-like`): (M, 2) reference centroids in `(y, x)` order, in microns.
* **img_full** (`array-like`): Reference image used for validation and optional debugging output.
* **img_crop** (`array-like`): Crop image to align.
* **ij_percentile_normalize** (`callable`): Normalization function used by downstream visualization/export helpers.
* **pixel_size_full_um** (`float`): Pixel size of the reference image in µm/px.
* **pixel_size_crop_um** (`float`): Pixel size of the crop image in µm/px.
* **matcher** (`str`, default: `"graph"`): Matcher name (`"graph"`, `"quad"`, `"triangles"`, `"hashing"`).
* **features_crop** (`array-like`, optional): Feature matrix for crop nuclei; required for graph matcher.
* **features_full** (`array-like`, optional): Feature matrix for full nuclei; required for graph matcher.
* **df_full** (`pandas.DataFrame`, optional): Full feature dataframe (used for dynamic scale estimation).
* **df_crop** (`pandas.DataFrame`, optional): Crop feature dataframe (used for dynamic scale estimation).
* **labels_full** (`array-like`, optional): Segmentation mask for the full image (debugging / export).
* **labels_crop** (`array-like`, optional): Segmentation mask for the crop image (debugging / export).
* **matcher_config** (`dict`, optional): Structured matcher configuration (merged with defaults).
* **matcher_kwargs** (`dict`, optional): Runtime overrides (flat or hierarchical) for matcher parameters.
* **save_dir** (`str | pathlib.Path`, optional): Output directory for debug artifacts.
* **save_prefix** (`str`, default: `"match"`): Prefix for saved debug outputs.

**Returns**

* `dict`: Result dictionary with keys `best_scale`, `best_R`, `best_t`, `bbox_full_px`, `match_quality`, `success`, and others. **Canonical field names (`scale`, `R_yx`, `t_um_yx`, `bbox_full_px_y0y1x0x1`, …) appear only in persisted JSON/JSONL records** written by `save_nucleisky_transform`, not in the live result dict.
* See the shared schema table: [Exports → Canonical transform schema](../exports.md#5-canonical-transform-schema-2d-and-3d).

---

### `run_adaptive_matching_and_export`

**Signature**

```python
def run_adaptive_matching_and_export(
    *,
    df_full,
    df_crop,
    img_full,
    img_crop,
    pixel_size_full_um: float,
    pixel_size_crop_um: float,
    result_dir: str | Path,
    cfg_selected: Optional[dict] = None,
    base_seed: int = 0,
    margin_px: int = 20,
    store_full_out: bool = False,
    max_total_time_s: Optional[float] = None,
    features_full=None,
    features_crop=None,
    img_full_seg=None,
    img_crop_seg=None,
    pixel_size_full_seg_um=None,
    pixel_size_crop_seg_um=None,
    labels_full=None,
    labels_crop=None,
    save_segmentation_masks: bool = True,
    ij_percentile_normalize: Optional[Any] = None,
) -> Tuple[Dict[str, Any], list]:

```

**Description**
End-to-end adaptive pipeline: validates inputs, selects matcher order based on nuclei count, runs adaptive matching, writes transform metadata, and exports aligned images (including optional segmentation masks).

**Arguments**

* **df_full** (`pandas.DataFrame`): Feature table for the full image (must include `centroid_y_um` and `centroid_x_um` columns).
* **df_crop** (`pandas.DataFrame`): Feature table for the crop image (must include `centroid_y_um` and `centroid_x_um` columns).
* **img_full** (`array-like`): Full reference image (2D).
* **img_crop** (`array-like`): Crop image (2D).
* **pixel_size_full_um** (`float`): Pixel size of the full image in µm/px.
* **pixel_size_crop_um** (`float`): Pixel size of the crop image in µm/px.
* **result_dir** (`str | pathlib.Path`): Base output directory; exports land under `matching/adaptive/exports_adaptive`.
* **cfg_selected** (`dict`, optional): Matcher configuration dictionary (defaults to `DEFAULT_MATCHER_CONFIG` when `None`).
* **base_seed** (`int`, default: `0`): Random seed used in adaptive matching.
* **margin_px** (`int`, default: `20`): Pixel padding around ROI exports.
* **store_full_out** (`bool`, default: `False`): Store full matcher outputs in the history list.
* **max_total_time_s** (`float`, optional): Optional global time budget for adaptive matching.
* **features_full** (`array-like`, optional): Feature matrix for the full image (graph matcher). When missing, the pipeline attempts to extract from `df_full['feature_vector']`.
* **features_crop** (`array-like`, optional): Feature matrix for the crop image (graph matcher). When missing, the pipeline attempts to extract from `df_crop['feature_vector']`.
* **img_full_seg** (`array-like`, optional): Full image at segmentation scale.
* **img_crop_seg** (`array-like`, optional): Crop image at segmentation scale.
* **pixel_size_full_seg_um** (`float`, optional): Pixel size at segmentation scale for the full image.
* **pixel_size_crop_seg_um** (`float`, optional): Pixel size at segmentation scale for the crop image.
* **labels_full** (`array-like`, optional): Segmentation mask for the full image.
* **labels_crop** (`array-like`, optional): Segmentation mask for the crop image.
* **save_segmentation_masks** (`bool`, default: `True`): Whether to save TIFF masks under the export directory.
* **ij_percentile_normalize** (`callable | Any`, optional): Optional normalization function for visualization/export.

**Returns**

* `Tuple[dict, list]`: `(best_result, history)` where `best_result` is the successful match output and `history` tracks all attempts.

## Pre-processing

### `scale_normalize_pair_for_segmentation`

**Signature**

```python
def scale_normalize_pair_for_segmentation(
    img_full,
    img_crop,
    pixel_size_full_um,
    pixel_size_crop_um,
    *,
    strategy="coarsest",
    manual_target_um=None,
    max_upsample=4.0,
    min_downsample=0.25,
    order=1,
    dtype_out=np.float32,
):

```

**Description**
Rescales full and crop images prior to segmentation so nuclei appear at comparable pixel sizes. Uses a target µm/px policy and returns both rescaled images and effective pixel sizes.

**Arguments**

* **img_full** (`array-like`): Full reference image to rescale.
* **img_crop** (`array-like`): Crop image to rescale.
* **pixel_size_full_um** (`float`): Original pixel size for the full image (µm/px).
* **pixel_size_crop_um** (`float`): Original pixel size for the crop image (µm/px).
* **strategy** (`str`, default: `"coarsest"`): Target selection policy (`"coarsest"`, `"finest"`, `"match_full"`, `"match_crop"`, `"custom"`).
* **manual_target_um** (`float`, optional): Explicit µm/px target when using the `"custom"` strategy.
* **max_upsample** (`float`, default: `4.0`): Upper bound on upsampling scale.
* **min_downsample** (`float`, default: `0.25`): Lower bound on downsampling scale.
* **order** (`int`, default: `1`): Interpolation order for rescaling.
* **dtype_out** (`numpy.dtype`, default: `np.float32`): Output dtype for rescaled images.

**Returns**

* `Tuple`: `(img_full_seg, img_crop_seg, pixel_size_full_seg_um, pixel_size_crop_seg_um, scale_factor_full, scale_factor_crop, target_um_per_px_requested)`.

---

### `ij_percentile_normalize`

**Signature**

```python
def ij_percentile_normalize(img, p_low=2, p_high=98):

```

**Description**
ImageJ-style percentile normalization: computes low/high percentiles on the raw image, linearly scales so `p_low -> 0` and `p_high -> 1`, then clips to `[0, 1]`. Used by visualization and export helpers.

**Arguments**

* **img** (`array-like`): Input image (will be cast to `float32`).
* **p_low** (`float`, default: `2`): Lower percentile for normalization.
* **p_high** (`float`, default: `98`): Upper percentile for normalization.

**Returns**

* `numpy.ndarray`: Normalized image in `float32`.

---

## Demo utilities

### `generate_random_crop`

**Signature**

```python
def generate_random_crop(
    img_full,
    patch_h,
    patch_w,
    zoom_range,
    max_angle_deg,
    pixel_size_um,
    rng=None,
):

```

**Description**
Generates a synthetic ROI crop from a full reference image by sampling a random center, applying a random rotation, and rescaling within a zoom range. Returns the crop, its effective pixel size (µm/px), and the sampled parameters. Intended for demo notebooks and synthetic testing.

**Arguments**

* **img_full** (`array-like`): Full reference image to sample from.
* **patch_h** (`int`): Output crop height in pixels.
* **patch_w** (`int`): Output crop width in pixels.
* **zoom_range** (`tuple[float, float]`): Min/max zoom factor to sample.
* **max_angle_deg** (`float`): Maximum absolute rotation in degrees.
* **pixel_size_um** (`float`): Pixel size of the full image in µm/px.
* **rng** (`numpy.random.Generator`, optional): Random number generator.

**Returns**

* `Tuple[numpy.ndarray, float, tuple]`: `(crop, crop_pixel_size_um, (cy, cx, angle_deg, zoom_factor))`.

## Advanced configuration & overrides

NucleiSky2D supports two layers of matcher configuration:

1. **Structured config** via `matcher_config` (merged with defaults).
2. **Runtime overrides** via `matcher_kwargs` (applied last).

### `matcher_config`

`matcher_config` should use a hierarchical structure with `_common` values plus matcher-specific sections:

```python
matcher_config = {
    "_common": {
        "inlier_radius_um": 1.5,
        "scale_min": 0.4,
        "scale_max": 2.5,
    },
    "graph": {
        "angle_max_deg": 15,
    },
}

```

### `matcher_kwargs`

`matcher_kwargs` can be either:

* **Hierarchical** (recommended), same structure as `matcher_config`
* **Flat**, with keys corresponding to matcher parameters

Unknown keys will raise an error when configuration is validated, so keep overrides aligned with documented defaults in `docs/2D/matchers.md`.

## Visualization

### `plot_warp_overlay`

**Signature**

```python
def plot_warp_overlay(
    plot_data: dict,
    save_dir=None,
    save_prefix="match",
):

```

**Description**
Renders 2D registration QC panels from precomputed warp outputs (`plot_data`). It displays source/target images, overlay(s), and `1 - SSIM` derived error map(s). When `save_dir` is provided, it writes a PNG figure named `{save_prefix}_overlay.png`.

**Arguments**

* **plot_data** (`dict`): Visualization payload produced by internal warp/QC utilities. Expected keys include image arrays (`crop_orig_n`, `full_roi_n`, `crop_warp_n`), optional reverse-warp arrays (`full_warp_n`), error maps (`err_1`, `err_2`), SSIM values (`ssim_val_1`, `ssim_val_2`), and display dimensions (`dims_1`, `dims_2`).
* **save_dir** (`str | pathlib.Path`, optional): Output directory for saved figure.
* **save_prefix** (`str`, default: `"match"`): Prefix used for the saved filename.

**Returns**

* `None`: The function renders matplotlib figures and optionally saves them to disk.

**Overlay color convention**

* **Green** channel = target image.
* **Magenta** channels (red + blue) = warped source image.
* **White/grey overlap** indicates stronger spatial agreement.

### `show_alignment_original_and_rescaled`

**Signature**

```python
def show_alignment_original_and_rescaled(
    res,
    *,
    ij_percentile_normalize=ij_percentile_normalize,
    img_full_orig=None,
    img_crop_orig=None,
    pixel_size_full_orig_um=None,
    pixel_size_crop_orig_um=None,
    save_dir=None,
    margin_um=5.0,
    also_warp_full_to_crop=True,
    compute_warp=True,
    verbose=False,
):

```

**Description**
Displays alignment overlays on the original-resolution images. Optionally saves plots and warped overlays under a `original/` subdirectory.

**Arguments**

* **res** (`dict`): Result dictionary from `NucleiSky` containing transform parameters.
* **ij_percentile_normalize** (`callable`): Normalization function used in visualization.
* **img_full_orig** (`array-like`): Original full-resolution reference image.
* **img_crop_orig** (`array-like`): Original full-resolution crop image.
* **pixel_size_full_orig_um** (`float`): Pixel size of the full image (µm/px) at original scale.
* **pixel_size_crop_orig_um** (`float`): Pixel size of the crop image (µm/px) at original scale.
* **save_dir** (`str | pathlib.Path`, optional): Directory for saving visualization outputs.
* **margin_um** (`float`, default: `5.0`): Margin around ROI in microns.
* **also_warp_full_to_crop** (`bool`, default: `True`): Whether to warp full-to-crop as well as crop-to-full.
* **compute_warp** (`bool`, default: `True`): Whether to compute warp outputs for plotting.
* **verbose** (`bool`, default: `False`): Print diagnostic output when missing transforms.

**Returns**

* `BBox | None`: Bounding box for the aligned ROI when available; otherwise `None`.

## I/O & Export

### `make_result_dir`

**Signature**

```python
def make_result_dir(big_image_path=None, root_dir=None, tag="NucleiSky"):

```

**Description**
Creates a timestamped output directory. When `root_dir` is not provided, the folder is placed next to `big_image_path` (if supplied) or the current working directory.

**Arguments**

* **big_image_path** (`str | pathlib.Path`, optional): Path to a reference image used to choose the default parent directory.
* **root_dir** (`str | pathlib.Path`, optional): Explicit parent directory for the results folder.
* **tag** (`str`, default: `"NucleiSky"`): Prefix for the created directory name.

**Returns**

* `pathlib.Path`: Path to the created results directory.

---

### `load_image`

**Signature**

```python
def load_image(path_str: str | Path):

```

**Description**
Loads image data from a path, selecting an appropriate backend for TIFF, NumPy `.npy`, OME-Zarr/Zarr (directory or `.zarr`), or common image formats (via scikit-image fallback).

**Arguments**

* **path_str** (`str | pathlib.Path`): Path to the image or zarr group.

**Returns**

* `numpy.ndarray | zarr.Array | zarr.Group`: Loaded image object.

---

### `save_nucleisky_transform`

**Signature**

```python
def save_nucleisky_transform(
    res: dict,
    out_path: str | Path,
    *,
    matcher_name: str = "unknown",
    pixel_size_full_um: float,
    pixel_size_crop_um: float,
    require_success: bool = True,
):

```

**Description**
Writes a JSON record containing similarity transform parameters, pixel sizes, affine coefficients in pixels, and match quality metadata.

**Arguments**

* **res** (`dict`): Result dictionary from `NucleiSky` (or any dict) containing `best_scale`, `best_R`, and `best_t`. These keys are read directly; canonical names (`scale`, `R_yx`, `t_um_yx`) are **not** accepted by this function.
* **out_path** (`str | pathlib.Path`): Output JSON path.
* **matcher_name** (`str`, default: `"unknown"`): Label stored in the JSON record.
* **pixel_size_full_um** (`float`): Full image pixel size in µm/px.
* **pixel_size_crop_um** (`float`): Crop image pixel size in µm/px.
* **require_success** (`bool`, default: `True`): Raise if the result is not successful.

**Returns**

* `dict`: JSON-serializable transform record that was written to disk.

---

### `load_nucleisky_transform`

**Signature**

```python
def load_nucleisky_transform(path: str | Path) -> dict:

```

**Description**
Loads a transform JSON record and validates that the required affine and pixel size keys exist.

**Arguments**

* **path** (`str | pathlib.Path`): Path to the transform JSON.

**Returns**

* `dict`: Transform record with `A_px`, `b_px`, `pixel_size_full_um`, and `pixel_size_crop_um`.

---

### `load_transforms_any`

**Signature**

```python
def load_transforms_any(path_str: str, *, strict: bool = False):

```

**Description**
Loads transform records from either a single JSON file or a JSONL file (one record per line). By default (`strict=False`) JSONL loading is permissive for backward compatibility. With `strict=True`, each line is validated and failures report file/line context.

**Arguments**

* **path_str** (`str | pathlib.Path`): Path to `.json` or `.jsonl` file.
* **strict** (`bool`, default: `False`): When `True`, validate each JSONL record strictly and raise line-aware errors for malformed/invalid records.

**Returns**

* `list[dict]`: List of transform records with `_source_path`, `_source_kind`, and `_line` fields.

---

### `append_transform_jsonl`

**Signature**

```python
def append_transform_jsonl(record: dict, out_jsonl: str | Path):
```

**Description**
Appends one JSON object per line to a JSONL transform-history file. Parent directories are created automatically if needed.

**Arguments**

* **record** (`dict`): Transform record to append. Recommended canonical fields are defined in [Exports → Canonical transform schema](../exports.md#5-canonical-transform-schema-2d-and-3d).
* **out_jsonl** (`str | pathlib.Path`): Destination `.jsonl` file path.

**Behavior notes**

* This helper performs serialization/appending only; it does not validate schema before writing.
* Use `load_transforms_any(..., strict=True)` when you need strict record validation on read.

---

### `get_pixel_size_um_from_tiff`

**Signature**

```python
def get_pixel_size_um_from_tiff(
    file_path: str,
    *,
    return_details: bool = False,
    return_xy: bool = False,
    page_index: int = 0,
    allow_guess_unit_when_missing: bool = False,
    anisotropy_warn_threshold: float = 0.01,
):

```

**Description**
Extracts pixel size in µm/px from TIFF metadata, prioritizing OME-XML physical sizes, then TIFF resolution tags (with ImageJ unit hints), and finally ImageJ z-spacing as a fallback.

**Arguments**

* **file_path** (`str | pathlib.Path`): Path to the TIFF file.
* **return_details** (`bool`, default: `False`): Return a details dictionary in addition to the value.
* **return_xy** (`bool`, default: `False`): Return `(x_um, y_um)` instead of a mean value.
* **page_index** (`int`, default: `0`): TIFF page index to inspect.
* **allow_guess_unit_when_missing** (`bool`, default: `False`): Allow heuristic unit guesses when resolution unit is missing.
* **anisotropy_warn_threshold** (`float`, default: `0.01`): Relative XY anisotropy threshold for warnings.

**Returns**

* `float | tuple | None` (or `(value, details)` if `return_details=True`): Mean pixel size in µm/px, `(x_um, y_um)` when `return_xy=True`, or `None` when no metadata is found.

---

### `export_aligned_dataset`

**Signature**

```python
def export_aligned_dataset(
    res: dict,
    *,
    out_dir: str | Path,
    img_full,
    img_crop,
    pixel_size_full_um: float,
    pixel_size_crop_um: float,
    axes_full="YX",
    axes_crop="YX",
    export_region: str = "roi",
    margin_px: int = 20,
    bbox_full_px: BBox | tuple | list | dict | None = None,
    bbox_convention: str = "y0y1x0x1",
    always_two_stacks: bool = False,
    pixel_size_equal_rtol: float = 1e-3,
    order_intensity: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    as_uint16_if_float: bool = False,
    format: str = "tiff",
):

```

**Description**
Exports aligned full and warped crop imagery to TIFF or OME-Zarr. Supports both matcher outputs and saved transform records; ROI export is the default behavior.

**Arguments**

* **res** (`dict`): Transform output from `NucleiSky` or saved transform record.
* **out_dir** (`str | pathlib.Path`): Output directory.
* **img_full** (`array-like`): Full reference image.
* **img_crop** (`array-like`): Crop image to warp.
* **pixel_size_full_um** (`float`): Pixel size of the full image in µm/px.
* **pixel_size_crop_um** (`float`): Pixel size of the crop image in µm/px.
* **axes_full** (`str`, default: `"YX"`): Axes string for the full image (e.g., `"YX"`, `"ZCYX"`).
* **axes_crop** (`str`, default: `"YX"`): Axes string for the crop image.
* **export_region** (`str`, default: `"roi"`): `"roi"` for a bounding box ROI or `"full"` for the full reference grid.
* **margin_px** (`int`, default: `20`): Pixel margin around ROI exports.
* **bbox_full_px** (`BBox | tuple | list | dict | None`, optional): Optional ROI bounding box override.
* **bbox_convention** (`str`, default: `"y0y1x0x1"`): Coordinate convention for `bbox_full_px` inputs.
* **always_two_stacks** (`bool`, default: `False`): Force two-channel output even when images have similar pixel size.
* **pixel_size_equal_rtol** (`float`, default: `1e-3`): Relative tolerance used when comparing pixel sizes.
* **order_intensity** (`int`, default: `1`): Interpolation order for warping intensities.
* **mode** (`str`, default: `"constant"`): Padding mode for warping.
* **cval** (`float`, default: `0.0`): Constant padding value for warping.
* **as_uint16_if_float** (`bool`, default: `False`): Convert float images to uint16 when exporting.
* **format** (`str`, default: `"tiff"`): Output format (`"tiff"` or any string containing `"zarr"`).

**Returns**

* `dict`: Export metadata including output paths, format, and the bounding box used.

**Note:** `export_aligned_imagej_stacks` is an alias of `export_aligned_dataset`.

---

### `warp_dataset_with_transform`

**Signature**

```python
def warp_dataset_with_transform(
    moving,
    *,
    A_px,
    b_px,
    out_shape_yx,
    axes="YX",
    order=1,
    mode="constant",
    cval=0.0,
):

```

**Description**
Warps an in-memory dataset into a target grid using an affine transform expressed in pixels. This is useful when you want arrays instead of on-disk exports. Internally, the data are expanded into `TZCYX` order for consistent processing.

**Arguments**

* **moving** (`array-like`): Input data to warp.
* **A_px** (`array-like`): `2x2` affine matrix in pixel space.
* **b_px** (`array-like`): Length-2 translation vector in pixel space.
* **out_shape_yx** (`tuple[int, int]`): Output `(Y, X)` shape.
* **axes** (`str`, default: `"YX"`): Axes string describing `moving` (e.g., `"CYX"`, `"ZCYX"`, `"TZCYX"`).
* **order** (`int`, default: `1`): Interpolation order.
* **mode** (`str`, default: `"constant"`): Padding mode used by `scipy.ndimage.affine_transform`.
* **cval** (`float`, default: `0.0`): Constant value used for padding when `mode="constant"`.

**Returns**

* `numpy.ndarray`: Warped array in the same axes order as the input.

---

### `inspect_image_header`

**Signature**

```python
def inspect_image_header(path_str: str):

```

**Description**
Reads image metadata (shape, dtype, axes, pixel size) without loading full image data. Supports TIFF, NPY, and Zarr inputs (requires `zarr` installed for Zarr).

**Arguments**

* **path_str** (`str | pathlib.Path`): Path to the image file or directory.

**Returns**

* `dict`: Metadata dictionary with keys like `path`, `kind`, `shape`, `dtype`, `axes`, and `pixel_size_um`.

## Segmentation & Features

### `segment_nuclei_dispatch`

**Signature**

```python
def segment_nuclei_dispatch(
    img,
    method,
    pixel_size_um,
    settings=None,
    segmentor: Optional[Segmentor] = None,
):

```

**Description**
Unified dispatch for multiple segmentation backends. Delegates to the specified backend (thresholding, Cellpose, or InstanSeg), using either the provided `Segmentor` instance or the global default.

**Arguments**

* **img** (`array-like`): Input image to segment.
* **method** (`str`): Segmentation backend (`"threshold"`, `"cellpose"`, `"instanseg"`).
* **pixel_size_um** (`float`): Pixel size in µm/px (used by some backends).
* **settings** (`dict`, optional): Backend-specific settings dict with per-method keys.
* **segmentor** (`Segmentor`, optional): Optional custom segmentor instance.

**Returns**

* `numpy.ndarray`: 2D label image of segmented nuclei.

---

### `extract_nuclear_features`

**Signature**

```python
def extract_nuclear_features(
    label_img,
    intensity_img=None,
    pixel_size_um=1.0,
    k_neighbors=10,
    min_area_px=None,
    max_area_px=None,
    edge_margin_px=0,
):

```

**Description**
Computes geometric and neighborhood features for labeled nuclei. Returns a dataframe with centroid coordinates, physical measurements, and a per-nucleus feature vector. (`intensity_img` is accepted for API compatibility.)

**Arguments**

* **label_img** (`array-like`): 2D integer label image.
* **intensity_img** (`array-like`, optional): Intensity image for weighted measurements.
* **pixel_size_um** (`float`, default: `1.0`): Pixel size in µm/px.
* **k_neighbors** (`int`, default: `10`): Number of neighbors for kNN distance features.
* **min_area_px** (`int | None`, default: `None`): Minimum nucleus area to retain. When `None`, no minimum-area filtering is applied.
* **max_area_px** (`float | None`, default: `None`): Maximum nucleus area to retain. When `None`, no maximum-area filtering is applied.
* **edge_margin_px** (`int`, default: `0`): Margin for excluding nuclei touching borders.

**Returns**

* `pandas.DataFrame`: Features table containing centroid columns, shape metrics, and `feature_vector` entries for each nucleus.

---

### `add_centroids_orig_px_columns`

**Signature**

```python
def add_centroids_orig_px_columns(df, scale_factor, *, y_col="centroid_y_px", x_col="centroid_x_px"):

```

**Description**
Adds `centroid_y_px_orig` and `centroid_x_px_orig` columns to a feature dataframe after segmentation on rescaled images.

**Arguments**

* **df** (`pandas.DataFrame`): Feature table containing centroid pixel columns.
* **scale_factor** (`float`): Scale factor used to rescale the image for segmentation.
* **y_col** (`str`, default: `"centroid_y_px"`): Column name for Y centroids.
* **x_col** (`str`, default: `"centroid_x_px"`): Column name for X centroids.

**Returns**

* `pandas.DataFrame`: The input dataframe with added original-pixel centroid columns (or unchanged if empty).

---

### `extract_centroids_um`

**Signature**

```python
def extract_centroids_um(df, *, name: str):

```

**Description**
Extracts `(N, 2)` centroid coordinates in microns from a feature dataframe and validates shape, finiteness, and minimum count.

**Arguments**

* **df** (`pandas.DataFrame`): Feature table containing `centroid_y_um` and `centroid_x_um`.
* **name** (`str`): Label used in error messages.

**Returns**

* `numpy.ndarray`: `(N, 2)` array of centroids in `(y, x)` order.
