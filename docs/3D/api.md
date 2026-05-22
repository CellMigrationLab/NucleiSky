# NucleiSky3D API Reference

[:arrow_left: Documentation index](../index.md)

This reference documents the public API for the NucleiSky3D registration pipeline. Signatures and behavior are taken directly from the library source to ensure accuracy.

## Core Pipeline

### `NucleiSky3D`

**Signature**

```python
def NucleiSky3D(
    centroids_crop_um,
    centroids_full_um,
    full_shape_px_zyx,
    crop_shape_px_zyx,
    pixel_size_full_um_zyx,
    pixel_size_crop_um_zyx,
    matcher: str = "pyramid",
    *,
    matcher_config: Dict[str, Any] | None = None,
    matcher_kwargs: Dict[str, Any] | None = None,
    df_full=None,
    df_crop=None,
    return_dists: bool = False,
) -> Dict[str, Any]:

```

**Description**
Config-driven wrapper that runs the selected 3D geometric matcher, validates voxel-size and centroid inputs, applies deduplication/sanitization, and returns the best-fit 3D similarity transform with match quality metrics.

**Arguments**

* **centroids_crop_um** (`array-like`): `(N, 3)` crop centroids in **`(z, y, x)`** order, in microns.
* **centroids_full_um** (`array-like`): `(M, 3)` reference centroids in **`(z, y, x)`** order, in microns.
* **full_shape_px_zyx** (`tuple[int, int, int]`): Full/reference volume shape in **`(z, y, x)`** order.
* **crop_shape_px_zyx** (`tuple[int, int, int]`): Crop/patch volume shape in **`(z, y, x)`** order.
* **pixel_size_full_um_zyx** (`float | tuple[float, float, float]`): Full/reference voxel size in µm/voxel in **`(z, y, x)`** order.
* **pixel_size_crop_um_zyx** (`float | tuple[float, float, float]`): Crop voxel size in µm/voxel in **`(z, y, x)`** order.
* **matcher** (`str`, default: `"pyramid"`): Matcher name (`"pyramid"` or `"hashing"`).
* **matcher_config** (`dict`, optional): Structured matcher configuration (merged with defaults).
* **matcher_kwargs** (`dict`, optional): Runtime overrides (flat or hierarchical) for matcher parameters.
* **df_full** (`pandas.DataFrame`, optional): Full feature dataframe for matcher-specific constraints and metadata.
* **df_crop** (`pandas.DataFrame`, optional): Crop feature dataframe for matcher-specific constraints and metadata.
* **return_dists** (`bool`, default: `False`): Whether to include nearest-neighbor distance vectors in the returned result.

**Note**

* `return_dists` controls whether NN distances are returned; these arrays can be large and may cause JSON serialization failures if left `True` in production workflows.

**Returns**

* `dict`: Result dictionary with keys `best_scale`, `best_R`, `best_t`, `best_bbox`, `match_quality`, `success`, and others. **Canonical field names (`scale`, `R_zyx`, `t_um_zyx`, `bbox_full_px_z0z1y0y1x0x1`, …) appear only in persisted JSON/JSONL records** written by `save_nucleisky_transform_3d`, not in the live result dict.
* Legacy aliases (`best_scale`, `best_R`, `best_t`, `best_bbox`) are normalized by loaders when reading saved records.
* See shared schema and loader notes: [Exports → Canonical transform schema](../exports.md#5-canonical-transform-schema-2d-and-3d).

---

### `run_adaptive_matching_and_export_3d`

**Signature**

```python
def run_adaptive_matching_and_export_3d(
    *,
    df_full,
    df_crop,
    img_full_orig=None,
    img_crop_orig=None,
    pixel_size_full_orig_um_zyx,
    pixel_size_crop_orig_um_zyx,
    result_dir: str,
    cfg_selected: Optional[dict] = None,
    base_seed: int = 0,
    store_full_out: bool = False,
    max_total_time_s: Optional[float] = None,
    img_full_seg=None,
    img_crop_seg=None,
    pixel_size_full_seg_um_zyx=None,
    pixel_size_crop_seg_um_zyx=None,
    labels_full=None,
    labels_crop=None,
    save_segmentation_masks: bool = True,
    verbose: bool = True,
    print_fn=print,
) -> Tuple[Dict[str, Any], list]:

```

**Description**
End-to-end adaptive 3D pipeline: validates precomputed feature tables, normalizes dual-scale voxel metadata, runs adaptive matcher selection (Pyramid, then Hashing depending on nuclei counts), writes transform/history artifacts, and exports aligned outputs.

**Arguments**

* **df_full** (`pandas.DataFrame`): Feature table for the full/reference volume; must include centroid columns in um or px.
* **df_crop** (`pandas.DataFrame`): Feature table for the crop volume; must include centroid columns in um or px.
* **img_full_orig** (`array-like`, optional): Original-resolution full/reference volume used for final TIFF export.
* **img_crop_orig** (`array-like`, optional): Original-resolution crop volume used for final TIFF export.
* **pixel_size_full_orig_um_zyx** (`float | tuple[float, float, float]`): Original full voxel size in µm/voxel in **`(z, y, x)`** order.
* **pixel_size_crop_orig_um_zyx** (`float | tuple[float, float, float]`): Original crop voxel size in µm/voxel in **`(z, y, x)`** order.
* **result_dir** (`str`): Base output directory; exports are written under `matching/adaptive_3d/exports_adaptive`.
* **cfg_selected** (`dict`, optional): Matcher configuration dictionary (defaults to `DEFAULT_MATCHER_CONFIG` when `None`).
* **base_seed** (`int`, default: `0`): Random seed used in adaptive matching.
* **store_full_out** (`bool`, default: `False`): Store full matcher outputs in the history list.
* **max_total_time_s** (`float`, optional): Optional global time budget for adaptive matching.
* **img_full_seg** (`array-like`, optional): Segmentation-scale full/reference volume used for matching context.
* **img_crop_seg** (`array-like`, optional): Segmentation-scale crop volume used for matching context.
* **pixel_size_full_seg_um_zyx** (`float | tuple[float, float, float]`, optional): Segmentation-scale full voxel size in µm/voxel in **`(z, y, x)`** order.
* **pixel_size_crop_seg_um_zyx** (`float | tuple[float, float, float]`, optional): Segmentation-scale crop voxel size in µm/voxel in **`(z, y, x)`** order.
* **labels_full** (`array-like`, optional): Full/reference segmentation label volume (also used for shape context and optional mask export).
* **labels_crop** (`array-like`, optional): Crop segmentation label volume (also used for shape context and optional mask export).
* **save_segmentation_masks** (`bool`, default: `True`): Whether to save segmentation masks as TIFFs under the export directory.
* **verbose** (`bool`, default: `True`): Enables/disables adaptive pipeline log output.
* **print_fn** (callable, default: `print`): Logging sink function used when `verbose=True`.

**Returns**

* `Tuple[dict, list]`: `(best_out, history)` where `best_out` is the best adaptive 3D match output and `history` tracks all matcher attempts with scores and metadata.

## Preprocessing & Features

### `scale_normalize_pair_for_segmentation`

**Signature**

```python
def scale_normalize_pair_for_segmentation(
    img_full,
    img_crop,
    voxel_size_full_um_zyx,
    voxel_size_crop_um_zyx,
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
Rescales full and crop 3D volumes before segmentation so nuclei appear at comparable physical scale, then returns the rescaled volumes with effective voxel metadata and applied scale factors. *Developer Note:* When downsampling (`s < 1.0`) and `order > 0`, the function automatically applies anti-aliasing via `skimage.transform.rescale`.

**Arguments**

* **img_full** (`array-like`): Full/reference 3D volume in **`(z, y, x)`** order.
* **img_crop** (`array-like`): Crop 3D volume in **`(z, y, x)`** order.
* **voxel_size_full_um_zyx** (`float | tuple[float, float, float]`): Original full voxel size in µm/voxel. Scalar values are broadcast to isotropic spacing; tuples must be in **`(z, y, x)`** order.
* **voxel_size_crop_um_zyx** (`float | tuple[float, float, float]`): Original crop voxel size in µm/voxel. Scalar values are broadcast to isotropic spacing; tuples must be in **`(z, y, x)`** order.
* **strategy** (`str`, default: `"coarsest"`): Target spacing policy (`"coarsest"`, `"finest"`, `"match_full"`, `"match_crop"`, `"custom"`).
* **manual_target_um** (`float | tuple[float, float, float]`, optional): Explicit target voxel size for `"custom"`; interpreted as isotropic when scalar or anisotropic **`(z, y, x)`** when length-3.
* **max_upsample** (`float`, default: `4.0`): Upper bound on per-axis upsampling scale.
* **min_downsample** (`float`, default: `0.25`): Lower bound on per-axis downsampling scale.
* **order** (`int`, default: `1`): Interpolation order used for volumetric resampling.
* **dtype_out** (`numpy.dtype`, default: `np.float32`): Output dtype for the rescaled volumes.

**Returns**

* `Tuple`: A 7-element tuple `(img_full_seg, img_crop_seg, voxel_full_seg_um_zyx, voxel_crop_seg_um_zyx, scale_factor_full_zyx, scale_factor_crop_zyx, target_um_per_voxel_requested_zyx)` where:
* `img_full_seg`, `img_crop_seg`: Rescaled 3D volumes.
* `voxel_full_seg_um_zyx`, `voxel_crop_seg_um_zyx`: Effective post-rescale voxel sizes (µm/voxel, **`(z, y, x)`**).
* `scale_factor_full_zyx`, `scale_factor_crop_zyx`: Per-axis scale factors applied to each input volume.
* `target_um_per_voxel_requested_zyx`: Requested common target spacing in **`(z, y, x)`**.



---

### `extract_nuclear_features_3d`

**Signature**

```python
def extract_nuclear_features_3d(label_img_3d, pixel_size_um=1.0, k_neighbors=5):

```

**Description**
Extracts per-nucleus 3D geometric and local-neighborhood features from a labeled volume using SimpleITK shape statistics and KD-tree nearest-neighbor queries. *Note: This function explicitly requires the `SimpleITK` package to be installed in your environment, otherwise it will raise an `ImportError*`.

**Arguments**

* **label_img_3d** (`array-like`): Integer label volume in **`(z, y, x)`** order; background should be `0` and each nucleus should have a unique positive label.
* **pixel_size_um** (`float | tuple[float, float, float]`, default: `1.0`): Physical voxel size used to convert between pixel and micron coordinates. A scalar is treated as isotropic; a length-3 tuple is interpreted as **`(z, y, x)`**.
* **k_neighbors** (`int`, default: `5`): Number of nearest-neighbor distance columns (`nn1_dist_um ... nnk_dist_um`) to compute per nucleus.

**Returns**

* `pandas.DataFrame`: One row per labeled nucleus with centroid, size, shape, neighborhood, and matcher-ready fields. Core columns include:
* Identification and centroids: `label`, `centroid_z_px`, `centroid_y_px`, `centroid_x_px`, `centroid_z_um`, `centroid_y_um`, `centroid_x_um`.
* Morphology: `volume_voxels`, `volume_um3`, `surface_area_um2`, `equiv_spherical_diameter_um`, `sphericity`, `volume_norm`.
* Spatial context: `nn1_dist_um ... nn{k_neighbors}_dist_um`, `local_density_r20`, `local_density_norm`.
* Matcher feature embedding: `feature_vector` (stacked `[volume_norm, sphericity, local_density_norm]`).



## Demo utilities

### `generate_random_subvolume_3d`

**Signature**

```python
def generate_random_subvolume_3d(
    img_full: np.ndarray,
    crop_shape_zyx: Tuple[int, int, int],
    scale_range: Tuple[float, float],
    voxel_size_um: float | Iterable[float],
    rng: np.random.Generator | None = None,
):

```

**Description**
Generates a synthetic 3D crop from a full/reference volume by sampling a random center, random scale, and random in-plane (strictly Z-axis / XY plane) rotation, then interpolating with trilinear sampling. Returns the crop, the effective crop voxel size, and ground-truth transform parameters (`scale`, `R`, `t`) for matcher validation and notebook demos. *Developer Note:* The function safely handles out-of-bounds coordinates near edges by using `mode="reflect"` padding.

**Arguments**

* **img_full** (`numpy.ndarray`): Full reference volume in **`(z, y, x)`** order.
* **crop_shape_zyx** (`tuple[int, int, int]`): Output crop shape in voxels, in **`(z, y, x)`**.
* **scale_range** (`tuple[float, float]`): Uniform scale sampling range `(min, max)` with `0 < min < max`.
* **voxel_size_um** (`float | tuple[float, float, float]`): Full-volume voxel size in µm/voxel. Scalars are treated as isotropic; tuples must be in **`(z, y, x)`** order.
* **rng** (`numpy.random.Generator`, optional): Random number generator. If omitted, a default generator is created.

**Returns**

* `Tuple[numpy.ndarray, numpy.ndarray, dict]`: `(crop, crop_voxel_size_um, ground_truth)` where:
* `crop`: Synthetic crop volume in **`(z, y, x)`** order.
* `crop_voxel_size_um`: Crop voxel size (µm/voxel) in **`(z, y, x)`**, computed as `voxel_size_um / scale`.
* `ground_truth`: Dictionary with:
* `scale` (`float`): Sampled similarity scale.
* `R` (`numpy.ndarray`): `3x3` rotation matrix (Z-axis rotation only).
* `t` (`numpy.ndarray`): Translation in microns (`(z, y, x)`), corresponding to the sampled crop center in full-volume coordinates.





**Example**

```python
import numpy as np
from nucleisky3d.demo_utils import generate_random_subvolume_3d

# Example full volume (replace with your microscopy stack)
img_full = np.random.default_rng(0).normal(size=(96, 256, 256)).astype(np.float32)

rng = np.random.default_rng(42)
crop, crop_voxel_um, gt = generate_random_subvolume_3d(
    img_full=img_full,
    crop_shape_zyx=(48, 96, 96),
    scale_range=(0.8, 1.3),
    voxel_size_um=(1.0, 0.3, 0.3),
    rng=rng,
)

print(crop.shape)                 # (48, 96, 96)
print(crop_voxel_um)              # effective voxel size in µm
print(gt["scale"], gt["R"], gt["t"])  # known transform parameters

```

## I/O & Export

### `similarity_um_to_affine_px_3d`

**Signature**

```python
def similarity_um_to_affine_px_3d(
    *,
    best_scale: float,
    best_R: np.ndarray,
    best_t: np.ndarray,
    pixel_size_full_um: Iterable[float],
    pixel_size_crop_um: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:

```

**Description**
Converts a 3D similarity transform expressed in microns (`scale`, `R`, `t`) into affine coefficients in pixel coordinates for forward mapping from crop to full volume. **All vectors, voxel sizes, and matrix coordinates are interpreted in `(z, y, x)` axis order as column vectors**.

**Arguments**

* **best_scale** (`float`): Positive finite similarity scale.
* **best_R** (`numpy.ndarray`): `3x3` rotation matrix for the similarity transform, ordered for **`(z, y, x)`** coordinates.
* **best_t** (`numpy.ndarray`): Length-3 translation vector in microns, in **`(z, y, x)`** order.
* **pixel_size_full_um** (`float | tuple[float, float, float]`): Full/reference voxel size in µm/voxel. Scalars are broadcast isotropically; length-3 values must be **`(z, y, x)`**.
* **pixel_size_crop_um** (`float | tuple[float, float, float]`): Crop voxel size in µm/voxel. Scalars are broadcast isotropically; length-3 values must be **`(z, y, x)`**.

**Returns**

* `tuple[np.ndarray, np.ndarray]`: `(A_px, b_px)` where `A_px` is `3x3` and `b_px` is length-3 such that `full_px = A_px @ crop_px + b_px` in **`(z, y, x)`** pixel coordinates.

---

### `warp_crop_to_full_volume`

**Signature**

```python
def warp_crop_to_full_volume(
    img_crop: np.ndarray,
    *,
    full_shape_zyx: Iterable[int],
    pixel_size_full_um,
    pixel_size_crop_um,
    res: dict | None = None,
    best_scale: float | None = None,
    best_R: np.ndarray | None = None,
    best_t: np.ndarray | None = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
) -> np.ndarray:

```

**Description**
Warps a crop volume into the full-volume coordinate space using the estimated alignment transform. The warp is computed by converting the similarity transform to pixel-space affine coefficients and resampling onto the requested full-grid output shape. **Input/output volumes and all shape/spacing parameters must follow `(z, y, x)` axis order**.

**Arguments**

* **img_crop** (`numpy.ndarray`): Crop volume to warp, shape `(Z, Y, X)` in **`(z, y, x)`** order.
* **full_shape_zyx** (`Iterable[int]`): Output full-volume shape as `(Z, Y, X)` / **`(z, y, x)`**.
* **pixel_size_full_um** (`float | tuple[float, float, float]`): Full/reference voxel size in µm/voxel, interpreted as **`(z, y, x)`**.
* **pixel_size_crop_um** (`float | tuple[float, float, float]`): Crop voxel size in µm/voxel, interpreted as **`(z, y, x)`**.
* **res** (`dict`, optional): Result dictionary containing `best_scale`, `best_R`, and `best_t`.
* **best_scale** (`float`, optional): Explicit similarity scale override.
* **best_R** (`numpy.ndarray`, optional): Explicit `3x3` rotation override in **`(z, y, x)`** coordinates.
* **best_t** (`numpy.ndarray`, optional): Explicit translation override in microns, **`(z, y, x)`**.
* **order** (`int`, default: `1`): Interpolation order passed to `scipy.ndimage.affine_transform`.
* **mode** (`str`, default: `"constant"`): Boundary mode passed to `affine_transform`.
* **cval** (`float`, default: `0.0`): Fill value used when `mode="constant"`.
* **output_dtype** (`numpy.dtype`, optional): Optional output dtype cast for the warped volume.

**Returns**

* `numpy.ndarray`: Warped crop sampled on the full-grid shape, with axes in **`(z, y, x)`** order.

---

### `export_aligned_crop_tiff`

**Signature**

```python
def export_aligned_crop_tiff(
    img_full: np.ndarray,
    img_crop: np.ndarray,
    *,
    output_path: str | Path,
    pixel_size_full_um,
    pixel_size_crop_um,
    as_uint16_if_float: bool = False,
    res: dict | None = None,
    best_scale: float | None = None,
    best_R: np.ndarray | None = None,
    best_t: np.ndarray | None = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
    export_region: str = "full",
    write_metadata_json: bool = False,
) -> Path:

```

**Description**
Exports the aligned crop as an ImageJ-compatible TIFF in reference/full-volume space. Depending on `export_region`, it writes either a full-grid aligned volume (`"full"`) or an ROI cropped to `res["best_bbox"]` (`"bbox"`/`"roi"`), and can optionally write a sidecar JSON. TIFF metadata is written with `axes="ZYX"`, physical spacing properties, and safely triggers BigTIFF compatibility if the resulting array  GB; **all inputs are expected in `(z, y, x)` order**.

**Arguments**

* **img_full** (`numpy.ndarray`): Reference volume used for output grid sizing, shape `(Z, Y, X)` in **`(z, y, x)`** order.
* **img_crop** (`numpy.ndarray`): Crop volume to align and export, shape `(Z, Y, X)` in **`(z, y, x)`** order.
* **output_path** (`str | pathlib.Path`): Destination TIFF file path.
* **pixel_size_full_um** (`float | tuple[float, float, float]`): Full/reference voxel size in µm/voxel; tuple form must be **`(z, y, x)`**.
* **pixel_size_crop_um** (`float | tuple[float, float, float]`): Crop voxel size in µm/voxel; tuple form must be **`(z, y, x)`**.
* **as_uint16_if_float** (`bool`, default: `False`): If `True`, floating point outputs are min/max scaled from `[0, 1]` and cast to `uint16` (`0–65535`).
* **res** (`dict`, optional): Result dictionary containing transform fields (and `best_bbox` when exporting `"bbox"`).
* **best_scale** (`float`, optional): Explicit similarity scale override.
* **best_R** (`numpy.ndarray`, optional): Explicit `3x3` rotation override in **`(z, y, x)`** coordinates.
* **best_t** (`numpy.ndarray`, optional): Explicit translation override in microns, **`(z, y, x)`**.
* **order** (`int`, default: `1`): Interpolation order for warping.
* **mode** (`str`, default: `"constant"`): Boundary mode for resampling.
* **cval** (`float`, default: `0.0`): Fill value for constant-mode resampling.
* **output_dtype** (`numpy.dtype`, optional): Optional output dtype cast before writing.
* **export_region** (`str`, default: `"full"`): Export scope: `"full"` for full-grid output, or `"bbox"`/`"roi"` for `best_bbox` ROI output.
* **write_metadata_json** (`bool`, default: `False`): Whether to write `<output_path>.json` with shape/spacing/export metadata.

**Returns**

* `pathlib.Path`: Path to the written TIFF file.

---

## Transform loaders (3D JSON / JSONL)

For transform persistence and replay, prefer canonical records described in [Exports → Canonical transform schema](../exports.md#5-canonical-transform-schema-2d-and-3d).

**Behavior summary**

* `load_nucleisky_transform_3d(path.json)`:
  * loads a single JSON record,
  * normalizes supported legacy aliases (`best_scale`, `best_R`, `best_t`, `best_bbox`, and supported voxel-size aliases),
  * validates canonical structure (`scale`, `R_zyx`, `t_um_zyx`, `A_px`, `b_px`, `pixel_size_*_um_zyx`).
* `load_transforms_any_3d(path)`:
  * accepts `.json` or `.jsonl`;
  * for `.jsonl`, parses each non-empty line, normalizes aliases, and validates each record;
  * malformed JSONL lines or invalid records raise `ValueError`.

**Malformed-line failure note**

Malformed JSONL lines may surface directly as decoder exceptions (typically `json.JSONDecodeError`, a `ValueError` subclass). Unlike 2D strict JSONL mode, 3D loaders do not currently guarantee wrapped path/line-aware diagnostics for malformed JSON parse failures.

* Failure classes are distinct: malformed JSON text fails during decode (e.g., `json.JSONDecodeError`), while structurally valid JSON that violates transform schema fails during semantic validation (validator-raised `ValueError`).

---

### `append_transform_jsonl` (3D)

**Signature**

```python
def append_transform_jsonl(record: dict, out_jsonl: str | Path):
```

**Description**
Appends one 3D transform record per line to a JSONL history file. Parent directories are created automatically if needed.

**Arguments**

* **record** (`dict`): Transform record to append. Recommended canonical fields are listed in [Exports → Canonical transform schema](../exports.md#5-canonical-transform-schema-2d-and-3d).
* **out_jsonl** (`str | pathlib.Path`): Destination `.jsonl` file path.

**Behavior notes**

* This helper appends serialized JSON objects and does not validate record schema before writing.
* Validation and normalization are applied when records are loaded via `load_nucleisky_transform_3d` / `load_transforms_any_3d`.
