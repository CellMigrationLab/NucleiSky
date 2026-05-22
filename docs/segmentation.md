# Segmentation (2D + 3D)

[:arrow_left: Documentation index](index.md)

This page unifies segmentation guidance for **NucleiSky2D** and **NucleiSky3D**.

> **Acceleration note:** GPU usage is optional and applies only to deep-learning segmentation backends (for example, Cellpose and InstanSeg). The rest of NucleiSky runs on CPU, and `numba` is used for CPU-side acceleration in performance-critical code paths.

At a high level, segmentation is the step where raw images become labeled nuclei (`0` = background, `1..N` = objects). Those labels are then converted to centroids/features for point-cloud matching.

---

## Quick Chooser

* Use **2D segmentation** if your input is a single plane `(Y, X)`.
* Use **3D segmentation** if your input is a volume `(Z, Y, X)`.
* Already have masks? Use **Bring Your Own Masks (BYOM)** and skip model inference to directly validate and format your label images.

---

## 2D Segmentation

### Input/Output Contract

* **Input:** A single 2D image array, the `pixel_size_um`, the backend `method` (e.g., `"cellpose"`, `"instanseg"`, or `"threshold"`), and an optional `settings` dictionary.
* **Output:** A 2D integer label image of shape `(Y, X)`.

### Main Entry Point

The `segment_nuclei_dispatch` function acts as the unified API for 2D model inference. It routes requests to the corresponding backend using a shared, module-level `Segmentor` instance to safely manage resource caching (e.g., keeping PyTorch models in memory).

#### Function Signature

```python
from nucleisky2d.segmentation import segment_nuclei_dispatch

labels = segment_nuclei_dispatch(
    img, 
    method="instanseg", 
    pixel_size_um=0.65, 
    settings=None, 
    segmentor=None
)

```

**Parameters:**

* **`img`**: The 2D image array to segment.
* **`method`**: String specifying the backend. Options are `"cellpose"`, `"instanseg"`, or `"threshold"`.
* **`pixel_size_um`**: The pixel size in microns, used to automatically scale parameters for deep learning models.
* **`settings`**: An optional dictionary of backend-specific parameters to override the defaults (see below).
* **`segmentor`**: An optional custom `Segmentor` instance. If `None`, the global cached segmentor is used.

### Backend Settings

You can customize the segmentation by passing a nested dictionary to the `settings` parameter.

#### 1. Cellpose (`method="cellpose"`)

*Note: If you use the Cellpose backend in your pipeline, please cite their paper and visit the [Cellpose GitHub](https://github.com/MouseLand/cellpose).*

The `settings["cellpose"]` dictionary accepts the following keys:

* `pretrained_model` (default: `"cpsam"`): The name of the model weights to load.
* `diameter` (default: `None`): Average object diameter in pixels.
* `flow_threshold` (default: `0.4`): Maximum allowed error of the flows.
* `cellprob_threshold` (default: `0.0`): Threshold for the cell probability map.
* `min_size` (default: `15`): Minimum number of pixels per mask.
* `batch_size` (default: `1`): Number of images/tiles to run in a batch.
* `tile_size` (default: `None`): Sets the block size for evaluation. *Developer Note:* Overrides to `256` if a transformer backbone is detected.
* `overlap` (default: `None`): Fractional tile overlap (defaults to 0.1).
* `normalize` (default: `True`): Whether to let Cellpose normalize the image.
* `invert` (default: `False`): Invert image intensities.

#### 2. InstanSeg (`method="instanseg"`)

*Note: If you use the InstanSeg backend in your pipeline, please cite their paper and visit the [InstanSeg GitHub](https://github.com/instanseg/instanseg).*

The `settings["instanseg"]` dictionary accepts the following keys:

* `model_name` (default: `"brightfield_nuclei"`): InstanSeg model to load.
* `target` (default: `"nuclei"`): Specific cellular target.
* `mode` (default: `"auto"`): Evaluation mode. Options are `"auto"`, `"small"`, or `"medium"`.
* `auto_medium_pixels` (default: `6000000`): Threshold (in total pixels) at which `"auto"` mode switches from small to medium memory profiling.
* `robust_normalize` (default: `True`): Applies percentile-based normalization to uint8 before inference.
* `cleanup_fragments` (default: `True`): Removes highly fragmented or likely spurious masks.
* `resolve_cell_and_nucleus` (default: `False`): Attempts to resolve boundaries when tracking both entities.
* `use_mean_threshold` (default: `False`): Enables InstanSeg's mean-threshold mechanism.
* `mean_threshold` (default: `0.3`): Parameter for the threshold mechanism.
* `verbosity` (default: `0`): Console log verbosity level.

#### 3. Classic Thresholding (`method="threshold"`)

The `settings["threshold"]` dictionary accepts the following keys:

* `threshold_method` (default: `"otsu"`): Scikit-image thresholding algorithm. Options are `"otsu"`, `"li"`, `"yen"`, `"triangle"`, `"isodata"`.
* `foreground` (default: `"bright"`): Target intensities. Options are `"bright"` or `"dark"`.
* `channel` (default: `0`): Which channel to extract if the array is multi-channel.
* `gaussian_sigma` (default: `1.0`): Blur radius applied before thresholding.
* `min_object_size` (default: `80`): Area threshold to remove small noise artifacts.
* `min_hole_size` (default: `80`): Area threshold to fill small internal holes.
* `do_watershed` (default: `True`): Toggles whether to run the distance-transform watershed separation.
* `peak_min_distance` (default: `5`): Minimum distance between watershed seed peaks.
* `watershed_compactness` (default: `0.0`): Scikit-image watershed compactness factor.

### Optional Pre-Segmentation Scaling

*Best Practice for Deep Learning:* If your reference and crop images have different pixel sizes, neural networks (like Cellpose or InstanSeg) may struggle because nuclei will appear vastly different in scale. The `scale_normalize_pair_for_segmentation` function rescales both images to a common pixel size (by default, the coarsest resolution) before segmentation.

```python
from nucleisky2d.preprocess import scale_normalize_pair_for_segmentation

(
    img_full_seg,
    img_crop_seg,
    pixel_size_full_seg_um,
    pixel_size_crop_seg_um,
    scale_factor_full,
    scale_factor_crop,
    target_um_per_px_requested,
) = scale_normalize_pair_for_segmentation(
    img_full,
    img_crop,
    pixel_size_full_um,
    pixel_size_crop_um,
    strategy="coarsest",
)

```

### BYOM (Bring Your Own Masks) in 2D

If you have already segmented your images elsewhere (e.g., in QuPath, ImageJ, or a custom script), simply validate them using the `require_2d_label_mask` preprocessor. This ensures the arrays are 2D, strictly non-negative integers, and not empty.

```python
from nucleisky2d.preprocess import require_2d_label_mask

labels_full = require_2d_label_mask(labels_full, label="labels_full", expected_shape=img_full.shape[:2])
labels_crop = require_2d_label_mask(labels_crop, label="labels_crop", expected_shape=img_crop.shape[:2])

```

For end-to-end integration, see [2D Workflow](2D/workflow.md).

---

## 3D Segmentation

### Input/Output Contract

* **Input:** A volume array `(Z, Y, X)`, `pixel_size_um_zyx`, backend `method`, and an optional `settings` dictionary.
* **Output:** A 3D integer label volume `(Z, Y, X)`.

### Main Entry Point (2.5D Slice-and-Stitch)

NucleiSky3D relies on a 2.5D strategy: it runs your chosen 2D segmentation method on every Z-slice independently (using the mean of the Y and X voxel dimensions as the proxy pixel size). It then automatically stitches these slices into coherent 3D volumes using an optimized Intersection-over-Union (IoU) overlap matrix.

```python
from nucleisky3d.segmentation import segment_nuclei_2p5d

labels_zyx = segment_nuclei_2p5d(
    vol_zyx,
    method="threshold",
    pixel_size_um_zyx=(1.0, 0.5, 0.5), # Must be in Z, Y, X order
    settings={"threshold": {"threshold_method": "otsu"}},
    min_iou=0.3, # Overlap threshold required to link objects across slices
)

```

### Optional Pre-Segmentation Scaling

Similar to 2D, rescaling 3D volumes before segmentation improves deep-learning backend consistency. The 3D rescaler respects physical dimensions and performs necessary anti-aliased interpolation.

```python
from nucleisky3d.preprocess import scale_normalize_pair_for_segmentation

(
    vol_full_seg,
    vol_crop_seg,
    voxel_full_seg_um_zyx,
    voxel_crop_seg_um_zyx,
    scale_full_zyx,
    scale_crop_zyx,
    target_um_per_voxel_zyx,
) = scale_normalize_pair_for_segmentation(
    vol_full_zyx,
    vol_crop_zyx,
    voxel_full_um_zyx, # (Z,Y,X)
    voxel_crop_um_zyx, # (Z,Y,X)
    strategy="coarsest",
    max_upsample=4.0,
)

```

### BYOM (Bring Your Own Masks) in 3D

If you already have a fully processed 3D label volume, validate it with the 3D preprocessor:

```python
from nucleisky3d.preprocess import require_3d_label_mask

labels_3d = require_3d_label_mask(
    labels_3d, 
    label="my_3d_mask", 
    expected_shape=vol_zyx.shape
)

```

*Advanced Developer Note:* If you have your own list of 2D segmented arrays (one per slice) and want to utilize NucleiSky's highly optimized sparse-matrix IoU stitcher to build the 3D mask yourself, you can call it directly:

```python
from nucleisky3d.segmentation import stitch_2d_slices

# slice_labels is a list of 2D numpy arrays
labels_3d = stitch_2d_slices(slice_labels=slice_labels, min_iou=0.3, show_progress=True)

```

For end-to-end integration, see [3D Workflow](3D/workflow.md).


### BYOM edge-case checklist (2D + 3D)

- Background label `0` is ignored.
- Labels should be integer-like and non-negative.
- Non-contiguous positive labels are supported.
- Empty labels (no foreground objects) should fail validation or produce low-confidence matching.
- Merged/split nuclei reduce geometric distinctiveness and can degrade registration.
- Tiny-object filtering changes downstream feature counts and may change matcher behavior.
- Always inspect QC overlays and `match_quality` metrics before downstream analysis.

---

## Notes and Troubleshooting

* Segmentation backend packages (`cellpose`, `instanseg`) are optional; install them using extras from [Installation](installation.md).
* If geometric matches fail downstream, inspect your segmentation quality first. Over-segmented (split) or under-segmented (merged) nuclei frequently disrupt point-constellation matching.
