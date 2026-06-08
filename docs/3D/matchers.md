# The 3D Matcher Toolbox

[:arrow_left: Documentation index](../index.md)

This page documents the geometric matching algorithms that power NucleiSky3D.

Like its 2D counterpart, NucleiSky3D estimates a **similarity transform** in physical space (micrometers). This means it calculates a uniform scale, a 3D rigid rotation, and a 3D translation (always in `(Z, Y, X)` order) that best aligns the crop constellation to the full reference constellation.

---

## Overview (For the Biologist)

In 2D, NucleiSky offers a range of geometry families (graphs, triangles, quads). In 3D the maths is heavier, so the toolbox is streamlined to two engines:

### 1. The `pyramid` Matcher (Default for most datasets)

**How it works:** Imagine looking at 4 stars close to each other in the sky. If you connect them, they form a 3D pyramid (a tetrahedron). This matcher builds these little pyramids all over your crop and looks for identical pyramids in the full image.
**When to use it:** This is the default. It works well for small-to-medium crops (dozens to hundreds of cells) where the local arrangement of cells is distinct.

### 2. The `hashing` Matcher (For large point clouds)

**How it works:** The matcher builds local anchor frames from 3 reference landmarks, bins the normalized coordinates of a 4th landmark in that frame, and uses the crop to query those bins and vote for candidate 3D similarity transforms.
**When to use it:** Use this for large, dense point clouds (thousands of cells) where finding individual pyramids takes too long, or where segmentation is noisy and many cells are missing.

### The Adaptive Choice

If you use the adaptive pipeline (`run_adaptive_matching_and_export_3d`), it picks for you based on the number of detected nuclei in the crop:

* **`n_crop < 1000` cells** → Tries `pyramid`, falls back to `hashing`.
* **`n_crop >= 1000` cells** → Tries `hashing`, falls back to `pyramid`.

---

## Under the Hood (For the Developer)

Both backends are called through the `NucleiSky3D(...)` orchestrator and share common parameters. They both end with a spatial inlier check (`inlier_radius_um`) and optional ICP (Iterative Closest Point) refinement.

### `pyramid` (Tetrahedral)

**Entry point**: `run_pyramid_based_matching_um`

* **Strategy:** Builds a spatial k-NN graph in both point clouds. For each node, it computes local **tetrahedron descriptors** using combinations of 3 neighbors. Each 7-value descriptor stores a normalized volume term (`volume / mean_edge^3`, scaled internally) plus the 6 sorted edge lengths normalized by their mean edge length.
* **Matching:** It Z-scores these descriptors, drastically filters candidates via a strict **Mutual Nearest Neighbors (MNN)** check in descriptor space, and runs a 4-point RANSAC to estimate the 3D similarity transform.

### `hashing` (Geometric Hashing)

**Entry point**: `run_geometric_hashing_matching_3d_um`

* **Strategy:** Builds a hash table on the full cloud using robust local 4-point geometries. It picks 3 anchors (`i, j, k`) to define a rigid local 3D coordinate frame, and encodes the relative position of a 4th point (`l`) into that frame.
* **Matching:** It quantizes the 4th point's relative coordinates into `(x, y, z)` bins after normalizing by the anchor baseline length. The crop samples analogous tuples, queries the hash bins (with a neighbor-bin search radius to prevent quantization artifacts), and scores hypotheses by maximum inlier count. It includes fast pretest pruning (`pretest_n`) to skip bad hypotheses early.

---

## Advanced Configuration

`NucleiSky3D(...)` supports two layers of configuration, allowing you to fine-tune RANSAC iterations, thresholds, and scale bounds.

1. **`matcher_config`** (Structured dictionary): Deep-merged with the 3D defaults.
2. **`matcher_kwargs`** (Runtime overrides): Applied last, overriding anything else.

### The Config Structure

The configuration is a nested dictionary with a `_common` block (applied to all matchers) and backend-specific blocks (`pyramid`, `hashing3d`). The public matcher name is `"hashing"`; the legacy config section name `"hashing3d"` is still accepted and used by the defaults.

```python
matcher_config = {
    "_common": {
        "angle_max_deg": 15,          # Restrict search to ±15 degrees rotation
        "inlier_radius_um": 1.5,      # Overrides the 3.0µm default
    },
    "pyramid": {
        "n_iters": 200000,            # Boost RANSAC iterations for pyramid
    },
}

```

### Passing Overrides in Python

You can pass these directly to the `NucleiSky3D` function:

```python
from nucleisky3d.pipeline import NucleiSky3D

result = NucleiSky3D(
    centroids_crop_um=centroids_crop_um,
    centroids_full_um=centroids_full_um,
    full_shape_px_zyx=img_full.shape,
    crop_shape_px_zyx=img_crop.shape,
    pixel_size_full_um_zyx=(2.0, 0.5, 0.5),
    pixel_size_crop_um_zyx=(2.0, 0.5, 0.5),
    matcher="pyramid",
    matcher_config={
        "_common": {
            "inlier_radius_um": 2.0,
        },
    },
    # Flat runtime kwargs override the specific active matcher
    matcher_kwargs={
        "random_state": 42,
        "early_stop_frac": 0.9, 
    },
)

```

---

## Key Parameters Reference

### Common Parameters (`_common`)

| Parameter | Default | Description |
| --- | --- | --- |
| `inlier_radius_um` | `3.0` | Spatial tolerance (µm). Increase this if your segmentation is noisy or you expect slight biological deformations. |
| `scale_min` / `scale_max` | `0.8` / `1.2` | Minimum and maximum allowed scale factors. |
| `angle_max_deg` | `None` | Rotation bound. `None` means unrestricted (full 360° search in all 3 planes). |
| `use_dynamic_scale` | `False` | When `True`, narrows scale search bounds dynamically using global spacing estimates before RANSAC. |
| `use_icp_refinement` | `True` | Refines the final hypothesis using Iterative Closest Point. |
| `frac_inliers_thresh` | `0.45` | The success gate. At least 45% of the crop nuclei must find a physical match in the full image. |

### Pyramid Specific Parameters (`pyramid`)

* **`n_iters`** (Default: `150,000`): RANSAC iterations (increase for harder, noisier datasets).
* **`n_tetrahedra`** (Default: `15`): Number of tetrahedra sampled per node for descriptor averaging.
* **`k_nn_tetra`** (Default: `20`): Neighborhood size used to build graph connectivity.
* **`max_candidate_pairs`** (Default: `None`): Cap on descriptor candidate matches before RANSAC (prevents memory blowouts if `None` is overridden).
* **`early_stop_frac`** (Default: `None`): Stops RANSAC early if a hypothesis explains this fraction of the crop.

### Hashing Specific Parameters (`hashing` matcher; `hashing3d` config section)

* **`base_distance_um`** (Default: `10.0`): The reference baseline used to normalize local geometry. *(Crucial to tune: strongly controls collision rate vs discriminability!)*
* **`bin_size_xyz`** (Default: `0.15`): Hash quantization step in normalized coordinates.
* **`vote_thresh`** (Default: `3`): Minimum bucket-support threshold before a candidate transform is actually tested.
* **`n_iters`** (Default: `50,000`): Hypothesis sampling iterations.
* **`max_neighbors_full`** / **`max_neighbors_patch`** (Default: `40`): Caps on how many neighbors are used to build the local reference frames.
