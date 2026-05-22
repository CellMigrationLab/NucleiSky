# Limitations & Failure Modes

[:arrow_left: Documentation index](index.md)

This page consolidates limitations for both **2D** (`nucleisky2d`) and **3D** (`nucleisky3d`) workflows.

## Shared Assumptions (2D + 3D)

NucleiSky algorithms are strictly bound to finding a **similarity transform**. This means the mathematical model is restricted to exactly:

* **1 global uniform scale factor** (`best_scale: float`).
* **1 rigid rotation matrix** (`best_R`).
* **1 translation vector** (`best_t`).

It is intentionally **not** a general-purpose deformable registration framework. It does not model affine shear, non-uniform scaling (e.g., stretching the X axis but not Y), or local elastic warping. When tissue undergoes non-similarity deformations (like localized tearing or unequal swelling), the alignment will attempt to find the best "rigid + scale" compromise, which may result in local misalignments at the edges of the tissue.

---

## 2D Limitations and Failure Modes

### Quick Sanity Checklist

1. You are working in 2D centroid coordinates `(y, x)` (typically in microns).
2. The input array is strictly 2D. *Developer Note:* If RGB/RGBA arrays `(Y, X, 3/4)` are passed, they are flattened to grayscale. Z-stacks or time series will raise a strict validation error.
3. Pixel sizes (µm/px) are correct for full image and crop.
4. Crop and full image truly overlap.
5. Segmentation quality is consistent across both images.

### Typical Failure Modes

1. **No overlap between crop and full image**
* **Symptoms:** Repeated failures or visually wrong but mathematically "plausible" matches.
* **Try:** Verify file pairing/provenance and confirm shared structures.


2. **Sparse constellations (too few nuclei)**
* **Symptoms:** Unstable or failed registration. While algorithms theoretically need only 3 points (triangles) or 4 points (quads), practical biology requires at least ~10 points to overcome spatial ambiguity.
* **Try:** Increase crop area or improve segmentation sensitivity.


3. **Repetitive spatial patterns (e.g., muscle fibers, highly uniform arrays)**
* **Symptoms:** High inlier score but wrong absolute location (phase error).
* **Try:** Tighten `angle_max_deg` bounds and use feature-informed matching (`graph` matcher) when available.


4. **Poor/inconsistent segmentation**
* **Symptoms:** Missing/extra nuclei (split/merged cells) cause the geometric point clouds to diverge, resulting in local-only agreement.
* **Try:** Retune segmentation scaling or provide curated label masks via the BYOM workflow.


5. **Incorrect pixel size or mixed magnification issues**
* **Symptoms:** Globally wrong scale or drifting alignment across the image.
* **Try:** Validate microscope acquisition metadata and check your `pixel_size_um` values.



---

## 3D-Specific Limitations

1. **Absolute Minimum Nuclei Count**
* Unlike 2D, 3D spatial geometry requires more points to lock in a coordinate frame. Both the `pyramid` and `hashing3d` matchers strictly require a minimum of **4 non-degenerate nuclei** to compute a 3D hypothesis (forming a tetrahedron or a local 3D basis frame). If fewer than 4 points are found, the matchers will instantly return a failure.


2. **Severe Z-Anisotropy and Biological Shrinkage**
* Microscopy stacks often have lower Z-resolution than XY. The pipeline specifically checks metadata and emits a warning if in-plane XY anisotropy exceeds 1% (`0.01`), or if Z-anisotropy exceeds 300% (`3.0`) relative to the mean XY spacing.
* **Crucial limitation:** Because `estimate_similarity_3d(...)` applies a *single, uniform* 3D scale factor, the algorithm **cannot correct for anisotropic biological tissue shrinkage** (e.g., if tissue shrank by 30% in the Z-axis during optical clearing but remained stable in XY).


3. **Memory Overload on Volumetric Export**
* Warping full grid `(Z, Y, X)` volumes requires significant RAM. If you request a full-grid export of a massive array, it may trigger an Out-Of-Memory (OOM) kill.
* **Try:** Use `export_region="bbox"` to export only the localized region of interest, or rely entirely on the lightweight `.json` transform to warp individual points/spots.



For more 3D workflow context, see:

* [3D Data Preparation](3D/data_preparation.md)
* [3D Workflow](3D/workflow.md)

---

## Practical QC Rule

Trust overlays and visual QC outputs, not just scalar metrics. In repetitive tissues and challenging modalities, manual inspection of the `_overlay.png` (2D) or the `|diff|` error projections (3D) remains essential.
