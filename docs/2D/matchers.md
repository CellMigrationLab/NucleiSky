# 2D Matchers

[:arrow_left: Documentation index](../index.md)

> **Scope:** This page is for **2D image registration** workflows.
> Looking for the volumetric workflow? See the **3D equivalent**: [3D matchers](../3D/matchers.md).

## Introduction

NucleiSky2D aligns a cropped field of view (ROI / “crop”) to a larger reference image by estimating a **similarity transform** (scale, rotation, translation) that maximises geometric consistency between **nuclear centroids**.

Each matcher uses a different geometric abstraction — **graphs**, **triangles**, **quads**, or **geometric hashing** — to propose candidate correspondences and score them with a robust inlier check. There is no single universal “best” matcher; the idea is to have a small toolbox with complementary strengths and failure modes.

---

## How the matchers work

All matchers follow the same recipe — they mainly differ in *how* they propose candidate matches:

1. **Build local geometry descriptors** for each nucleus (triangles, quads, graph neighbourhoods, or hash entries).
2. **Propose correspondences** between the crop and the full set using those descriptors.
3. **Estimate a transform** from a small subset of matches.
4. **Score the hypothesis** by counting how many crop nuclei agree with it (the *inliers*).
5. **Accept the best hypothesis** if it explains enough of the crop, then optionally refine.

Key terms used throughout this page:

* **Similarity transform**: rotation + **uniform** scaling + translation. (No shearing, no non-uniform scaling.)
* **Inlier**: a crop point that lands within `inlier_radius_um` of its nearest compatible full-image neighbour after applying the transform.
* **RANSAC**: repeatedly sample small candidate correspondence sets, fit a transform, and keep the transform with the most inliers.
* **ICP**: refinement step that “nudges” the transform to better align the inliers once a good initialisation exists.

Practical note: NucleiSky2D works in **2D** and expects centroid arrays in `(y, x)` order, preferably in calibrated physical coordinates (µm).

---

## Adaptive strategy

`run_adaptive_matching_and_export` automatically chooses a matcher order based on the number of nuclei in the crop (`n_crop`):

* **`n_crop < 20`** → `quad` → `triangles` → `graph` → `hashing`
* **`20 ≤ n_crop < 1000`** → `triangles` → `graph` → `quad` → `hashing`
* **`n_crop ≥ 1000`** → `hashing` → `triangles` → `graph` → `quad`

The adaptive loop stops at the first **successful** match.

Important: the **graph** matcher only runs if feature vectors are available (for example, `features_crop` and `features_full` are provided, or `df_crop`/`df_full` contain `feature_vector` columns). If they are missing, adaptive mode will silently skip the graph matcher and continue with the remaining options.

---

## Matcher details

### Configuration overview

Matcher settings are defined in a nested configuration dictionary:

* `_common`: shared defaults applied to all matchers
* `graph`, `triangles`, `quad`, `hashing`: matcher-specific defaults

You can override any setting by passing `matcher_config` (structured overrides) and/or `matcher_kwargs` (runtime overrides) when calling `NucleiSky(...)`.

---

### Common parameters (`_common`)

These parameters are shared across matchers. The inlier radius is the spatial tolerance for declaring that a transformed crop centroid agrees with the reference constellation.

| Parameter | Default | Description |
| --- | --- | --- |
| `inlier_radius_um` | `2.0` | Inlier tolerance (µm). Increase if detections are noisier or if you expect small geometric distortions. |
| `scale_min` | `0.5` | Minimum allowed scale factor for the similarity transform. |
| `scale_max` | `2.0` | Maximum allowed scale factor for the similarity transform. |
| `angle_max_deg` | `None` | Rotation bound (degrees). `None` means unrestricted. |
| `random_state` | `42` | Random seed for sampling-based steps (reproducibility). |
| `use_dynamic_scale` | `True` | If `True`, the pipeline may narrow scale bounds using local spacing estimates (when the required inputs are available). |
| `dynamic_rel_tol` | `0.2` | Relative tolerance when estimating scale bounds from local spacing. |
| `use_icp_refinement` | `True` | If `True`, refine the best hypothesis with ICP. |
| `margin_um` | `5.0` | Padding margin (µm) used when building aligned bounding boxes/export regions. |
| `frac_inliers_thresh` | `0.6` | Minimum fraction of crop nuclei that must be inliers for the final match to be accepted. |

---

### What counts as a “successful” match?

A matcher proposes a similarity transform for the crop (`best_scale`, `best_R`, `best_t`). The match is considered **successful** if:

* a valid transform was produced, and
* after applying it, the **fraction of crop nuclei** whose nearest-neighbor distance to the full constellation is `<= inlier_radius_um` is at least `frac_inliers_thresh`.

In addition to `frac_inliers`, the pipeline also reports quality metrics, including the number of inliers and the mean inlier error (µm), which are useful for QC and hypothesis comparison.

---

### Minimum inliers (how `min_inliers_*` is actually used)

Several matchers expose:

* `min_inliers_abs` (absolute floor), and
* `min_inliers_frac` (fraction of `n_crop`)

Internally, the pipeline computes a single `min_inliers` value that is stable across small/large crops:

* `min_inliers = max(min_inliers_abs, int(min_inliers_frac * n_crop))`
* It explicitly enforces that `min_inliers` is never less than 3 (the absolute minimum points required to compute an affine transform).
* Then it is clipped so it never exceeds a “cap fraction” of the crop size (typically 80%).

This provides a reasonable inlier requirement for both sparse and dense constellations without requiring hand-tuning per dataset size.

---

## Matchers

### Graph matcher

**Description**
The graph matcher builds geometric kNN graphs in both crop and full constellations, derives rotation-robust node descriptors, and optionally fuses those graph features with provided nucleus features (for example morphology). The graph descriptor stores nearest-neighbor distances normalized by each node's local median neighbor distance, rotation-normalized angle terms (`cos Δ`, `sin Δ`), the local/global median-distance ratio, and degree information. It proposes feature-consistent correspondences, estimates a similarity transform via sampling, and can refine the result with ICP.

**When it shines**

* You have informative per-nucleus features (morphology/shape) and want extra discrimination in repetitive spatial patterns.
* Medium-sized constellations where feature-guided matching is feasible.

**Watch-outs**

* Requires feature vectors (adaptive mode skips it if features are not provided).
* If features are noisy or inconsistent across modalities, geometry-only matchers may be more stable.

**Configuration (`graph`)**

| Parameter | Default | Description |
| --- | --- | --- |
| `k_nn_graph` | `8` | Spatial neighbours used to build the kNN graph. |
| `k_ngh_feat` | `5` | Neighbors used inside each node descriptor. |
| `standardize` | `True` | Z-score features before weighting. |
| `w_shape` | `0.4` | Weight for nucleus shape features in the combined descriptor. |
| `w_graph` | `0.8` | Weight for graph-structure features. |
| `w_triangles` | `0.3` | Weight for triangle-based local geometry features. |
| `n_triangles` | `10` | Local triangles sampled per nucleus for triangle features. |
| `n_feat_neighbors` | `20` | Nearest neighbours in feature space considered for correspondences. |
| `n_iters` | `50000` | Sampling iterations. |
| `min_inliers_abs` | `5` | Minimum absolute inliers required (pre-quality gate). |
| `min_inliers_frac` | `0.12` | Minimum inlier fraction required (pre-quality gate). |
| `min_triangle_area_um2` | `1e-6` | Reject degenerate triangles below this area. |
| `enforce_unique_full_matches` | `True` | Enforce one-to-one mapping on the full side during candidate selection. |
| `feat_ratio` | `0.85` | Lowe-style ratio test for ambiguous feature matches (lower = stricter). |
| `feat_max_dist` | `None` | Optional absolute cap in feature distance (`None` disables). |
| `require_mutual` | `False` | Require mutual nearest-neighbor feature matches. |
| `k_spatial` | `4` | Spatial neighbour count for geometric gating of feature matches. |
| `require_feat_consistency` | `False` | Drop candidates inconsistent with feature-space geometry. |
| `prosac` | `True` | Bias sampling toward higher-quality feature matches early. |
| `pretest_n` | `20` | Number of points used in a quick pretest before full scoring. |
| `refit_on_inliers` | `True` | Refit the transform using all inliers once a good hypothesis is found. |
| `min_inlier_radius_frac_nn` | `0.2` | Floor for inlier radius as a fraction of local neighbour spacing. |
| `max_candidate_pairs` | `200000` | Cap on candidate pairs to limit runtime. |
| `n_candidates_per_patch` | `10` | Max candidates retained per crop point. |
| `n_candidates_per_full` | `10` | Max candidates retained per full point. |
| `pretest_relax` | `0.6` | Relaxation factor for the pretest step. |
| `soft_fail_return_best` | `True` | Return the best hypothesis even if strict gates are missed. |
| `min_inliers_cap_frac` | `0.80` | Cap fraction used when computing stable inlier requirements. |

---

### Triangle matcher

**Description**
The triangle matcher builds local triangle descriptors from kNN neighbourhoods, matches crop triangles to full-image triangles in feature space, and estimates a similarity transform via sampling. Each triangle descriptor is the two-component `(v_b, v_h)` representation computed from two edge vectors: `v_b` is the normalized projection of one edge onto the other, and signed `v_h` is the normalized perpendicular height. It includes guards against degenerate triangles and can refine with ICP.

**When it shines**

* Medium-sized constellations where local geometry is stable.
* You want a geometry-only matcher that is often robust without requiring per-nucleus feature vectors.

**Watch-outs**

* Sensitive to degeneracy in sparse or strongly anisotropic point sets (many nearly-collinear triangles).

**Configuration (`triangles`)**

| Parameter | Default | Description |
| --- | --- | --- |
| `n_triangles` | `5` | Triangle features sampled per nucleus. |
| `n_iters` | `50000` | Sampling iterations. |
| `min_inliers_abs` | `5` | Minimum absolute inliers required (pre-quality gate). |
| `min_inliers_frac` | `0.12` | Minimum inlier fraction required (pre-quality gate). |
| `angle_max_deg` | `None` | Maximum allowed absolute rotation (degrees). `None` means unrestricted. |
| `k_nn_tri` | `8` | kNN size used to form local triangle descriptors. |
| `n_feat_neighbors` | `1` | Nearest neighbors in triangle-feature space per query. |
| `max_candidate_pairs` | `None` | Cap on candidate feature pairs (`None` disables). |
| `early_stop_frac` | `1.0` | Early stop when this fraction of crop points are inliers. |
| `early_stop_inliers` | `None` | Explicit early-stop inlier count (overrides `early_stop_frac` when set). |
| `min_triangle_area_um2` | `1e-6` | Minimum triangle area to avoid degenerate hypotheses. |
| `use_scale_aware_area_floor` | `True` | Raise area floor based on local spacing to reduce scale bias. |
| `area_floor_alpha` | `0.02` | Multiplier for spacing when computing the scale-aware area floor. |

---

### Quad matcher

**Description**
The quad matcher forms local center-plus-neighbor quads: for each center point, it samples triplets from nearby points. The descriptor sorts the three neighbors by distance from the center, normalizes coordinates by the farthest neighbor distance, aligns the farthest neighbor to a canonical axis, stores normalized distance ratios, aligned coordinates for the two nearer neighbors, and three internal angles. It proposes correspondences via descriptor matching and estimates a similarity transform via sampling. It can optionally test 3-of-4 subsets (“triplet hypotheses”) to tolerate one bad point inside a quad, then refine with ICP.

**When it shines**

* Small constellations where higher-order geometry provides extra constraints.
* Situations where triangles are too ambiguous, but you still have enough points (≥ 4).

**Watch-outs**

* Needs enough points and stable local neighbourhoods; missing neighbours can reduce reliability.

**Configuration (`quad`)**

| Parameter | Default | Description |
| --- | --- | --- |
| `k_nn_quad` | `40` | kNN size used to generate candidate quads. |
| `n_desc_neighbors` | `7` | Descriptor-space neighbors used to propose quad matches. |
| `n_iters` | `50000` | Sampling iterations. |
| `min_inliers_abs` | `5` | Minimum absolute inliers required (pre-quality gate). |
| `min_inliers_frac` | `0.12` | Minimum inlier fraction required (pre-quality gate). |
| `angle_max_deg` | `None` | Maximum allowed absolute rotation (degrees). `None` means unrestricted. |
| `k_candidates` | `8` | Neighbors per center used to form quads. |
| `n_quads_per_center` | `14` | Max quad samples per center nucleus. |
| `min_area2` | `1e-6` | Minimum normalised internal area to reject near-collinear quads. |
| `max_candidate_pairs` | `30000` | Cap on candidate quad pairs. |
| `use_triplet_hypotheses` | `True` | Test 3-of-4 subsets to tolerate one incorrect point. |
| `early_stop_frac` | `1.0` | Early stop when this fraction of crop points are inliers. |
| `early_stop_inliers` | `None` | Explicit early-stop inlier count (overrides `early_stop_frac` when set). |

---

### Geometric hashing matcher

**Description**
The hashing matcher builds a geometric hash table on the full constellation using anchor pairs `(i, j)` whose baseline is near `base_distance_um`. For each third point `k`, it projects the vector from `i` to `k` into the local 2D frame defined by `i→j`, stores normalized radius `r_norm = ||rel|| / ||i-j||` and angle `theta`, and bins them with `bin_size_r` and `angle_bin_deg`. It samples analogous crop triplets, looks up compatible bins (including nearby bins to reduce quantisation sensitivity), proposes transforms, and scores them by inlier count. Optional pretesting and early stopping are used to keep runtime manageable on large constellations.

**When it shines**

* Very large constellations where higher-order matching becomes expensive.
* Cases with missing points where a more global, vote-based method can still lock on.

**Watch-outs**

* Sensitive to hashing resolution and `base_distance_um` relative to typical inter-nucleus spacing.

**Configuration (`hashing`)**

| Parameter | Default | Description |
| --- | --- | --- |
| `base_distance_um` | `10.0` | Base distance that sets the scale of the hash grid. |
| `bin_size_r` | `0.1` | Radial bin size for normalized distances (smaller = more precise, potentially slower). |
| `angle_bin_deg` | `10` | Angular bin width (degrees). |
| `vote_thresh` | `3` | Minimum votes for a bin to be considered a viable candidate. |
| `n_iters` | `50000` | Sampling iterations. |
| `min_inliers_abs` | `5` | Minimum absolute inliers required (pre-quality gate). |
| `min_inliers_frac` | `0.12` | Minimum inlier fraction required (pre-quality gate). |
| `angle_max_deg` | `None` | Maximum allowed absolute rotation (degrees). `None` means unrestricted. |
| `max_neighbors_full` | `40` | Max neighbours per full-image anchor when building the hash. |
| `max_pairs_per_anchor` | `30` | Max anchor-to-neighbor pairs per full anchor. |
| `max_k_per_pair` | `20` | Max third points per anchor-pair to populate hash buckets. |
| `max_candidates_per_bin` | `200` | Cap on entries stored per hash bin. |
| `max_neighbors_patch` | `40` | Max neighbours per crop anchor when sampling triplets. |
| `max_pairs_per_anchor_patch` | `30` | Max anchor-to-neighbor pairs per crop anchor. |
| `max_k_per_pair_patch` | `20` | Max third points per crop anchor-pair. |
| `neighbor_bin_radius` | `1` | Number of neighboring bins searched (±) to reduce quantization artifacts. |
| `max_candidates_test` | `120` | Max hash candidates tested per iteration after ranking. |
| `randomize_candidates` | `False` | If `True`, randomise candidate testing rather than strict ranking. |
| `pretest_n` | `80` | Points used for a fast pretest before full scoring. |
| `early_stop_frac` | `1.0` | Early stop when this fraction of crop points are inliers. |

---

### A quick “which matcher should I try?” guide

* Start with **adaptive** (`run_adaptive_matching_and_export`) unless you have a specific reason not to.
* If you have good per-nucleus features and want extra discrimination: try **graph** (or ensure adaptive has features so it can use graph).
* If you want a strong geometry-only default for moderate sizes: **triangles**.
* If the crop is very small: **quad** can be surprisingly effective.
* If the crop is very large (many thousands of nuclei): **hashing** is designed to scale.
