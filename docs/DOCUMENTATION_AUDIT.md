# NucleiSky documentation audit

Audit date: 2026-06-08

## Scope and files inspected

This audit reviewed the public documentation entry points, package metadata, examples, and notebook inventory for release and manuscript readiness.

### Documentation and metadata

* `README.md`
* `docs/index.md`
* `docs/installation.md`
* `docs/segmentation.md`
* `docs/qc.md`
* `docs/exports.md`
* `docs/troubleshooting.md`
* `docs/limitations.md`
* `docs/2D/data_preparation.md`
* `docs/2D/workflow.md`
* `docs/2D/matchers.md`
* `docs/2D/api.md`
* `docs/3D/data_preparation.md`
* `docs/3D/workflow.md`
* `docs/3D/matchers.md`
* `docs/3D/api.md`
* `notebooks/README.md`
* `pyproject.toml`, `setup.py`, `requirements.txt`, `environment.yaml`
* `.github/CONTRIBUTING.md` and issue templates
* `LICENSE`

### Notebooks reviewed at inventory/static-text level

* `notebooks/NucleiSky2DApp/NucleiSky2DApp.ipynb`
* `notebooks/NucleiSky3DApp/NucleiSky3DApp.ipynb`
* `notebooks/NucleiSky2D_API_Workflow_Example/NucleiSky2D_API_Workflow_Example.ipynb`
* `notebooks/NucleiSky3D_API_Workflow_Example/NucleiSky3D_API_Workflow_Example.ipynb`
* `notebooks/NucleiSky2DBenchmarking/NucleiSky2DBenchmarking.ipynb`
* `notebooks/NucleiSky3DBenchmarking/NucleiSky3DBenchmarking.ipynb`

### Source files checked against documentation

* `src/nucleisky/__init__.py`
* `src/nucleisky/nucleisky2d/__init__.py`
* `src/nucleisky/nucleisky2d/config.py`
* `src/nucleisky/nucleisky2d/pipeline.py`
* `src/nucleisky/nucleisky2d/matching/graph.py`
* `src/nucleisky/nucleisky2d/matching/geometry.py`
* `src/nucleisky/nucleisky2d/matching/triangle.py`
* `src/nucleisky/nucleisky2d/matching/quad.py`
* `src/nucleisky/nucleisky2d/matching/hashing.py`
* `src/nucleisky/nucleisky3d/__init__.py`
* `src/nucleisky/nucleisky3d/config.py`
* `src/nucleisky/nucleisky3d/pipeline.py`
* `src/nucleisky/nucleisky3d/features.py`
* `src/nucleisky/nucleisky3d/io.py`
* `src/nucleisky/nucleisky3d/segmentation.py`
* `src/nucleisky/nucleisky3d/matching/pyramid.py`
* `src/nucleisky/nucleisky3d/matching/hashing3d.py`

## Main documentation entry points identified

* `README.md` is the main project landing page and citation/quick-start entry point.
* `docs/index.md` is the documentation map.
* `docs/installation.md` is the install and environment reference.
* `docs/2D/workflow.md` and `docs/3D/workflow.md` are the primary user workflows.
* `docs/2D/api.md` and `docs/3D/api.md` are the public API references.
* `docs/2D/matchers.md` and `docs/3D/matchers.md` explain algorithm behavior and configuration.
* `docs/segmentation.md` centralizes thresholding, Cellpose, Cellpose-SAM, InstanSeg, and 2.5D segmentation notes.
* `notebooks/README.md` now classifies notebooks as apps, API examples, or benchmark/manuscript workflows.

## Issues fixed in this audit

### Installation and package metadata

* Reconciled stale `setup.py` metadata with `pyproject.toml` by reducing `setup.py` to a minimal compatibility shim, preventing legacy metadata from advertising version `0.0.1`, Python `>=3.12.13`, and no dependencies.
* Updated the package description in `pyproject.toml` so it reflects both 2D and 3D functionality.
* Added project URLs to `pyproject.toml`.
* Added a `notebooks` optional dependency group for interactive and benchmark notebooks, and included those dependencies in the `all` extra.
* Clarified in `docs/installation.md` that package metadata supports Python 3.10+, while `environment.yaml` pins Python 3.12.13 for reproducible app/notebook builds.
* Clarified that GPU use is optional and limited to deep-learning segmentation backends, not core matching.

### Imports and public API

* Added compatibility packages `nucleisky2d` and `nucleisky3d` so documented imports such as `from nucleisky2d.pipeline import NucleiSky` and `from nucleisky3d.pipeline import NucleiSky3D` resolve after installation.
* Documented the public import paths in `docs/installation.md` and noted that `nucleisky.nucleisky2d` / `nucleisky.nucleisky3d` are longer backwards-compatible paths used by older notebooks.
* Added 3D API documentation for `centroids_from_df_3d`, `segment_nuclei_2p5d`, and key volume I/O helpers (`inspect_volume_header`, `load_volume`, `require_voxel_size_um_zyx`, `save_tiff_zyx`).

### README and user-facing claims

* Fixed broken Colab badge links for the 2D and 3D apps; the notebooks live inside per-notebook subdirectories.
* Reworded over-strong claims such as “scale, rotation, and origin don’t matter” and “perfectly recovering” to emphasize calibrated similarity-transform estimation and QC.
* Clarified that 3D is supported but should be treated as a proof-of-concept / manually QC-reviewed volumetric extension unless stronger validation is added.
* Fixed the BibTeX `howpublished` URL formatting and added a note that manuscript-specific citation/DOI details remain to be added.

### Matcher documentation

* Corrected the 2D adaptive matcher order for `n_crop >= 1000` to match the code: `hashing -> triangles -> graph -> quad`.
* Expanded 2D graph matcher documentation to describe local median distance normalization, rotation-normalized angle terms, local/global distance ratio, and degree terms.
* Expanded 2D triangle matcher documentation to explain the exposed `(v_b, v_h)` descriptor terms.
* Expanded 2D quad matcher documentation to describe center-plus-neighbor construction, normalized sorted-neighbor distances, canonical alignment, aligned-coordinate terms, and internal-angle terms.
* Expanded 2D geometric hashing documentation to describe anchor-pair baselines, normalized radius/angle variables, and binning.
* Expanded 3D pyramid matcher documentation to describe the 7-value tetrahedron descriptor: normalized volume term plus six sorted normalized edge lengths.
* Expanded 3D geometric hashing documentation to describe 3-anchor local frames and normalized `(x, y, z)` bins for the fourth landmark.
* Clarified that the public 3D matcher name is `hashing`, while the defaults still use the legacy `hashing3d` configuration section.

### Notebooks and examples

* Replaced the terse notebook index with a notebook-purpose table that distinguishes interactive apps, API examples, and manuscript/benchmark notebooks.
* Added notebook environment guidance for per-notebook `requirements.yaml`, merged `requirements.txt`, and package extras.
* Added benchmark reproducibility notes covering input calibration, segmentation settings, matcher seeds/configuration, checkpoint/reload outputs, success definitions, and plotting outputs.

## Unresolved issues requiring human input

### Citation and release metadata

* Add a real `CITATION.cff` before public release once the final manuscript title, author list, DOI/preprint DOI, and preferred citation are known.
* Decide whether the placeholder README BibTeX should cite a GitHub release, Zenodo archive, preprint, or final journal article.
* Add release notes or a `CHANGELOG.md` at the repository root. Per-notebook changelogs exist, but there is no package-level release history.

### Manuscript/data reproducibility

* Add a manuscript reproduction guide that maps each manuscript figure/panel to a notebook, input dataset, expected output files, and exact commit/release tag.
* Replace any remaining private dataset assumptions in benchmark notebooks with public dataset identifiers, accession numbers, DOI links, or explicit “bring your own data” placeholders.
* Define final success metrics for manuscript figures: matcher-reported inlier success, image-level SSIM/overlay success, or both.
* Confirm whether 3D benchmarks should be described as translation/subvolume-localisation benchmarks only, or whether arbitrary 3D rotation benchmarks will be added.

### Notebook technical debt

* Several notebooks still use the longer `nucleisky.nucleisky2d` / `nucleisky.nucleisky3d` import paths. These now work via the implementation package, but public examples should eventually be normalized to `nucleisky2d` / `nucleisky3d` for consistency.
* The 3D app and 3D benchmark notebooks contain legacy UI strings and aliases using `hashing3d`; the code supports this alias, but public-facing labels should be standardized to `hashing` with a note about legacy config names.
* Full notebook execution was not performed during this audit. Static review found placeholders and reload/checkpoint logic, but runtime validation with representative public data remains required.

### Optional dependencies and GPU behavior

* Verify the final optional-dependency matrix on a clean CPU-only environment and, separately, a GPU environment for Cellpose/InstanSeg.
* Confirm whether `cellpose[all]` is the desired published extra for all supported platforms; it can be heavy and may alter torch/CUDA environments.
* Confirm whether `jl-hidecode` should remain only in `requirements.txt` or be included in the `notebooks` extra. It was not added to the package extra because it appears to be a notebook-publishing helper rather than a runtime requirement.

### Smart microscopy / NIS-Elements and live-to-fixed workflows

* No dedicated public documentation was found for NIS-Elements JOBS smart microscopy re-targeting or live-to-fixed / synthetic nuclear labelling workflows. Add a short workflow page if these are manuscript-facing features, including required inputs, exported transform handoff format, coordinate units, and microscope-specific caveats.

## Suspected stale or publication-sensitive notebooks/examples

* `notebooks/NucleiSky2DBenchmarking/NucleiSky2DBenchmarking.ipynb`: publication-facing; requires final public data paths, exact success definitions, and full restart/reload verification.
* `notebooks/NucleiSky3DBenchmarking/NucleiSky3DBenchmarking.ipynb`: publication-facing; currently framed as reference-only 3D subvolume localisation and should not be over-described as arbitrary 3D registration validation.
* `notebooks/NucleiSky3DApp/NucleiSky3DApp.ipynb`: contains `hashing3d` UI labels/aliases; code compatibility is present, but labels should be user-facing `hashing` before release.
* `notebooks/NucleiSky2DApp/NucleiSky2DApp.ipynb` and `notebooks/NucleiSky3DApp/NucleiSky3DApp.ipynb`: likely require end-to-end widget testing in Colab and local Jupyter after package release.

## Missing or incomplete release-readiness items

* `LICENSE` exists.
* `.github/CONTRIBUTING.md` and issue templates exist.
* Package-level installation instructions exist and were updated.
* Minimal quick starts exist in workflow/API docs.
* `CITATION.cff` is missing.
* Root package changelog/release notes are missing.
* A data-availability statement and code-availability statement for the manuscript are missing from repository docs.
* Zenodo/GitHub archive instructions are missing.
* README badges exist, but verify PyPI badge/version after the first public release.

## Recommended final pre-release checklist

1. Create `CITATION.cff` with final author, title, DOI/preprint DOI, repository URL, and version metadata.
2. Add a root `CHANGELOG.md` or GitHub Releases notes for version `0.2.0`.
3. Add `docs/manuscript_reproducibility.md` mapping datasets/notebooks/figures/outputs.
4. Run all notebooks from a clean checkout with public or placeholder data and clear outputs before publication.
5. Validate `pip install .`, `pip install ".[all]"`, and `pip install ".[notebooks]"` in clean environments.
6. Validate import paths: `import nucleisky`, `import nucleisky2d`, `import nucleisky3d`, and representative submodule imports.
7. Run the full pytest suite, including optional-backend tests where dependencies are available.
8. Confirm CPU-only behavior and document GPU-only acceleration strictly as optional segmentation acceleration.
9. Standardize final manuscript terminology: landmark/object for generic point sets, nucleus only when nuclear segmentation is specifically required; query/crop versus full/reference; patch size for 2D and subvolume size for 3D; calibrated pixel/voxel coordinates versus micrometre coordinates.
10. Add smart microscopy / NIS-Elements and live-to-fixed workflow pages if they are included in the manuscript claims.
