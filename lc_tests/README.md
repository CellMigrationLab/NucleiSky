# Tests for the tools exposed to Napari / Fiji

Declarations: `src/nucleisky_lc_tools/__init__.py`. They run in NucleiSky's own environment through the LabConstrictor tools worker.

    python lc_tests/make_fixtures.py lc_tests/fixtures      # synthetic nuclei field + a rotated/scaled crop with known transform (needs numpy, scipy, tifffile)
    labconstrictor-tools check --module nucleisky_lc_tools
    labconstrictor-tools test  --module nucleisky_lc_tools --cases lc_tests/cases.json

`make_fixtures.py` also writes `reference_mask.tif` / `query_mask.tif` (simple thresholded label images) for the optional mask inputs. `cases.json` recovers the known transform (12 degrees, 2x) and checks that invalid calibrations are rejected with a readable error.
The first run is slow (numba compiles): allow a few minutes. The 20 cases also cover the matcher choice, other threshold settings, existing masks (wrong-size masks are refused), a fixed seed with a time limit, and an invalid pixel size.
When you change `src/nucleisky_lc_tools`, pass `--pythonpath src` so that your working copy is tested and not the installed one.
