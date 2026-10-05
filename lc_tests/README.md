# Tests for the tools exposed to Napari / Fiji

Declarations: `src/nucleisky_lc_tools/__init__.py`. They run in NucleiSky's own environment through the LabConstrictor tools worker.

    python lc_tests/make_fixtures.py lc_tests/fixtures      # synthetic nuclei field + a rotated/scaled crop with known transform (needs numpy, scipy, tifffile)
    labconstrictor-tools check --module nucleisky_lc_tools
    labconstrictor-tools test  --module nucleisky_lc_tools --cases lc_tests/cases.json

`cases.json` recovers the known transform (12 degrees, 2x) and checks that invalid calibrations are rejected with a readable error.
The first run is slow (numba compiles): allow a few minutes.
