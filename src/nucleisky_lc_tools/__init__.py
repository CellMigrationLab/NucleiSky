"""NucleiSky tools for Napari, Fiji and the command line (LabConstrictor tools bridge, experimental).

This module only *declares* the tools: it imports `labconstrictor_tools` and the standard library at the top, and NucleiSky,
numpy and scipy inside the function. Keep it that way: it is imported to list the tools, and `import nucleisky` alone takes
many seconds. It lives in its own package (not inside `nucleisky`) for the same reason.

    labconstrictor-tools check --module nucleisky_lc_tools
    labconstrictor-tools test  --module nucleisky_lc_tools --cases lc_tests/cases.json
"""

from typing import Annotated, Literal

from labconstrictor_tools import (
    Affine,
    ApplyTo,
    Axes,
    Description,
    Image,
    ImageOut,
    Min,
    Name,
    PixelSizeOf,
    Scalars,
    ToolError,
    Unit,
    check_cancel,
    progress,
    tool,
)


@tool("Relocalize 2D")
def relocalize(
    reference: Annotated[Image, Axes("YX"), Description("Large / full-field image")],
    query: Annotated[Image, Axes("YX"), Description("Crop to locate inside the reference")],
    reference_pixel_size_um: Annotated[float, Unit("um/px"), Min(0), PixelSizeOf("reference")] = 0.65,
    query_pixel_size_um: Annotated[float, Unit("um/px"), Min(0), PixelSizeOf("query")] = 0.325,
    segmentation: Literal["threshold"] = "threshold",
) -> tuple[
    Annotated[Affine, ApplyTo("query", "reference"), Name("alignment")],
    Annotated[ImageOut, Name("query_aligned")],
    Scalars,
]:
    """Find where a rotated / rescaled query image lies in a reference image."""
    import shutil
    import tempfile

    import numpy as np
    from nucleisky.nucleisky2d.features import add_centroids_orig_px_columns, extract_nuclear_features
    from nucleisky.nucleisky2d.io import save_nucleisky_transform
    from nucleisky.nucleisky2d.pipeline import run_adaptive_matching_and_export
    from nucleisky.nucleisky2d.preprocess import scale_normalize_pair_for_segmentation
    from nucleisky.nucleisky2d.segmentation import segment_nuclei_dispatch
    from scipy.ndimage import affine_transform

    if reference.ndim != 2 or query.ndim != 2:
        raise ToolError(
            "bad_input", "2D (YX) images required; got %s and %s" % (reference.shape, query.shape)
        )
    progress(0.05, "segmenting nuclei")
    f_seg, c_seg, ps_f, ps_c, sf, sc, _ = scale_normalize_pair_for_segmentation(
        reference, query, reference_pixel_size_um, query_pixel_size_um
    )
    seg = {"threshold": {"threshold_method": "otsu", "min_object_size": 5, "do_watershed": True}}
    df_f = add_centroids_orig_px_columns(
        extract_nuclear_features(segment_nuclei_dispatch(f_seg, segmentation, ps_f, seg), None, ps_f), sf
    )
    df_c = add_centroids_orig_px_columns(
        extract_nuclear_features(segment_nuclei_dispatch(c_seg, segmentation, ps_c, seg), None, ps_c), sc
    )
    check_cancel()
    progress(0.4, "matching %d query nuclei against %d reference nuclei" % (len(df_c), len(df_f)))
    out = tempfile.mkdtemp(prefix="nucleisky_")
    try:  # NucleiSky writes its exports to a directory; we use a scratch dir and delete it below
        best, _ = run_adaptive_matching_and_export(
            df_full=df_f,
            df_crop=df_c,
            img_full=reference,
            img_crop=query,
            pixel_size_full_um=reference_pixel_size_um,
            pixel_size_crop_um=query_pixel_size_um,
            result_dir=out,
            store_full_out=False,
        )
        rec = save_nucleisky_transform(
            best,
            out + "/transform.json",
            matcher_name=best.get("matcher", "unknown"),
            pixel_size_full_um=reference_pixel_size_um,
            pixel_size_crop_um=query_pixel_size_um,
            require_success=False,
        )
    except Exception as e:
        shutil.rmtree(out, ignore_errors=True)
        raise ToolError(
            "no_match",
            "NucleiSky could not match the images (%s: %s). Check the pixel sizes and that the query overlaps the reference."
            % (type(e).__name__, e),
        ) from e
    shutil.rmtree(out, ignore_errors=True)
    if not rec["success"]:
        raise ToolError("no_match", "No confident match found.")
    A, b = np.array(rec["A_px"]), np.array(rec["b_px"])
    Ainv = np.linalg.inv(A)
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = b
    aligned = affine_transform(
        query.astype(np.float32), Ainv, offset=-Ainv @ b, output_shape=reference.shape, order=1
    )
    progress(0.95, "done")
    return (
        M,
        aligned,
        {
            "rotation_deg": rec["rotation_deg"],
            "scale": rec["scale"],
            "bbox_y0y1x0x1": rec["bbox_full_px_y0y1x0x1"],
            "n_nuclei_reference": len(df_f),
            "n_nuclei_query": len(df_c),
            "matcher": rec["matcher"],
        },
    )
