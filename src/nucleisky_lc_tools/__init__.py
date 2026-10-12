"""NucleiSky tools for Napari, Fiji and the command line (LabConstrictor tools bridge, experimental).

This module only *declares* the tools: it imports `labconstrictor_tools` and the standard library at the top, and NucleiSky,
numpy and scipy inside the function. Keep it that way: it is imported to list the tools, and `import nucleisky` alone takes
many seconds. It lives in its own package (not inside `nucleisky`) for the same reason.

    labconstrictor-tools check --module nucleisky_lc_tools
    labconstrictor-tools test  --module nucleisky_lc_tools --cases lc_tests/cases.json
"""

from typing import Annotated, Literal, Optional

from labconstrictor_tools import (
    Advanced,
    Affine,
    ApplyTo,
    Axes,
    ChoicesFrom,
    Description,
    EnabledWhen,
    Group,
    Image,
    ImageOut,
    Label,
    Labels,
    Max,
    Min,
    Name,
    PixelSizeOf,
    PointsOut,
    RegionOf,
    Replace,
    Scalars,
    ShapesOut,
    ToolError,
    Unit,
    Widget,
    check_cancel,
    progress,
    tool,
)


_NO_MATCH = {
    "auto": "No match found: none of the matchers could place the query in the reference. Check the pixel sizes and that the query overlaps the reference.",
    **{
        m: "No match found with the '%s' matcher. This is not an error in the images: try Matcher = auto, or another matcher." % m
        for m in ("quad", "triangles", "graph", "hashing")
    },
}


_THRESHOLD = EnabledWhen("segmentation", "threshold")
_INSTANSEG = EnabledWhen("segmentation", "instanseg")
_DEEP = EnabledWhen("segmentation", "cellpose", "instanseg")
_CELLPOSE = EnabledWhen("segmentation", "cellpose")


# Failures of the segmentation back end that mean "this machine cannot run it" (package missing, weights not downloadable or not
# found, no internet): reported as an installation problem. Anything else is a bug and must surface as a failure with its traceback.
_SEGMENTATION_UNAVAILABLE = (ImportError, OSError)


def _check_images(reference, query, reference_mask, query_mask):
    if reference.ndim != 2 or query.ndim != 2:
        raise ToolError("bad_input", "2D (YX) images required; got %s and %s" % (reference.shape, query.shape))
    for name, mask, image in (("Reference mask", reference_mask, reference), ("Query mask", query_mask, query)):
        if mask is not None and tuple(mask.shape) != tuple(image.shape):
            raise ToolError(
                "mask_shape_mismatch",
                "%s must have the same size as its image: mask %s, image %s" % (name, tuple(mask.shape), tuple(image.shape)),
            )


def _instance_labels(mask, what):
    """A label image NucleiSky can use: finite, whole-numbered, non-negative. A binary mask (only 0 and 1) holds one label for
    every nucleus, so its connected objects are labelled here instead of being counted as a single nucleus."""
    import numpy as np
    from scipy.ndimage import label

    labels = np.asarray(mask)
    if labels.dtype == bool:
        labels = labels.astype(np.uint8)
    if labels.dtype.kind == "f":
        if not np.isfinite(labels).all() or not (labels == np.round(labels)).all():
            raise ToolError("bad_mask", "The %s mask must hold whole-number labels (0 = background); it has fractions or NaN." % what)
    elif labels.dtype.kind not in "iu":
        raise ToolError("bad_mask", "The %s mask must be an integer label image, not %s." % (what, labels.dtype))
    if labels.size and labels.min() < 0:
        raise ToolError("bad_mask", "The %s mask has negative labels." % what)
    labels = labels.astype(np.int64)
    if labels.size and labels.max() <= 1:
        progress(0.04, "the %s mask is binary: labelling its connected objects" % what)
        labels = label(labels > 0)[0].astype(np.int64)
    return labels


def _nuclei_table(what, mask, image_seg, ps_seg, factor, ps_original, segmentation, seg):
    """Feature table of one image: from its mask (at original resolution) or from a segmentation of the rescaled image."""
    import traceback

    from nucleisky.nucleisky2d.features import add_centroids_orig_px_columns, extract_nuclear_features
    from nucleisky.nucleisky2d.segmentation import segment_nuclei_dispatch

    if mask is not None:
        labels, ps, scale = _instance_labels(mask, what), ps_original, 1.0
    else:
        try:
            labels, ps, scale = segment_nuclei_dispatch(image_seg, segmentation, ps_seg, seg), ps_seg, factor
        except _SEGMENTATION_UNAVAILABLE as e:
            traceback.print_exc()  # stderr: the log keeps the cause
            if segmentation in ("cellpose", "instanseg"):
                raise ToolError(
                    "segmentation_unavailable",
                    "%s could not run (%s: %s). Check that its package and model weights are installed (the first run needs internet)."
                    % (segmentation, type(e).__name__, e),
                ) from e
            raise
    frame = add_centroids_orig_px_columns(extract_nuclear_features(labels, None, ps), scale)
    if len(frame) == 0:
        raise ToolError(
            "no_nuclei", "No nuclei found in the %s%s." % (what, " mask" if mask is not None else " (try other segmentation settings)")
        )
    return frame, labels, scale


def _inside_region(what, frame, region, image):
    """Keep the nuclei whose centroid lies inside the region (RegionOf). A region that keeps none is an error that says so."""
    import numpy as np

    from labconstrictor_tools.region import bbox

    bbox(region, image)  # the right size, and not empty
    inside = np.asarray(region) > 0
    height, width = inside.shape
    rows = np.clip(np.rint(frame["centroid_y_px_orig"].to_numpy()).astype(int), 0, height - 1)
    columns = np.clip(np.rint(frame["centroid_x_px_orig"].to_numpy()).astype(int), 0, width - 1)
    kept = frame[inside[rows, columns]].reset_index(drop=True)
    if len(kept) == 0:
        raise ToolError("no_nuclei", "No nuclei of the %s lie inside the selected region: select a region with nuclei, or untick the selection." % what)
    return kept


def _outlines(labels, scale, kept_labels):
    """Outlines of the nuclei that were used, in pixels of the original image (the segmentation ran on a rescaled copy)."""
    from labconstrictor_tools.shapes import labels_to_shapes

    keep = set(int(v) for v in kept_labels)
    collection = labels_to_shapes(labels)
    features = [f for f in collection["features"] if f["properties"]["label"] in keep]
    factor = float(scale) if scale not in (None, 0) else 1.0  # the same mapping as the centroids: original = rescaled / scale

    def rescale(node):
        return [rescale(x) for x in node] if isinstance(node[0], list) else [node[0] / factor, node[1] / factor]

    for feature in features:
        feature["geometry"]["coordinates"] = rescale(feature["geometry"]["coordinates"])
    return {"type": "FeatureCollection", "features": features}


def _segmentation_settings(**options):
    """The settings the segmentation back ends take, from the form (only what was set is passed on: the rest keeps its default)."""
    cellpose = {
        "diameter": options["cellpose_diameter_px"],
        "flow_threshold": options["cellpose_flow_threshold"],
        "cellprob_threshold": options["cellpose_cellprob_threshold"],
        "min_size": options["cellpose_min_size_px"],
        "batch_size": options["cellpose_batch_size"],
        "tile_size": options["cellpose_tile_size_px"],
        "overlap": options["cellpose_tile_overlap"],
        "normalize": options["cellpose_normalize"],
        "invert": options["cellpose_invert"],
    }
    instanseg = {
        "model_name": options["instanseg_model"],
        "target": options["instanseg_target"],
        "cleanup_fragments": options["instanseg_cleanup_fragments"],
        "mode": options["instanseg_mode"],
    }
    if options["instanseg_pixel_size_um"] is not None:
        instanseg["pixel_size_um"] = options["instanseg_pixel_size_um"]
    return {
        "device": options["device"],
        "threshold": {
            "threshold_method": options["threshold_method"],
            "gaussian_sigma": options["blur_sigma"],
            "min_object_size": options["min_area_px"],
            "do_watershed": options["watershed_split"],
            "peak_min_distance": options["peak_distance_px"],
        },
        "cellpose": {k: v for k, v in cellpose.items() if v is not None},
        "instanseg": instanseg,
    }


def _match(df_f, df_c, reference, query, reference_pixel_size_um, query_pixel_size_um, matcher, fixed_seed, max_seconds):
    """Run NucleiSky's adaptive matching. "No match" is NucleiSky's own verdict (success is False); an exception is a failure.

    This calls the matching engine itself (`run_adaptive_nucleisky`), not the wrapper that also exports every attempt to disk:
    that export cannot handle an unsuccessful match and fails with an unrelated TypeError/KeyError, which used to be
    mistaken for "no match" by a catch-all. Nothing here needs those exports.
    """
    import sys
    import tempfile

    from nucleisky.nucleisky2d.features import extract_centroids_um, stack_feature_vectors
    from nucleisky.nucleisky2d.io import save_nucleisky_transform
    from nucleisky.nucleisky2d.pipeline import run_adaptive_nucleisky

    try:
        features_f = stack_feature_vectors(df_f, name="df_full")
        features_c = stack_feature_vectors(df_c, name="df_crop")
    except Exception as e:  # noqa: BLE001 - optional: without shape features only the graph matcher cannot run
        print("NucleiSky: no feature vectors (%s: %s): the graph matcher is skipped" % (type(e).__name__, e), file=sys.stderr)
        features_f = features_c = None
    best, _ = run_adaptive_nucleisky(
        matcher_order=None if matcher == "auto" else [matcher],
        base_seed=0 if fixed_seed is None else int(fixed_seed),
        matcher_config=None,
        store_full_out=False,
        stop_on_success=True,
        max_total_time_s=None if max_seconds is None else float(max_seconds),
        centroids_crop_um=extract_centroids_um(df_c, name="df_crop"),
        centroids_full_um=extract_centroids_um(df_f, name="df_full"),
        img_full=reference,
        img_crop=query,
        ij_percentile_normalize=None,
        pixel_size_full_um=float(reference_pixel_size_um),
        pixel_size_crop_um=float(query_pixel_size_um),
        features_crop=features_c,
        features_full=features_f,
        df_full=df_f,
        df_crop=df_c,
        save_dir=None,
        save_prefix="adaptive",
    )
    if not best.get("success", False):
        raise ToolError("no_match", _NO_MATCH[matcher])
    with tempfile.TemporaryDirectory(prefix="nucleisky_") as scratch:  # the function writes a file; only its record is used
        rec = save_nucleisky_transform(
            best,
            scratch + "/transform.json",
            matcher_name=best.get("matcher") or "unknown",
            pixel_size_full_um=reference_pixel_size_um,
            pixel_size_crop_um=query_pixel_size_um,
            require_success=False,
        )
    if not rec["success"]:
        raise ToolError("no_match", _NO_MATCH[matcher])
    return rec


def _nuclei_points(frame):
    """The nuclei NucleiSky used, as points (pixel coordinates of the original image): the quickest way to see whether the
    segmentation (or the mask) found the nuclei it should."""
    points = frame[["centroid_y_px_orig", "centroid_x_px_orig"]].rename(columns={"centroid_y_px_orig": "y", "centroid_x_px_orig": "x"})
    for column in ("label", "area"):  # properties of each point, when the feature table has them
        if column in frame.columns:
            points[column] = frame[column].to_numpy()
    return points.reset_index(drop=True)


def _alignment_outputs(rec, query, reference, n_reference, n_query):
    import numpy as np
    from scipy.ndimage import affine_transform

    A, b = np.array(rec["A_px"]), np.array(rec["b_px"])
    Ainv = np.linalg.inv(A)
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = b
    aligned = affine_transform(query.astype(np.float32), Ainv, offset=-Ainv @ b, output_shape=reference.shape, order=1)
    return (
        M,
        aligned,
        {
            "rotation_deg": rec["rotation_deg"],
            "scale": rec["scale"],
            "offset_y_px": float(b[0]),
            "offset_x_px": float(b[1]),
            "bbox_y0y1x0x1": rec["bbox_full_px_y0y1x0x1"],
            "n_nuclei_reference": n_reference,
            "n_nuclei_query": n_query,
            "matcher": rec["matcher"],
        },
    )



@tool("List the devices")
def list_devices() -> Scalars:
    """The options of the `device` dropdown of the segmentation: the devices PyTorch can use on this machine."""
    from labconstrictor_tools.diagnostics import torch_devices

    return {"choices": ["auto"] + torch_devices()}


@tool("Relocalize 2D")
def relocalize(
    reference: Annotated[Image, Axes("YX"), Group("Images"), Description("Large / full-field image")],
    query: Annotated[Image, Axes("YX"), Group("Images"), Description("Crop to locate inside the reference")],
    reference_pixel_size_um: Annotated[
        float, Unit("um/px"), Min(0), PixelSizeOf("reference"), Group("Images")
    ] = 0.65,
    query_pixel_size_um: Annotated[float, Unit("um/px"), Min(0), PixelSizeOf("query"), Group("Images")] = 0.325,
    segmentation: Annotated[
        Literal["threshold", "cellpose", "instanseg"],
        Group("Segmentation"),
        Description("Used for every image that has no mask. cellpose / instanseg need the deep-learning packages and download model weights on first use"),
    ] = "threshold",
    device: Annotated[
        str,
        Group("Segmentation"),
        ChoicesFrom("list_devices"),
        _DEEP,
        Description("Where cellpose / instanseg run: auto = the GPU when there is one, cpu = never the GPU (use it when the GPU runs out of memory), cuda or mps = that device"),
    ] = "auto",
    matcher: Annotated[
        Literal["auto", "quad", "triangles", "graph", "hashing"],
        Group("Matching"),
        Description("auto = try the matchers in the order NucleiSky recommends for the number of nuclei until one succeeds; or force one"),
    ] = "auto",
    reference_mask: Annotated[
        Optional[Labels],
        Axes("YX"),
        Group("Images"),
        Description("Existing label image of the reference (same size): skips segmentation of the reference"),
    ] = None,
    query_mask: Annotated[
        Optional[Labels],
        Axes("YX"),
        Group("Images"),
        Description("Existing label image of the query (same size): skips segmentation of the query"),
    ] = None,
    threshold_method: Annotated[
        Literal["otsu", "li", "yen", "triangle", "isodata"], Group("Segmentation"), _THRESHOLD
    ] = "otsu",
    blur_sigma: Annotated[
        float, Min(0), Max(5), Widget("slider"), Unit("px"), Group("Segmentation"), _THRESHOLD, Description("Gaussian blur before thresholding")
    ] = 1.0,
    min_area_px: Annotated[
        int, Min(0), Unit("px"), Label("Min area"), Group("Segmentation"), _THRESHOLD, Description("Objects smaller than this are removed")
    ] = 5,
    watershed_split: Annotated[
        bool, Group("Segmentation"), _THRESHOLD, Description("Split touching nuclei with a watershed")
    ] = True,
    instanseg_model: Annotated[
        Literal["brightfield_nuclei", "fluorescence_nuclei_and_cells"], Label("InstanSeg model"), Group("Segmentation"), _INSTANSEG
    ] = "brightfield_nuclei",
    instanseg_target: Annotated[Literal["nuclei", "cells"], Label("InstanSeg target"), Group("Segmentation"), _INSTANSEG] = "nuclei",
    instanseg_cleanup_fragments: Annotated[
        bool, Label("InstanSeg: clean up fragments"), Group("Segmentation"), _INSTANSEG
    ] = True,
    instanseg_mode: Annotated[
        Literal["auto", "small", "medium"], Label("InstanSeg mode"), Group("Segmentation"), Advanced(), _INSTANSEG,
        Description("auto = small images in one piece and big ones tiled; or force one (small image / medium = tiled)"),
    ] = "auto",
    instanseg_pixel_size_um: Annotated[
        Optional[float], Min(0), Unit("um/px"), Label("InstanSeg pixel size"), Group("Segmentation"), Advanced(), _INSTANSEG,
        Description("Override the pixel size InstanSeg works at; unset = the (rescaled) pixel size of the image"),
    ] = None,
    cellpose_diameter_px: Annotated[
        Optional[float], Min(0), Unit("px"), Label("Cellpose diameter"), Group("Cellpose"), Advanced(), _CELLPOSE,
        Description("Expected nucleus diameter; unset = Cellpose estimates it"),
    ] = None,
    cellpose_flow_threshold: Annotated[
        float, Min(0), Max(3), Label("Cellpose flow threshold"), Group("Cellpose"), Advanced(), _CELLPOSE,
        Description("Maximum error of the flows of a nucleus: raise it to keep more (less regular) nuclei"),
    ] = 0.4,
    cellpose_cellprob_threshold: Annotated[
        float, Min(-6), Max(6), Label("Cellpose cell probability threshold"), Group("Cellpose"), Advanced(), _CELLPOSE,
        Description("Lower it to find more nuclei, raise it to find fewer"),
    ] = 0.0,
    cellpose_min_size_px: Annotated[
        int, Min(0), Unit("px"), Label("Cellpose minimum size"), Group("Cellpose"), Advanced(), _CELLPOSE,
        Description("Nuclei smaller than this are removed"),
    ] = 15,
    cellpose_batch_size: Annotated[
        int, Min(1), Max(64), Label("Cellpose batch size"), Group("Cellpose"), Advanced(), _CELLPOSE,
        Description("Tiles processed together: raise it for speed, lower it when the GPU runs out of memory"),
    ] = 1,
    cellpose_tile_size_px: Annotated[
        Optional[int], Min(64), Unit("px"), Label("Cellpose tile size"), Group("Cellpose"), Advanced(), _CELLPOSE,
        Description("Size of the tiles big images are cut into; unset = Cellpose's default"),
    ] = None,
    cellpose_tile_overlap: Annotated[
        Optional[float], Min(0), Max(0.9), Label("Cellpose tile overlap"), Group("Cellpose"), Advanced(), _CELLPOSE,
        Description("Fraction of overlap between tiles; unset = Cellpose's default"),
    ] = None,
    cellpose_normalize: Annotated[bool, Label("Cellpose: normalise intensities"), Group("Cellpose"), Advanced(), _CELLPOSE] = True,
    cellpose_invert: Annotated[bool, Label("Cellpose: invert the image"), Group("Cellpose"), Advanced(), _CELLPOSE, Description("For dark nuclei on a bright background")] = False,
    reference_region: Annotated[
        Optional[Labels], Axes("YX"), RegionOf("reference"), Group("Images"),
        Description("Only use the reference nuclei inside this region (the host fills it from the selection); unset = the whole reference"),
    ] = None,
    query_region: Annotated[
        Optional[Labels], Axes("YX"), RegionOf("query"), Group("Images"),
        Description("Only use the query nuclei inside this region (the host fills it from the selection); unset = the whole query"),
    ] = None,
    peak_distance_px: Annotated[
        int, Min(1), Unit("px"), Label("Peak distance"), Group("Fine-tuning"), Advanced(), _THRESHOLD, Description("Minimum distance between watershed seeds (threshold segmentation)")
    ] = 5,
    fixed_seed: Annotated[
        Optional[int], Group("Fine-tuning"), Advanced(), Description("Random seed for reproducible matching; unset = default seed 0")
    ] = None,
    max_seconds: Annotated[
        Optional[float], Min(1), Unit("s"), Label("Time limit"), Group("Fine-tuning"), Advanced(), Description("Time limit for the whole matching; unset = no limit")
    ] = None,
) -> tuple[
    Annotated[Affine, ApplyTo("query", "reference"), Name("alignment"), Replace()],
    Annotated[ImageOut, Name("query_aligned"), Replace()],
    Scalars,
    Annotated[PointsOut, Name("reference_nuclei"), ApplyTo("reference"), Replace()],
    Annotated[PointsOut, Name("query_nuclei"), ApplyTo("query"), Replace()],
    Annotated[ShapesOut, Name("reference_outlines"), ApplyTo("reference"), Replace()],
    Annotated[ShapesOut, Name("query_outlines"), ApplyTo("query"), Replace()],
]:
    """Find where a rotated / rescaled query image lies in a reference image."""
    _check_images(reference, query, reference_mask, query_mask)
    needs_segmentation = reference_mask is None or query_mask is None
    if needs_segmentation and segmentation != "threshold":
        first_use = " (the first run downloads model weights)" if segmentation == "instanseg" else ""
        progress(0.03, "loading %s%s" % (segmentation, first_use))
    progress(0.05, "segmenting nuclei" if needs_segmentation else "using the given masks")
    f_seg = c_seg = ps_f = ps_c = None
    sf = sc = 1.0
    if needs_segmentation:  # with two masks nothing is segmented: no rescaling either
        from nucleisky.nucleisky2d.preprocess import scale_normalize_pair_for_segmentation

        f_seg, c_seg, ps_f, ps_c, sf, sc, _ = scale_normalize_pair_for_segmentation(
            reference, query, reference_pixel_size_um, query_pixel_size_um
        )
    seg = _segmentation_settings(
        device=device, threshold_method=threshold_method, blur_sigma=blur_sigma, min_area_px=min_area_px,
        watershed_split=watershed_split, peak_distance_px=peak_distance_px, instanseg_model=instanseg_model,
        instanseg_target=instanseg_target, instanseg_cleanup_fragments=instanseg_cleanup_fragments, instanseg_mode=instanseg_mode,
        instanseg_pixel_size_um=instanseg_pixel_size_um, cellpose_diameter_px=cellpose_diameter_px,
        cellpose_flow_threshold=cellpose_flow_threshold, cellpose_cellprob_threshold=cellpose_cellprob_threshold,
        cellpose_min_size_px=cellpose_min_size_px, cellpose_batch_size=cellpose_batch_size, cellpose_tile_size_px=cellpose_tile_size_px,
        cellpose_tile_overlap=cellpose_tile_overlap, cellpose_normalize=cellpose_normalize, cellpose_invert=cellpose_invert,
    )  # fmt: skip
    df_f, labels_f, scale_f = _nuclei_table("reference", reference_mask, f_seg, ps_f, sf, reference_pixel_size_um, segmentation, seg)
    df_c, labels_c, scale_c = _nuclei_table("query", query_mask, c_seg, ps_c, sc, query_pixel_size_um, segmentation, seg)
    if reference_region is not None:
        df_f = _inside_region("reference", df_f, reference_region, reference)
    if query_region is not None:
        df_c = _inside_region("query", df_c, query_region, query)
    check_cancel()
    progress(0.4, "matching %d query nuclei against %d reference nuclei" % (len(df_c), len(df_f)))
    rec = _match(
        df_f, df_c, reference, query, reference_pixel_size_um, query_pixel_size_um, matcher, fixed_seed, max_seconds
    )
    result = _alignment_outputs(rec, query, reference, len(df_f), len(df_c))
    progress(0.95, "done")
    return (
        *result,
        _nuclei_points(df_f),
        _nuclei_points(df_c),
        _outlines(labels_f, scale_f, df_f["label"]),
        _outlines(labels_c, scale_c, df_c["label"]),
    )
