"""nucleisky3d package."""

from __future__ import annotations

from .features import centroids_from_df_3d, extract_nuclear_features_3d
from .io import (
    get_voxel_size_um_from_tiff,
    inspect_volume_header,
    load_nucleisky_transform_3d,
    load_transforms_any_3d,
    load_volume,
    require_voxel_size_um_zyx,
    save_nucleisky_transform_3d,
    append_transform_jsonl,
)
from .pipeline import (
    NucleiSky3D,
    evaluate_match_quality_3d,
    pick_best_transform_3d,
    run_adaptive_matching_and_export_3d,
    run_adaptive_nucleisky_3d,
)
from .matching.geometry import estimate_dynamic_scale_bounds_3d
from .matching.hashing3d import (
    geometric_hashing_match_similarity_3d,
    run_geometric_hashing_matching_3d_um,
)
from .matching.pyramid import run_pyramid_based_matching_um
from .preprocess import (
    choose_common_target_um_per_voxel,
    ij_percentile_normalize,
    rescale_to_target_um_per_voxel,
    scale_normalize_pair_for_segmentation,
)
from .segmentation import segment_nuclei_2p5d, stitch_2d_slices
from .types import BBox3D
from .visualization import imshow_safe, imshow_safe3d, plot_warp_overlay3D

__all__ = [
    "segment_nuclei_2p5d",
    "stitch_2d_slices",
    "extract_nuclear_features_3d",
    "centroids_from_df_3d",
    "NucleiSky3D",
    "run_adaptive_nucleisky_3d",
    "run_adaptive_matching_and_export_3d",
    "pick_best_transform_3d",
    "evaluate_match_quality_3d",
    "run_pyramid_based_matching_um",
    "run_geometric_hashing_matching_3d_um",
    "geometric_hashing_match_similarity_3d",
    "estimate_dynamic_scale_bounds_3d",
    "save_nucleisky_transform_3d",
    "append_transform_jsonl",
    "load_nucleisky_transform_3d",
    "load_transforms_any_3d",
    "get_voxel_size_um_from_tiff",
    "inspect_volume_header",
    "choose_common_target_um_per_voxel",
    "ij_percentile_normalize",
    "rescale_to_target_um_per_voxel",
    "scale_normalize_pair_for_segmentation",
    "load_volume",
    "require_voxel_size_um_zyx",
    "imshow_safe3d",
    "imshow_safe",
    "plot_warp_overlay3D",
    "BBox3D",
]
