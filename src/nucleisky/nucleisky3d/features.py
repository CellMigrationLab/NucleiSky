"""features.py Feature extraction utilities for 3D nuclei (centroids, regionprops, etc.)."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - exercised in environments without SimpleITK.
    sitk = None

_EPS = 1e-8


def _require_simpleitk() -> None:
    """Raise a clear runtime error when SimpleITK-dependent APIs are used without it."""
    if sitk is None:
        raise ImportError(
            "SimpleITK is required for nucleisky3d.features but is not installed. "
            "Install it with `pip install SimpleITK` to use 3D feature extraction."
        )

def _surface_area_from_mask(mask_3d, pixel_size_um):
    """
    Estimate the surface area of a 3D binary mask using SimpleITK.
    (Kept for backward compatibility, though no longer needed in the main feature extractor).
    """
    mask = np.asarray(mask_3d, dtype=np.uint8)
    if mask.ndim != 3:
        raise ValueError(f"_surface_area_from_mask expects a 3D mask. Got shape={mask.shape}")
    if not mask.any():
        return 0.0

    _require_simpleitk()

    spacing_arr = np.asarray(pixel_size_um, dtype=float)
    if spacing_arr.size == 1:
        spacing = (float(spacing_arr),) * 3
    elif spacing_arr.shape == (3,):
        spacing = tuple(float(v) for v in spacing_arr)
    else:
        raise ValueError("_surface_area_from_mask expects pixel_size_um as scalar or length-3 (z, y, x).")

    # Convert to SimpleITK Image
    sitk_img = sitk.GetImageFromArray(mask)
    
    # ITK/SimpleITK expects spacing in (X, Y, Z) order!
    sitk_img.SetSpacing((spacing[2], spacing[1], spacing[0]))

    # Execute shape stats filter
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(sitk_img)

    # In 3D, SimpleITK's Perimeter is the surface area
    if shape_stats.HasLabel(1):
        return shape_stats.GetPerimeter(1)
    return 0.0


def extract_nuclear_features_3d(label_img_3d, pixel_size_um=1.0, k_neighbors=5):
    """
    Extract basic 3D nuclear features using SimpleITK for massive speedups.
    """
    label_img_3d = np.asarray(label_img_3d)
    _require_simpleitk()
    if label_img_3d.ndim != 3:
        raise ValueError(
            "extract_nuclear_features_3d expects a 3D label image. "
            f"Got shape={label_img_3d.shape}"
        )

    px_arr = np.asarray(pixel_size_um, dtype=float)
    if px_arr.size == 1:
        px = np.repeat(float(px_arr), 3)
    elif px_arr.shape == (3,):
        px = px_arr
    else:
        raise ValueError(
            "pixel_size_um must be a positive scalar or length-3 (z, y, x). "
            f"Got {pixel_size_um}"
        )
    if not np.isfinite(px).all() or (px <= 0).any():
        raise ValueError(f"pixel_size_um must be positive finite. Got {pixel_size_um}")

    # 1. Convert to SimpleITK Image (Requires uint32 or uint64 for label maps)
    sitk_img = sitk.GetImageFromArray(label_img_3d.astype(np.uint32))
    
    # CRITICAL: SimpleITK expects spacing in (X, Y, Z) order
    sitk_img.SetSpacing((float(px[2]), float(px[1]), float(px[0])))

    # 2. Run highly optimized C++ Shape Statistics Filter once for ALL labels
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(sitk_img)

    labels = shape_stats.GetLabels()
    if not labels:
        print("[features_3d] No labeled objects found.")
        return pd.DataFrame()

    # 3. Extract properties
    data = []
    for lbl in labels:
        # GetCentroid returns physical coordinates in (X, Y, Z)
        cx_um, cy_um, cz_um = shape_stats.GetCentroid(lbl)
        
        # Convert physical back to pixel coordinates
        cx_px = cx_um / px[2]
        cy_px = cy_um / px[1]
        cz_px = cz_um / px[0]

        data.append({
            "label": lbl,
            "centroid_z_px": cz_px,
            "centroid_y_px": cy_px,
            "centroid_x_px": cx_px,
            "centroid_z_um": cz_um,
            "centroid_y_um": cy_um,
            "centroid_x_um": cx_um,
            "volume_voxels": shape_stats.GetNumberOfPixels(lbl),
            "volume_um3": shape_stats.GetPhysicalSize(lbl),
            "surface_area_um2": shape_stats.GetPerimeter(lbl) # Perimeter = Surface Area in 3D
        })

    df = pd.DataFrame(data)

    # 4. Vectorized mathematics (Instantaneous)
    df["equiv_spherical_diameter_um"] = 2.0 * (
        (3.0 * df["volume_um3"]) / (4.0 * np.pi + _EPS)
    ) ** (1.0 / 3.0)

    df["sphericity"] = (
        (np.pi ** (1.0 / 3.0)) * (6.0 * df["volume_um3"]) ** (2.0 / 3.0)
    ) / (df["surface_area_um2"] + _EPS)
    df["sphericity"] = df["sphericity"].clip(upper=1.0)

    med_volume = float(np.median(df["volume_um3"].to_numpy()))
    df["volume_norm"] = df["volume_um3"] / (med_volume + _EPS)

    # 5. Spatial density + nearest-neighbor distance features.
    centroids_um = df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(
        dtype=np.float32, copy=False
    )
    tree = cKDTree(centroids_um)
    nn_dists = np.zeros(len(df), dtype=np.float32)
    k_neighbors = int(max(1, k_neighbors))

    if len(df) > 1:
        k_query = min(len(df), k_neighbors + 1)
        dists = tree.query(centroids_um, k=k_query)[0]
        dists = np.atleast_2d(np.asarray(dists, dtype=np.float32))
        if dists.shape[0] != len(df):
            dists = dists.T
        nn_dists = dists[:, 1]

        for k in range(1, k_neighbors + 1):
            col = f"nn{k}_dist_um"
            df[col] = np.nan
            if (k + 1) <= dists.shape[1]:
                df[col] = dists[:, k]
    else:
        for k in range(1, k_neighbors + 1):
            df[f"nn{k}_dist_um"] = np.nan

    med_nn = float(np.median(nn_dists[nn_dists > 0])) if np.any(nn_dists > 0) else 1.0

    radius_um = 20.0
    neigh_lists = tree.query_ball_point(centroids_um, r=float(radius_um))
    df["local_density_r20"] = np.asarray([len(lst) - 1 for lst in neigh_lists], dtype=np.int32)
    df["local_density_norm"] = df["local_density_r20"] / (med_nn + _EPS)

    feature_cols = ["volume_norm", "sphericity", "local_density_norm"]
    feat_mat = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    df["feature_vector"] = list(feat_mat)

    return df


def centroids_from_df_3d(df: pd.DataFrame, use_um: bool = True) -> np.ndarray:
    """Extract centroids as an ``(N, 3)`` array from a 3D feature dataframe.

    Parameters
    ----------
    df:
        Dataframe returned by :func:`extract_nuclear_features_3d` or any dataframe
        containing centroid columns.
    use_um:
        If ``True`` (default), prefer ``*_um`` centroid columns; otherwise read
        ``*_px`` centroid columns.
    """
    if df is None or len(df) == 0:
        return np.empty((0, 3), dtype=np.float32)

    um_cols = ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
    px_cols = ["centroid_z_px", "centroid_y_px", "centroid_x_px"]
    cols = um_cols if use_um else px_cols

    if not all(col in df.columns for col in cols):
        raise ValueError(f"DataFrame missing required centroid columns: {cols}")

    return df[cols].to_numpy(dtype=np.float32, copy=False)
