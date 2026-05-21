
""" features.py Feature extraction utilities (centroids, regionprops, etc.)."""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from scipy.spatial import cKDTree
from .preprocess import _require_df_columns


_EPS = 1e-8


def extract_nuclear_features(
    label_img,
    intensity_img=None,
    pixel_size_um=1.0,
    k_neighbors=10,
    min_area_px=None,
    max_area_px=None,
    edge_margin_px=0,
):
    """
    Drop-in replacement:
      - vectorizes kNN distance fill (faster)
      - validates 2D label image
      - creates feature_vector as list of float32 numpy rows (stable for np.stack)
    """
    label_img = np.asarray(label_img)
    if label_img.ndim != 2:
        raise ValueError(f"extract_nuclear_features expects a 2D label image. Got shape={label_img.shape}")

    H, W = label_img.shape[:2]

    props_to_measure = [
        "label",
        "centroid",
        "area",
        "perimeter",
        "major_axis_length",
        "minor_axis_length",
        "eccentricity",
        "solidity",
        "extent",
        "bbox",
    ]

    rp = regionprops_table(
        label_img,
        intensity_image=intensity_img,
        properties=props_to_measure,
    )
    df = pd.DataFrame(rp)

    if df.empty:
        print("[features] No labeled objects found.")
        return df

    # ----- area filter -----
    if min_area_px is None:
        min_area_px = -np.inf
    if max_area_px is None:
        max_area_px =  np.inf

    mask_valid = (df["area"] >= float(min_area_px)) & (df["area"] <= float(max_area_px))

    # ----- edge filter -----
    if edge_margin_px > 0:
        m = int(edge_margin_px)
        min_row = df["bbox-0"]
        min_col = df["bbox-1"]
        max_row = df["bbox-2"]
        max_col = df["bbox-3"]

        mask_edge = (
            (min_row <= m) |
            (min_col <= m) |
            (max_row >= H - m) |
            (max_col >= W - m)
        )
        mask_valid &= ~mask_edge

    df = df.loc[mask_valid].copy().reset_index(drop=True)
    if df.empty:
        print("[features] No nuclei passed area/edge filter.")
        return df

    px = float(pixel_size_um)
    if not np.isfinite(px) or px <= 0:
        raise ValueError(f"pixel_size_um must be positive finite. Got {pixel_size_um}")

    # ---- coordinates & physical units ----
    df["centroid_y_px"] = df["centroid-0"]
    df["centroid_x_px"] = df["centroid-1"]
    df["centroid_y_um"] = df["centroid_y_px"] * px
    df["centroid_x_um"] = df["centroid_x_px"] * px

    df["area_um2"]          = df["area"] * (px**2)
    df["perimeter_um"]      = df["perimeter"] * px
    df["major_axis_um"]     = df["major_axis_length"] * px
    df["minor_axis_um"]     = df["minor_axis_length"] * px
    df["equiv_diameter_um"] = 2.0 * np.sqrt(df["area_um2"] / np.pi)

    # ---- scale-invariant descriptors ----
    eps = 1e-8
    df["aspect_ratio"] = df["major_axis_length"] / (df["minor_axis_length"] + eps)
    df["circularity"]  = 4.0 * np.pi * df["area"] / (df["perimeter"]**2 + eps)
    df["circularity"]  = df["circularity"].clip(upper=1.0)

    med_area = float(np.median(df["area_um2"].to_numpy()))
    med_peri = float(np.median(df["perimeter_um"].to_numpy()))
    df["area_um2_norm"]     = df["area_um2"] / (med_area + eps)
    df["perimeter_um_norm"] = df["perimeter_um"] / (med_peri + eps)

    # ---- neighbourhood features in µm ----
    centroids_um = df[["centroid_y_um", "centroid_x_um"]].to_numpy(dtype=np.float32, copy=False)
    tree = cKDTree(centroids_um)

    N = len(df)
    k_neighbors = int(k_neighbors)
    kn = min(k_neighbors + 1, N)  # include self
    dists, _ = tree.query(centroids_um, k=kn)
    dists = np.atleast_2d(dists)[:, 1:]  # drop self (N, kn-1)

    nn_d = np.zeros((N, k_neighbors), dtype=np.float32)
    m = min(k_neighbors, dists.shape[1])
    if m > 0:
        nn_d[:, :m] = dists[:, :m].astype(np.float32, copy=False)

    for j in range(k_neighbors):
        df[f"nn{j+1}_dist_um"] = nn_d[:, j]

    # local density within fixed radius
    radius_um = 20.0
    neigh_lists = tree.query_ball_point(centroids_um, r=float(radius_um))
    df["local_density_r20"] = np.asarray([len(lst) - 1 for lst in neigh_lists], dtype=np.int32)

    all_nn = dists[(dists > 0) & np.isfinite(dists)]
    med_nn = float(np.median(all_nn)) if all_nn.size else 1.0
    df["local_density_norm"] = df["local_density_r20"] / (med_nn + eps)

    # ---- feature vector ----
    feature_cols = [
        "area_um2_norm",
        "perimeter_um_norm",
        "aspect_ratio",
        "eccentricity",
        "solidity",
        "extent",
        "circularity",
        "equiv_diameter_um",
        "local_density_norm",
    ] + [f"nn{i+1}_dist_um" for i in range(k_neighbors)]

    feat_mat = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    df["feature_vector"] = list(feat_mat)  # list of row arrays; stable for np.stack

    return df


def centroids_from_df(df, use_um=False, use_orig_px=False):
    """
    Return centroids as an (N,2) array in (y,x) order.

    Parameters
    ----------
    use_um : bool
        If True: returns centroid_y_um / centroid_x_um.
    use_orig_px : bool
        If True and columns centroid_*_px_orig exist: returns those.
        Otherwise falls back to centroid_y_px / centroid_x_px.

    Returns
    -------
    (N,2) float32 ndarray
    """

    if df is None or len(df) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if use_um:
        y_col, x_col = "centroid_y_um", "centroid_x_um"
    else:
        if use_orig_px and ("centroid_y_px_orig" in df.columns) and ("centroid_x_px_orig" in df.columns):
            y_col, x_col = "centroid_y_px_orig", "centroid_x_px_orig"
        else:
            y_col, x_col = "centroid_y_px", "centroid_x_px"

    if y_col not in df.columns or x_col not in df.columns:
        raise KeyError(f"DataFrame missing required centroid columns: {y_col}, {x_col}")

    return df[[y_col, x_col]].to_numpy(dtype=np.float32, copy=False)


def add_centroids_orig_px_columns(df, scale_factor, *, y_col="centroid_y_px", x_col="centroid_x_px"):
    """
    After feature extraction on a rescaled image, add centroid coords mapped back
    to the ORIGINAL image pixel grid.

      orig_px = rescaled_px / scale_factor
    """
    if df is None or len(df) == 0:
        return df
    sf = float(scale_factor) if scale_factor not in (None, 0) else 1.0

    if y_col in df.columns and x_col in df.columns:
        df["centroid_y_px_orig"] = df[y_col] / sf
        df["centroid_x_px_orig"] = df[x_col] / sf
    return df


def extract_centroids_um(df, *, name: str):
    _require_df_columns(df, ["centroid_y_um", "centroid_x_um"], name=name)
    C = df[["centroid_y_um", "centroid_x_um"]].to_numpy(dtype=float, copy=False)
    if C.ndim != 2 or C.shape[1] != 2:
        raise ValueError(f"{name} centroids must be (N,2). Got {C.shape}")
    if len(C) < 3:
        raise ValueError(f"{name} needs at least 3 centroids for matching; got N={len(C)}.")
    if not np.isfinite(C).all():
        bad = np.where(~np.isfinite(C).all(axis=1))[0]
        raise ValueError(f"{name} centroids contain non-finite rows at indices: {bad[:20].tolist()}")
    return C


def _sanitize_features(feat, ref_feat=None):
    """
    Ensure finite feature matrix for KDTree by imputing non-finite values.
    If ref_feat is provided, medians are computed on ref_feat (preferred).
    """
    X = np.asarray(feat, float)
    if X.ndim == 1:
        X = X[:, None]
    X = X.copy()

    R = np.asarray(ref_feat, float) if ref_feat is not None else X
    if R.ndim == 1:
        R = R[:, None]

    for c in range(X.shape[1]):
        col_ref = R[:, c]
        finite_ref = col_ref[np.isfinite(col_ref)]
        fill = float(np.median(finite_ref)) if finite_ref.size else 0.0
        bad = ~np.isfinite(X[:, c])
        if np.any(bad):
            X[bad, c] = fill

    X[~np.isfinite(X)] = 0.0
    return X


def _zscore_with_ref(feat, ref_mu=None, ref_sigma=None):
    X = np.asarray(feat, float)
    if X.ndim == 1:
        X = X[:, None]

    if ref_mu is None or ref_sigma is None:
        mu = np.nanmean(X, axis=0, keepdims=True)
        sigma = np.nanstd(X, axis=0, keepdims=True)
    else:
        mu = np.asarray(ref_mu, float)
        sigma = np.asarray(ref_sigma, float)

    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)

    Z = (X - mu) / (sigma + _EPS)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def stack_feature_vectors(df, *, name: str):
    _require_df_columns(df, ["feature_vector"], name=name)
    if len(df) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    vecs = df["feature_vector"].to_numpy()
    arrs = []
    for i, v in enumerate(vecs):
        a = np.asarray(v, dtype=float).ravel()
        if a.size == 0:
            raise ValueError(f"{name} feature_vector at row {i} is empty.")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} feature_vector at row {i} contains NaN/Inf.")
        arrs.append(a)

    Ls = {a.size for a in arrs}
    if len(Ls) != 1:
        lens = sorted(list(Ls))[:10]
        raise ValueError(f"{name} feature_vector lengths are inconsistent (examples: {lens}). Ensure fixed-length vectors.")

    X = np.vstack(arrs).astype(np.float32, copy=False)
    return X


def _robust_median(x, fallback=1.0, positive_only=False):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if positive_only:
        x = x[x > 0]
    if x.size == 0:
        return float(fallback)
    m = float(np.median(x))
    return float(m) if np.isfinite(m) else float(fallback)


def _robust_mad(x, fallback=1.0):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(fallback)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return mad if np.isfinite(mad) and mad > 0 else float(fallback)
