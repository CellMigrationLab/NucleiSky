
"""Notebook-only demo/synthetic utilities."""

import numpy as np
from scipy.ndimage import map_coordinates

def generate_random_crop(img_full, patch_h, patch_w, zoom_range, max_angle_deg, pixel_size_um, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    zoom_factor = float(rng.uniform(*zoom_range))
    angle_deg = float(rng.uniform(-max_angle_deg, max_angle_deg))
    theta = np.deg2rad(angle_deg)

    H, W = img_full.shape
    half_diag = np.sqrt(patch_h**2 + patch_w**2) / (2 * zoom_range[0])
    margin_y = int(np.ceil(half_diag))
    margin_x = int(np.ceil(half_diag))

    if H <= 2 * margin_y or W <= 2 * margin_x:
        raise ValueError("Big image too small for chosen patch/zoom/rotation range.")

    cy = rng.integers(margin_y, H - margin_y)
    cx = rng.integers(margin_x, W - margin_x)

    yy, xx = np.mgrid[0:patch_h, 0:patch_w]
    cy_patch, cx_patch = (patch_h - 1) / 2.0, (patch_w - 1) / 2.0
    y_c = yy - cy_patch
    x_c = xx - cx_patch
    y_world = y_c / zoom_factor
    x_world = x_c / zoom_factor
    c, s = np.cos(theta), np.sin(theta)
    y_rot = c * y_world - s * x_world
    x_rot = s * y_world + c * x_world
    Y_big = cy + y_rot
    X_big = cx + x_rot

    crop = map_coordinates(img_full.astype(np.float32), [Y_big, X_big], order=1, mode="reflect")
    crop_final = crop.astype(img_full.dtype)
    crop_pixel_size = pixel_size_um / zoom_factor

    return crop_final, crop_pixel_size, (cy, cx, angle_deg, zoom_factor)
