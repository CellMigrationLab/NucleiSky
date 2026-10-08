"""Host-side fixture generator (numpy+scipy+tifffile only). Synthetic nuclei field + known-transform crop."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from tifffile import imwrite

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
H = W = 1600
PS = 0.65
img = np.zeros((H, W), np.float32)
n = 450
ys, xs = rng.uniform(40, H - 40, n), rng.uniform(40, W - 40, n)
rad = rng.uniform(5, 9, n)
amp = rng.uniform(0.5, 1.0, n)
yy, xx = np.mgrid[0:H, 0:W]
for y, x, r, a in zip(ys, xs, rad, amp):
    y0, y1, x0, x1 = int(y - 3 * r), int(y + 3 * r) + 1, int(x - 3 * r), int(x + 3 * r) + 1
    sub = np.exp(-(((yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2) / (2 * r * r)))
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], a * sub)
img = gaussian_filter(img, 1.0) + rng.normal(0, 0.01, img.shape).astype(np.float32)
full = np.clip(img * 4000, 0, 65535).astype(np.uint16)

zoom, ang, ph, pw = 2.0, 12.0, 520, 520
cy, cx = 800, 760
th = np.deg2rad(ang)
c, s = np.cos(th), np.sin(th)
y0g, x0g = np.mgrid[0:ph, 0:pw]
yc, xc = (y0g - (ph - 1) / 2) / zoom, (x0g - (pw - 1) / 2) / zoom
Y, X = cy + c * yc - s * xc, cx + s * yc + c * xc
crop = map_coordinates(full.astype(np.float32), [Y, X], order=1, mode="reflect")
crop = np.clip(
    crop * np.random.default_rng(3).uniform(0.9, 1.1) + np.random.default_rng(4).normal(0, 20, crop.shape),
    0,
    65535,
).astype(np.uint16)
imwrite(out / "reference.tif", full)
imwrite(out / "query.tif", crop)
# simple label images (connected components above a fraction of the maximum), for the "existing mask" inputs
from scipy.ndimage import label as _label

imwrite(out / "reference_mask.tif", _label(full > 0.25 * full.max())[0].astype(np.int32))
imwrite(out / "query_mask.tif", _label(crop > 0.25 * crop.max())[0].astype(np.int32))
# the same masks as binary images (one value for every nucleus: the tool must label the connected objects itself) and a
# deliberately invalid one (fractions)
imwrite(out / "reference_mask_binary.tif", (full > 0.25 * full.max()).astype(np.uint8))
imwrite(out / "query_mask_binary.tif", (crop > 0.25 * crop.max()).astype(np.uint8))
imwrite(out / "query_mask_fractions.tif", (_label(crop > 0.25 * crop.max())[0] + 0.5).astype(np.float32))
# ground truth: crop px (y,x) -> full px:  full = M @ (crop - ctr) + (cy,cx), M = R/zoom
M = np.array([[c, -s], [s, c]]) / zoom
ctr = np.array([(ph - 1) / 2, (pw - 1) / 2])
b = np.array([cy, cx]) - M @ ctr
json.dump(
    {
        "A_px": M.tolist(),
        "b_px": b.tolist(),
        "rotation_deg": ang,
        "scale": zoom,
        "reference": {"shape": [H, W], "pixel_size_um": PS},
        "query": {"shape": [ph, pw], "pixel_size_um": PS / zoom},
    },
    open(out / "ground_truth.json", "w"),
    indent=2,
)
print("ok", full.shape, crop.shape)

# regions for the RegionOf inputs: a generous box around where the query lies (a tight one leaves too few nuclei to match), an empty one, and one of the wrong size
region = np.zeros(full.shape, np.int32)
region[300:1500, 300:1500] = 1
imwrite(out / "reference_region.tif", region)
imwrite(out / "empty_region.tif", np.zeros(full.shape, np.int32))
imwrite(out / "wrong_size_region.tif", np.ones((100, 100), np.int32))
