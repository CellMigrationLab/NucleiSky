import numpy as np
import pytest

from nucleisky2d.segmentation import (
    Segmentor,
    _remove_small_holes_compat,
    _remove_small_objects_compat,
)
from nucleisky3d.segmentation import segment_nuclei_2p5d


@pytest.mark.geometry
def test_remove_small_objects_compat_threshold_semantics_and_shape_dtype():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:3, 1:3] = True      # area 4
    mask[4:7, 4:7] = True      # area 9
    mask[8:12, 8:12] = True    # area 16

    out = _remove_small_objects_compat(mask, min_object_size=9)
    assert out.shape == mask.shape
    assert out.dtype == np.bool_
    assert not out[1:3, 1:3].any()      # below threshold removed
    assert out[4:7, 4:7].all()          # exactly threshold retained
    assert out[8:12, 8:12].all()        # above threshold retained


@pytest.mark.geometry
def test_remove_small_holes_compat_threshold_semantics_and_background():
    mask = np.ones((12, 12), dtype=bool)
    mask[2:4, 2:4] = False      # enclosed hole area 4
    mask[6:9, 6:9] = False      # enclosed hole area 9
    mask[0, :] = False          # background-connected region, not a hole

    out = _remove_small_holes_compat(mask, min_hole_size=9)
    assert out.shape == mask.shape
    assert out.dtype == np.bool_
    assert out[2:4, 2:4].all()          # all 4 pixels of the area-4 hole are filled (4 < 9)
    assert not out[6:9, 6:9].all()      # all 9 pixels of the area-9 hole are retained (9 is not < 9)
    assert not out[0, :].any()          # background-connected row remains False


@pytest.mark.integration
def test_threshold_segmentation_path_no_futurewarning_and_stable_filtering(recwarn):
    img = np.zeros((24, 24), dtype=np.float32)
    img[2:4, 2:4] = 1.0
    img[8:14, 8:14] = 1.0

    seg = Segmentor()
    labels = seg.segment_threshold(
        img,
        threshold_method="otsu",
        gaussian_sigma=0.0,
        min_object_size=9,
        min_hole_size=9,
        do_watershed=False,
    )
    assert labels.shape == img.shape
    assert labels.dtype == np.int32
    assert labels.max() == 1  # tiny object removed; large object retained

    fws = [w for w in recwarn if issubclass(w.category, FutureWarning)]
    assert not fws, f"Unexpected FutureWarning(s): {[str(w.message) for w in fws]}"


@pytest.mark.integration
def test_3d_threshold_segmentation_still_filters_tiny_components_anisotropic():
    vol = np.zeros((4, 20, 20), dtype=np.float32)
    vol[:, 2:4, 2:4] = 1.0      # tiny component
    vol[:, 8:14, 8:14] = 1.0    # large component

    labels = segment_nuclei_2p5d(
        volume_zyx=vol,
        method="threshold",
        pixel_size_um_zyx=(2.0, 0.5, 0.5),
        settings={"threshold": {"threshold_method": "otsu", "min_object_size": 20, "do_watershed": False}},
        show_progress=False,
    )
    assert labels.shape == vol.shape
    assert labels.max() >= 1
    coords = np.argwhere(labels > 0)
    assert coords[:, 1].min() >= 8
