from nucleisky3d.utils import compute_min_inliers_stable


def test_compute_min_inliers_small_crop_is_capped_to_80_percent():
    Nc = 30
    min_inliers = compute_min_inliers_stable(
        Nc,
        min_inliers_abs=40,
        min_inliers_frac=0.95,
        hard_floor=3,
        cap_frac=0.80,
    )

    assert min_inliers <= int(0.80 * Nc)


def test_compute_min_inliers_changes_with_abs_and_frac_inputs():
    Nc = 30
    baseline = compute_min_inliers_stable(
        Nc,
        min_inliers_abs=5,
        min_inliers_frac=0.10,
        hard_floor=3,
        cap_frac=0.80,
    )
    larger_abs = compute_min_inliers_stable(
        Nc,
        min_inliers_abs=12,
        min_inliers_frac=0.10,
        hard_floor=3,
        cap_frac=0.80,
    )
    larger_frac = compute_min_inliers_stable(
        Nc,
        min_inliers_abs=5,
        min_inliers_frac=0.35,
        hard_floor=3,
        cap_frac=0.80,
    )

    assert larger_abs > baseline
    assert larger_frac > baseline
