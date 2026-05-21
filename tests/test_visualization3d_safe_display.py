import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nucleisky3d.visualization import _downsample_for_display, imshow_safe, imshow_safe3d


def test_downsample_for_display_returns_strided_view_for_large_image():
    img = np.arange(100 * 80, dtype=np.float32).reshape(100, 80)
    out, step = _downsample_for_display(img, max_dim=25)

    assert out.shape == (25, 20)
    assert step == 4
    np.testing.assert_array_equal(out, img[::4, ::4])


def test_downsample_for_display_noop_for_small_image():
    img = np.arange(20 * 10, dtype=np.float32).reshape(20, 10)
    out, step = _downsample_for_display(img, max_dim=100)

    assert step == 1
    np.testing.assert_array_equal(out, img)


def test_imshow_safe3d_downsamples_and_normalizes_rgb():
    img = np.random.default_rng(0).random((120, 90, 3), dtype=np.float32)
    fig, ax = plt.subplots(1, 1)
    disp = imshow_safe3d(ax, img, title="rgb", max_dim=30)

    assert disp.shape == (30, 23, 3)
    assert disp.dtype == np.float32
    assert disp.min() >= 0.0
    assert disp.max() <= 1.0
    plt.close(fig)


def test_imshow_safe3d_is_exposed_at_package_level():
    import nucleisky3d

    assert hasattr(nucleisky3d, "imshow_safe3d")


def test_imshow_safe_alias_points_to_imshow_safe3d():
    assert imshow_safe is imshow_safe3d
