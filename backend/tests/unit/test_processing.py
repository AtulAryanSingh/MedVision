import numpy as np

from processing.filters import apply_gaussian
from processing.histogram import apply_cdf_threshold
from processing.morphology import label_connected_components


def test_apply_gaussian_preserves_shape_and_dtype():
    arr = np.zeros((8, 8), dtype=np.float32)
    arr[3:5, 3:5] = 10.0
    out = apply_gaussian(arr, sigma=1.0)
    assert out.shape == arr.shape
    assert out.dtype == np.float32
    assert float(out.max()) <= float(arr.max())
    assert float(out.min()) >= float(arr.min())


def test_apply_cdf_threshold_is_binary():
    arr = np.linspace(0, 255, 25, dtype=np.float32).reshape(5, 5)
    out = apply_cdf_threshold(arr, percentile=50.0)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_label_connected_components_detects_single_region():
    arr = np.zeros((20, 20), dtype=np.float32)
    arr[5:10, 5:10] = 255.0
    label_img, comps = label_connected_components(arr, threshold=128.0, min_area=5)
    assert label_img.shape == (20, 20, 3)
    assert len(comps) == 1
    assert comps[0]["area"] >= 25
