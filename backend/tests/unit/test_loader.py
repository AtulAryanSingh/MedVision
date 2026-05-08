import numpy as np

from core.loader import _canon_xyz_from_array_and_spacing, get_slice_2d, normalise_to_uint8


def test_canon_xyz_from_xyz_spacing_order():
    arr = np.zeros((4, 5, 6), dtype=np.float32)  # (Z, Y, X)
    out = _canon_xyz_from_array_and_spacing(arr, [0.5, 0.7, 1.2], spacing_order="xyz")
    assert out["sizeX"] == 6
    assert out["sizeY"] == 5
    assert out["sizeZ"] == 4
    assert out["spacing_zyx"] == [1.2, 0.7, 0.5]


def test_get_slice_2d_clamps_index():
    arr = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    slc = get_slice_2d(arr, axis=0, index=100)
    np.testing.assert_array_equal(slc, arr[-1, :, :])


def test_normalise_to_uint8_constant_returns_zeros():
    arr = np.ones((3, 3), dtype=np.float32) * 42.0
    out = normalise_to_uint8(arr)
    assert out.dtype == np.uint8
    assert np.all(out == 0)
