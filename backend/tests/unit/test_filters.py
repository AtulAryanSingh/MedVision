"""
tests/unit/test_filters.py

Unit tests for processing/filters.py.

Covers
------
apply_gaussian  – output shape, value smoothing (max ≤ input max), sigma=0
apply_median    – output shape, dtype, noise-reduction property
apply_sobel     – output shape, dtype, uniform image → all edges zero
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")


# ─────────────────────────────────────────────────────────────────────────────
# apply_gaussian
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyGaussian:
    def test_shape_preserved(self, sample_2d_array):
        from processing.filters import apply_gaussian
        result = apply_gaussian(sample_2d_array, sigma=2.0)
        assert result.shape == sample_2d_array.shape

    def test_output_dtype_float32(self, sample_2d_array):
        from processing.filters import apply_gaussian
        result = apply_gaussian(sample_2d_array, sigma=2.0)
        assert result.dtype == np.float32

    def test_values_stay_within_input_range(self, sample_2d_array):
        """Gaussian blur should not produce values outside [min, max] of input."""
        from processing.filters import apply_gaussian
        result = apply_gaussian(sample_2d_array, sigma=2.0)
        assert result.min() >= sample_2d_array.min() - 1e-3
        assert result.max() <= sample_2d_array.max() + 1e-3

    def test_large_sigma_increases_uniformity(self, sample_2d_array):
        """A very large sigma should reduce the standard deviation significantly."""
        from processing.filters import apply_gaussian
        result = apply_gaussian(sample_2d_array, sigma=20.0)
        assert result.std() < sample_2d_array.std()

    def test_sigma_zero_does_not_raise(self, sample_2d_array):
        from processing.filters import apply_gaussian
        result = apply_gaussian(sample_2d_array, sigma=0.0)
        assert result.shape == sample_2d_array.shape

    def test_3d_array_shape_preserved(self, sample_3d_volume):
        from processing.filters import apply_gaussian
        result = apply_gaussian(sample_3d_volume, sigma=1.0)
        assert result.shape == sample_3d_volume.shape


# ─────────────────────────────────────────────────────────────────────────────
# apply_median
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyMedian:
    def test_shape_preserved(self, sample_2d_array):
        from processing.filters import apply_median
        result = apply_median(sample_2d_array, kernel_size=3)
        assert result.shape == sample_2d_array.shape

    def test_output_dtype_float32(self, sample_2d_array):
        from processing.filters import apply_median
        result = apply_median(sample_2d_array, kernel_size=3)
        assert result.dtype == np.float32

    def test_removes_isolated_noise(self):
        """Median filter should suppress isolated high-value pixels."""
        from processing.filters import apply_median
        base = np.zeros((16, 16), dtype=np.float32)
        # Plant a single bright pixel
        base[8, 8] = 255.0
        result = apply_median(base, kernel_size=3)
        assert result[8, 8] < 255.0, "Isolated spike should be suppressed"

    def test_even_kernel_size_forced_to_odd(self, sample_2d_array):
        """Even kernel sizes should be forced to the next odd value without error."""
        from processing.filters import apply_median
        result = apply_median(sample_2d_array, kernel_size=4)
        assert result.shape == sample_2d_array.shape


# ─────────────────────────────────────────────────────────────────────────────
# apply_sobel
# ─────────────────────────────────────────────────────────────────────────────

class TestApplySobel:
    def test_shape_is_2d(self, sample_2d_array):
        from processing.filters import apply_sobel
        result = apply_sobel(sample_2d_array)
        assert result.ndim == 2

    def test_output_dtype_float32(self, sample_2d_array):
        from processing.filters import apply_sobel
        result = apply_sobel(sample_2d_array)
        assert result.dtype == np.float32

    def test_output_range_0_to_255(self, sample_2d_array):
        from processing.filters import apply_sobel
        result = apply_sobel(sample_2d_array)
        assert result.min() >= 0.0
        assert result.max() <= 255.0 + 1e-3

    def test_uniform_image_gives_zero_edges(self):
        """A uniform-intensity image has no edges → Sobel result should be all zeros."""
        from processing.filters import apply_sobel
        uniform = np.full((32, 32), 128.0, dtype=np.float32)
        result = apply_sobel(uniform)
        assert np.all(result == 0.0)

    def test_step_edge_detected(self):
        """A step edge should produce a non-zero Sobel response along the edge."""
        from processing.filters import apply_sobel
        step = np.zeros((32, 32), dtype=np.float32)
        step[:, 16:] = 200.0  # left half dark, right half bright
        result = apply_sobel(step)
        # Maximum response should be at the edge column (around col 16)
        assert result.max() > 0.0
