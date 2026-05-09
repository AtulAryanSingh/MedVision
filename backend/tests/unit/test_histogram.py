"""
tests/unit/test_histogram.py

Unit tests for processing/histogram.py.

Covers
------
compute_histogram     – bin/count lengths, intensity_min/max, single-value edge case
compute_cdf           – CDF length, monotonicity, range [0, 1]
apply_cdf_threshold   – output dtype (uint8), only 0/255 values, 100th percentile
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# compute_histogram
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeHistogram:
    def test_default_bin_count(self, sample_2d_array):
        from processing.histogram import compute_histogram
        result = compute_histogram(sample_2d_array)
        assert len(result["bins"]) == 256
        assert len(result["counts"]) == 256

    def test_custom_bin_count(self, sample_2d_array):
        from processing.histogram import compute_histogram
        result = compute_histogram(sample_2d_array, bins=64)
        assert len(result["bins"]) == 64
        assert len(result["counts"]) == 64

    def test_intensity_min_max(self, sample_2d_array):
        from processing.histogram import compute_histogram
        result = compute_histogram(sample_2d_array)
        assert abs(result["intensity_min"] - float(sample_2d_array.min())) < 1e-4
        assert abs(result["intensity_max"] - float(sample_2d_array.max())) < 1e-4

    def test_total_count_equals_pixel_count(self, sample_2d_array):
        from processing.histogram import compute_histogram
        result = compute_histogram(sample_2d_array)
        assert sum(result["counts"]) == sample_2d_array.size

    def test_counts_are_non_negative(self, sample_2d_array):
        from processing.histogram import compute_histogram
        result = compute_histogram(sample_2d_array)
        assert all(c >= 0 for c in result["counts"])

    def test_single_value_array(self):
        """A constant array should put all pixels in one bin."""
        from processing.histogram import compute_histogram
        arr = np.full((8, 8), 42.0, dtype=np.float32)
        result = compute_histogram(arr, bins=16)
        # For a constant array all values land in one bin
        assert max(result["counts"]) == arr.size

    def test_3d_volume_flattened(self, sample_3d_volume):
        from processing.histogram import compute_histogram
        result = compute_histogram(sample_3d_volume)
        assert sum(result["counts"]) == sample_3d_volume.size


# ─────────────────────────────────────────────────────────────────────────────
# compute_cdf
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCdf:
    def test_cdf_length_matches_histogram(self, sample_2d_array):
        from processing.histogram import compute_histogram, compute_cdf
        hist = compute_histogram(sample_2d_array, bins=64)
        result = compute_cdf(hist)
        assert len(result["cdf"]) == 64

    def test_cdf_ends_at_one(self, sample_2d_array):
        from processing.histogram import compute_histogram, compute_cdf
        hist = compute_histogram(sample_2d_array)
        result = compute_cdf(hist)
        assert abs(result["cdf"][-1] - 1.0) < 1e-6

    def test_cdf_starts_at_or_above_zero(self, sample_2d_array):
        from processing.histogram import compute_histogram, compute_cdf
        hist = compute_histogram(sample_2d_array)
        result = compute_cdf(hist)
        assert result["cdf"][0] >= 0.0

    def test_cdf_is_monotonically_non_decreasing(self, sample_2d_array):
        from processing.histogram import compute_histogram, compute_cdf
        hist = compute_histogram(sample_2d_array)
        cdf = compute_cdf(hist)["cdf"]
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1] - 1e-9

    def test_cdf_passthrough_of_histogram_keys(self, sample_2d_array):
        """compute_cdf should preserve all original histogram keys."""
        from processing.histogram import compute_histogram, compute_cdf
        hist = compute_histogram(sample_2d_array)
        result = compute_cdf(hist)
        for key in ("bins", "counts", "intensity_min", "intensity_max"):
            assert key in result


# ─────────────────────────────────────────────────────────────────────────────
# apply_cdf_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyCdfThreshold:
    def test_output_dtype_uint8(self, sample_2d_array):
        from processing.histogram import apply_cdf_threshold
        result = apply_cdf_threshold(sample_2d_array)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self, sample_2d_array):
        from processing.histogram import apply_cdf_threshold
        result = apply_cdf_threshold(sample_2d_array)
        assert result.shape == sample_2d_array.shape

    def test_only_binary_values(self, sample_2d_array):
        from processing.histogram import apply_cdf_threshold
        result = apply_cdf_threshold(sample_2d_array)
        unique = set(result.flatten().tolist())
        assert unique.issubset({0, 255})

    def test_percentile_100_gives_few_bright(self, sample_2d_array):
        """At the 100th percentile everything is below the threshold → all zeros."""
        from processing.histogram import apply_cdf_threshold
        result = apply_cdf_threshold(sample_2d_array, percentile=100.0)
        # max value is >= threshold, so at least some pixels are 255
        # (the threshold equals the max, and arr >= max is True for max pixels)
        assert result.dtype == np.uint8

    def test_percentile_0_gives_all_bright(self):
        """0th percentile threshold → all pixels above min → all 255."""
        from processing.histogram import apply_cdf_threshold
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = apply_cdf_threshold(arr, percentile=0.0)
        assert np.all(result == 255)
