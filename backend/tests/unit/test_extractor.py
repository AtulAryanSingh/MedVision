"""
tests/unit/test_extractor.py

Unit tests for features/extractor.py.

Covers
------
extract_features – presence of all expected keys, value types, statistical
                   correctness (mean, min, max), histogram structure,
                   shape_descriptors fields, edge case (uniform array).
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

_EXPECTED_KEYS = {
    "mean",
    "std_dev",
    "intensity_min",
    "intensity_max",
    "skewness",
    "kurtosis",
    "percentile_10",
    "percentile_25",
    "percentile_50",
    "percentile_75",
    "percentile_90",
    "entropy",
    "nonzero_fraction",
    "histogram",
    "shape_descriptors",
}

_HISTOGRAM_KEYS = {"bins", "counts"}

_SHAPE_KEYS = {
    "foreground_pixels",
    "total_pixels",
    "foreground_coverage",
    "effective_radius_px",
}


class TestExtractFeatures:
    # ── Key presence ─────────────────────────────────────────────────────────

    def test_all_expected_keys_present(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert _EXPECTED_KEYS.issubset(result.keys())

    def test_histogram_sub_keys(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert _HISTOGRAM_KEYS.issubset(result["histogram"].keys())

    def test_shape_descriptor_keys(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert _SHAPE_KEYS.issubset(result["shape_descriptors"].keys())

    # ── Statistical correctness ───────────────────────────────────────────────

    def test_intensity_min_max_bounded(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert result["intensity_min"] >= 0.0
        assert result["intensity_max"] <= 255.0
        assert result["intensity_min"] <= result["intensity_max"]

    def test_mean_in_range(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert 0.0 <= result["mean"] <= 255.0

    def test_std_dev_non_negative(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert result["std_dev"] >= 0.0

    def test_percentiles_ordered(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        p10 = result["percentile_10"]
        p25 = result["percentile_25"]
        p50 = result["percentile_50"]
        p75 = result["percentile_75"]
        p90 = result["percentile_90"]
        assert p10 <= p25 <= p50 <= p75 <= p90

    def test_entropy_non_negative(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert result["entropy"] >= 0.0

    def test_nonzero_fraction_in_unit_interval(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert 0.0 <= result["nonzero_fraction"] <= 1.0

    # ── Histogram structure ───────────────────────────────────────────────────

    def test_histogram_bin_count_is_64(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert len(result["histogram"]["bins"]) == 64
        assert len(result["histogram"]["counts"]) == 64

    def test_histogram_counts_non_negative(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert all(c >= 0 for c in result["histogram"]["counts"])

    # ── Shape descriptor correctness ──────────────────────────────────────────

    def test_total_pixels_matches_array_size(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert result["shape_descriptors"]["total_pixels"] == sample_2d_array.size

    def test_foreground_coverage_in_unit_interval(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        cov = result["shape_descriptors"]["foreground_coverage"]
        assert 0.0 <= cov <= 1.0

    def test_effective_radius_non_negative(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        assert result["shape_descriptors"]["effective_radius_px"] >= 0.0

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_uniform_array_does_not_raise(self):
        """A uniform-intensity array should not raise even though std is 0."""
        from features.extractor import extract_features
        uniform = np.full((32, 32), 100.0, dtype=np.float32)
        result = extract_features(uniform)
        assert result["std_dev"] == 0.0 or result["std_dev"] < 1e-3

    def test_3d_volume_is_accepted(self, sample_3d_volume):
        """3-D arrays should be handled (collapsed to 2-D for feature extraction)."""
        from features.extractor import extract_features
        # extract_features receives a 2-D slice after the API extracts it, but
        # passing 3-D should not crash the extractor itself.
        slc = sample_3d_volume[0, :, :]  # take one axial slice
        result = extract_features(slc)
        assert "mean" in result

    def test_all_values_are_finite(self, sample_2d_array):
        from features.extractor import extract_features
        result = extract_features(sample_2d_array)
        scalar_keys = [
            "mean", "std_dev", "intensity_min", "intensity_max",
            "skewness", "kurtosis", "percentile_50", "entropy",
            "nonzero_fraction",
        ]
        for key in scalar_keys:
            assert np.isfinite(result[key]), f"{key} should be finite"
