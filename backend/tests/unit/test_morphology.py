"""
tests/unit/test_morphology.py

Unit tests for processing/morphology.py.

Covers
------
apply_erosion               – shape, float32 output, erosion shrinks foreground
apply_dilation              – shape, float32 output, dilation expands foreground
apply_opening               – shape, float32 output
apply_closing               – shape, float32 output
label_connected_components  – component count, stats keys, colour image shape
draw_bounding_boxes         – shape, dtype, returns canvas with no error
compute_center_of_mass      – keys "row" and "col", values are finite floats
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")


def _two_blob_array() -> np.ndarray:
    """
    64×64 array with two distinct bright blobs separated by dark background.
    Used to test connected-component labelling.
    """
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[5:20, 5:20] = 200.0   # blob 1  (15×15 pixels)
    arr[40:55, 40:55] = 200.0  # blob 2  (15×15 pixels)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# apply_erosion
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyErosion:
    def test_shape_preserved(self, sample_2d_array):
        from processing.morphology import apply_erosion
        result = apply_erosion(sample_2d_array)
        assert result.shape == sample_2d_array.shape

    def test_output_dtype_float32(self, sample_2d_array):
        from processing.morphology import apply_erosion
        result = apply_erosion(sample_2d_array)
        assert result.dtype == np.float32

    def test_erosion_reduces_bright_area(self):
        """Eroding a bright square should shrink its bright area."""
        from processing.morphology import apply_erosion
        blob = np.zeros((32, 32), dtype=np.float32)
        blob[10:22, 10:22] = 255.0
        eroded = apply_erosion(blob, kernel_size=5)
        bright_before = np.count_nonzero(blob)
        bright_after = np.count_nonzero(eroded)
        assert bright_after <= bright_before


# ─────────────────────────────────────────────────────────────────────────────
# apply_dilation
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyDilation:
    def test_shape_preserved(self, sample_2d_array):
        from processing.morphology import apply_dilation
        result = apply_dilation(sample_2d_array)
        assert result.shape == sample_2d_array.shape

    def test_output_dtype_float32(self, sample_2d_array):
        from processing.morphology import apply_dilation
        result = apply_dilation(sample_2d_array)
        assert result.dtype == np.float32

    def test_dilation_expands_bright_area(self):
        """Dilating a bright square should expand its bright area."""
        from processing.morphology import apply_dilation
        blob = np.zeros((32, 32), dtype=np.float32)
        blob[10:22, 10:22] = 255.0
        dilated = apply_dilation(blob, kernel_size=5)
        bright_before = np.count_nonzero(blob)
        bright_after = np.count_nonzero(dilated)
        assert bright_after >= bright_before


# ─────────────────────────────────────────────────────────────────────────────
# apply_opening
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyOpening:
    def test_shape_preserved(self, sample_2d_array):
        from processing.morphology import apply_opening
        result = apply_opening(sample_2d_array)
        assert result.shape == sample_2d_array.shape

    def test_output_dtype_float32(self, sample_2d_array):
        from processing.morphology import apply_opening
        result = apply_opening(sample_2d_array)
        assert result.dtype == np.float32

    def test_opening_removes_small_blobs(self):
        """Opening with a large kernel should remove isolated single pixels."""
        from processing.morphology import apply_opening
        base = np.zeros((32, 32), dtype=np.float32)
        base[5, 5] = 255.0  # isolated pixel
        result = apply_opening(base, kernel_size=7)
        assert result[5, 5] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# apply_closing
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyClosing:
    def test_shape_preserved(self, sample_2d_array):
        from processing.morphology import apply_closing
        result = apply_closing(sample_2d_array)
        assert result.shape == sample_2d_array.shape

    def test_output_dtype_float32(self, sample_2d_array):
        from processing.morphology import apply_closing
        result = apply_closing(sample_2d_array)
        assert result.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# label_connected_components
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelConnectedComponents:
    def test_colour_image_shape(self):
        from processing.morphology import label_connected_components
        arr = _two_blob_array()
        colour_img, _ = label_connected_components(arr, threshold=128.0, min_area=10)
        assert colour_img.shape == (64, 64, 3)

    def test_colour_image_dtype_uint8(self):
        from processing.morphology import label_connected_components
        arr = _two_blob_array()
        colour_img, _ = label_connected_components(arr, threshold=128.0, min_area=10)
        assert colour_img.dtype == np.uint8

    def test_detects_two_components(self):
        from processing.morphology import label_connected_components
        arr = _two_blob_array()
        _, components = label_connected_components(arr, threshold=128.0, min_area=10)
        assert len(components) == 2

    def test_component_stats_keys(self):
        from processing.morphology import label_connected_components
        arr = _two_blob_array()
        _, components = label_connected_components(arr, threshold=128.0, min_area=10)
        for comp in components:
            assert "label" in comp
            assert "area" in comp
            assert "bounding_box" in comp
            assert "center" in comp

    def test_min_area_filter(self):
        """Components smaller than min_area should be excluded."""
        from processing.morphology import label_connected_components
        arr = np.zeros((32, 32), dtype=np.float32)
        arr[5, 5] = 255.0  # 1-pixel component
        _, components = label_connected_components(arr, threshold=128.0, min_area=50)
        assert len(components) == 0


# ─────────────────────────────────────────────────────────────────────────────
# draw_bounding_boxes
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawBoundingBoxes:
    def test_output_shape_is_3channel(self):
        from processing.morphology import draw_bounding_boxes
        arr = _two_blob_array()
        result = draw_bounding_boxes(arr, threshold=128.0)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_dtype_float32(self):
        from processing.morphology import draw_bounding_boxes
        arr = _two_blob_array()
        result = draw_bounding_boxes(arr, threshold=128.0)
        assert result.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# compute_center_of_mass
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCenterOfMass:
    def test_returns_dict_with_row_col(self, sample_2d_array):
        from processing.morphology import compute_center_of_mass
        result = compute_center_of_mass(sample_2d_array)
        assert "row" in result
        assert "col" in result

    def test_values_are_finite_floats(self, sample_2d_array):
        from processing.morphology import compute_center_of_mass
        result = compute_center_of_mass(sample_2d_array)
        assert np.isfinite(result["row"])
        assert np.isfinite(result["col"])

    def test_centroid_of_centred_bright_spot(self):
        """A bright spot at (16, 16) should yield centroid close to (16, 16)."""
        from processing.morphology import compute_center_of_mass
        arr = np.zeros((32, 32), dtype=np.float32)
        arr[14:18, 14:18] = 255.0  # 4×4 patch centred near (16, 16)
        result = compute_center_of_mass(arr)
        assert abs(result["row"] - 15.5) < 2.0
        assert abs(result["col"] - 15.5) < 2.0
