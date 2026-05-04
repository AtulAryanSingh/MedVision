"""
tests/unit/test_loader.py

Unit tests for core/loader.py.

Covers
------
normalise_to_uint8   – value range, output dtype, flat-input edge case
get_slice_2d         – 2-D pass-through, 3-D axial/coronal/sagittal, mid-index
array_to_base64_png  – valid base64 output, decodable as PNG
load_image (PNG)     – loads a synthetic PNG from disk via monkeypatched path
load_image (NIfTI)   – loads a synthetic .nii.gz and checks shape/metadata
"""

import base64
import io

import numpy as np
import pytest

pytest.importorskip("cv2")  # skip entire module if OpenCV is unavailable


# ─────────────────────────────────────────────────────────────────────────────
# normalise_to_uint8
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseToUint8:
    def test_output_dtype(self, sample_2d_array):
        from core.loader import normalise_to_uint8
        result = normalise_to_uint8(sample_2d_array)
        assert result.dtype == np.uint8

    def test_output_range(self, sample_2d_array):
        from core.loader import normalise_to_uint8
        result = normalise_to_uint8(sample_2d_array)
        assert result.min() == 0
        assert result.max() == 255

    def test_shape_preserved(self, sample_2d_array):
        from core.loader import normalise_to_uint8
        result = normalise_to_uint8(sample_2d_array)
        assert result.shape == sample_2d_array.shape

    def test_flat_input_returns_zeros(self):
        """An array of constant intensity should not raise and returns zeros."""
        from core.loader import normalise_to_uint8
        flat = np.full((4, 4), 42.0, dtype=np.float32)
        result = normalise_to_uint8(flat)
        assert result.dtype == np.uint8
        assert np.all(result == 0)

    def test_3d_volume(self, sample_3d_volume):
        from core.loader import normalise_to_uint8
        result = normalise_to_uint8(sample_3d_volume)
        assert result.dtype == np.uint8
        assert result.shape == sample_3d_volume.shape


# ─────────────────────────────────────────────────────────────────────────────
# get_slice_2d
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSlice2D:
    def test_2d_input_passthrough(self, sample_2d_array):
        from core.loader import get_slice_2d
        result = get_slice_2d(sample_2d_array, axis=0)
        assert result.shape == sample_2d_array.shape

    def test_axial_slice_shape(self, sample_3d_volume):
        """Axial (axis=0) slice should have shape (Y, X) == (32, 32)."""
        from core.loader import get_slice_2d
        result = get_slice_2d(sample_3d_volume, axis=0)
        assert result.shape == (32, 32)

    def test_coronal_slice_shape(self, sample_3d_volume):
        """Coronal (axis=1) slice should have shape (Z, X) == (8, 32)."""
        from core.loader import get_slice_2d
        result = get_slice_2d(sample_3d_volume, axis=1)
        assert result.shape == (8, 32)

    def test_sagittal_slice_shape(self, sample_3d_volume):
        """Sagittal (axis=2) slice should have shape (Z, Y) == (8, 32)."""
        from core.loader import get_slice_2d
        result = get_slice_2d(sample_3d_volume, axis=2)
        assert result.shape == (8, 32)

    def test_explicit_index(self, sample_3d_volume):
        """Explicit index should select the correct slice."""
        from core.loader import get_slice_2d
        # volume shape (8, 32, 32); request slice 3 along axis 0
        result = get_slice_2d(sample_3d_volume, axis=0, index=3)
        expected = sample_3d_volume[3, :, :]
        np.testing.assert_array_equal(result, expected)

    def test_default_index_is_midpoint(self, sample_3d_volume):
        """Default (None) index should return the middle slice."""
        from core.loader import get_slice_2d
        mid = sample_3d_volume.shape[0] // 2
        result = get_slice_2d(sample_3d_volume, axis=0)
        expected = sample_3d_volume[mid, :, :]
        np.testing.assert_array_equal(result, expected)

    def test_out_of_bounds_index_is_clamped(self, sample_3d_volume):
        """Out-of-bounds index should be clamped to the last valid slice."""
        from core.loader import get_slice_2d
        # shape[0] == 8, request index 100 → should clamp to 7
        result = get_slice_2d(sample_3d_volume, axis=0, index=100)
        expected = sample_3d_volume[7, :, :]
        np.testing.assert_array_equal(result, expected)


# ─────────────────────────────────────────────────────────────────────────────
# array_to_base64_png
# ─────────────────────────────────────────────────────────────────────────────

class TestArrayToBase64Png:
    def test_output_is_string(self, sample_2d_array):
        from core.loader import array_to_base64_png
        result = array_to_base64_png(sample_2d_array)
        assert isinstance(result, str)

    def test_output_is_valid_base64(self, sample_2d_array):
        from core.loader import array_to_base64_png
        result = array_to_base64_png(sample_2d_array)
        decoded = base64.b64decode(result)  # should not raise
        assert len(decoded) > 0

    def test_decoded_bytes_start_with_png_signature(self, sample_2d_array):
        from core.loader import array_to_base64_png
        result = array_to_base64_png(sample_2d_array)
        decoded = base64.b64decode(result)
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        assert decoded[:4] == b"\x89PNG"

    def test_works_on_3d_volume_slice(self, sample_3d_volume):
        from core.loader import array_to_base64_png, get_slice_2d
        slc = get_slice_2d(sample_3d_volume, axis=0)
        result = array_to_base64_png(slc)
        assert isinstance(result, str) and len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# load_image — PNG
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadImagePng:
    def test_returns_tuple(self, png_file):
        from core.loader import load_image
        result = load_image(png_file)
        assert isinstance(result, tuple) and len(result) == 2

    def test_array_dtype(self, png_file):
        from core.loader import load_image
        arr, _ = load_image(png_file)
        assert arr.dtype == np.float32

    def test_metadata_keys(self, png_file):
        from core.loader import load_image
        _, meta = load_image(png_file)
        for key in ("file_type", "shape", "ndim", "intensity_min", "intensity_max"):
            assert key in meta

    def test_file_type(self, png_file):
        from core.loader import load_image
        _, meta = load_image(png_file)
        assert meta["file_type"] in ("png", "standard", "jpg", "jpeg", "png_jpg")

    def test_intensity_range_is_finite(self, png_file):
        from core.loader import load_image
        arr, meta = load_image(png_file)
        assert np.isfinite(meta["intensity_min"])
        assert np.isfinite(meta["intensity_max"])
        assert meta["intensity_min"] <= meta["intensity_max"]


# ─────────────────────────────────────────────────────────────────────────────
# load_image — NIfTI
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadImageNifti:
    pytest.importorskip("nibabel")

    def test_returns_3d_volume(self, nifti_file):
        from core.loader import load_image
        arr, meta = load_image(nifti_file)
        assert arr.ndim == 3

    def test_volume_shape_zyx(self, nifti_file):
        """After transposing from nibabel's (X,Y,Z), shape should be (Z,Y,X)."""
        from core.loader import load_image
        arr, _ = load_image(nifti_file)
        # Our fixture: nibabel (X=32, Y=32, Z=8) → loader transposed to (Z=8, Y=32, X=32)
        assert arr.shape == (8, 32, 32)

    def test_metadata_is_3d(self, nifti_file):
        from core.loader import load_image
        _, meta = load_image(nifti_file)
        assert meta["is_3d"] is True

    def test_spacing_is_zyx_order(self, nifti_file):
        """spacing_zyx should be [sp_z, sp_y, sp_x] — i.e. [2.0, 1.5, 1.5]."""
        from core.loader import load_image
        _, meta = load_image(nifti_file)
        spacing = meta["spacing"]
        assert len(spacing) == 3
        # Affine diagonal was [1.5, 1.5, 2.0] → sp_x=1.5, sp_y=1.5, sp_z=2.0
        # Loader stores as [sp_z, sp_y, sp_x]
        assert abs(spacing[0] - 2.0) < 0.01, "sp_z should be ≈2.0"
        assert abs(spacing[1] - 1.5) < 0.01, "sp_y should be ≈1.5"
        assert abs(spacing[2] - 1.5) < 0.01, "sp_x should be ≈1.5"

    def test_file_type_is_nifti(self, nifti_file):
        from core.loader import load_image
        _, meta = load_image(nifti_file)
        assert meta["file_type"] == "nifti"
