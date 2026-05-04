"""
tests/integration/test_upload.py

Integration tests for POST /api/upload and POST /api/upload-series.

Covers
------
• Upload a valid PNG       → 200, image_id present, metadata correct
• Upload a valid NIfTI     → 200, is_3d True, spacing list of 3 values
• Unsupported extension    → 400
• Empty file               → 400
• Invalid content (PNG ext, garbage bytes) → 422
• upload-series – not-dcm file → 400
• upload-series – empty files list → 400

The `client` fixture in conftest.py redirects upload/cache dirs to tmp_path
so no files are written to the real backend/data directory.
"""

import io
import struct

import numpy as np
import pytest


def _make_png_bytes(width=16, height=16) -> bytes:
    """Generate a valid minimal grayscale PNG in memory using numpy + cv2."""
    import cv2

    arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def _make_nifti_bytes() -> bytes:
    """Generate a minimal NIfTI-1 file in memory."""
    import tempfile
    import nibabel as nib

    arr = np.zeros((8, 8, 4), dtype=np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(arr, affine)
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        tmp_path = f.name
    nib.save(img, tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    import os
    os.unlink(tmp_path)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload — PNG
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadPng:
    def test_status_200(self, client):
        png_bytes = _make_png_bytes()
        response = client.post(
            "/api/upload",
            files={"file": ("image.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_response_contains_image_id(self, client):
        png_bytes = _make_png_bytes()
        response = client.post(
            "/api/upload",
            files={"file": ("image.png", png_bytes, "image/png")},
        )
        data = response.json()
        assert "image_id" in data
        # image_id must be a valid UUID string
        import uuid
        uuid.UUID(data["image_id"])  # raises if invalid

    def test_response_metadata_fields(self, client):
        png_bytes = _make_png_bytes()
        response = client.post(
            "/api/upload",
            files={"file": ("image.png", png_bytes, "image/png")},
        )
        data = response.json()
        for field in ("shape", "ndim", "intensity_min", "intensity_max", "size_bytes"):
            assert field in data, f"Missing field: {field}"

    def test_response_is_not_3d(self, client):
        """PNG uploads should not be flagged as 3-D volumes."""
        png_bytes = _make_png_bytes()
        response = client.post(
            "/api/upload",
            files={"file": ("image.png", png_bytes, "image/png")},
        )
        data = response.json()
        assert data["is_3d"] is False


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload — NIfTI
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadNifti:
    pytest.importorskip("nibabel")

    def test_nifti_upload_status_200(self, client):
        nifti_bytes = _make_nifti_bytes()
        response = client.post(
            "/api/upload",
            files={"file": ("volume.nii", nifti_bytes, "application/octet-stream")},
        )
        assert response.status_code == 200

    def test_nifti_is_3d(self, client):
        nifti_bytes = _make_nifti_bytes()
        response = client.post(
            "/api/upload",
            files={"file": ("volume.nii", nifti_bytes, "application/octet-stream")},
        )
        data = response.json()
        assert data["is_3d"] is True

    def test_nifti_spacing_has_three_values(self, client):
        nifti_bytes = _make_nifti_bytes()
        response = client.post(
            "/api/upload",
            files={"file": ("volume.nii", nifti_bytes, "application/octet-stream")},
        )
        data = response.json()
        assert "spacing" in data
        assert len(data["spacing"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload — Error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadErrors:
    def test_unsupported_extension_returns_400(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("document.pdf", b"fake content", "application/pdf")},
        )
        assert response.status_code == 400

    def test_empty_file_returns_400(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("image.png", b"", "image/png")},
        )
        assert response.status_code == 400

    def test_garbage_bytes_with_png_extension_returns_422(self, client):
        """A file with a .png extension but non-PNG bytes should fail parsing."""
        response = client.post(
            "/api/upload",
            files={"file": ("image.png", b"THIS IS NOT A PNG", "image/png")},
        )
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload-series
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadSeries:
    def test_non_dcm_file_returns_400(self, client):
        response = client.post(
            "/api/upload-series",
            files=[("files", ("image.png", b"fake", "image/png"))],
        )
        assert response.status_code == 400

    def test_empty_file_list_returns_400(self, client):
        """Sending no files should return 422 (FastAPI validation) or 400."""
        # FastAPI requires at least one file; no files → 422 validation error
        response = client.post("/api/upload-series", files=[])
        assert response.status_code in (400, 422)
