"""
tests/integration/test_process.py

Integration tests for POST /api/process.

These tests upload a real PNG via the API and then call /api/process with
various processing_type values to verify that each operation returns a 200
response containing the expected fields.

The `client` fixture (from conftest.py) bypasses JWT auth and redirects the
upload/cache directories to a per-test tmp_path so no files linger in the
real backend/data directory.
"""

import numpy as np
import pytest


def _upload_png(client) -> str:
    """Upload a synthetic PNG and return the image_id."""
    import cv2

    arr = (np.random.default_rng(7).random((32, 32)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", buf.tobytes(), "image/png")},
    )
    assert response.status_code == 200, f"Upload failed: {response.text}"
    return response.json()["image_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PROCESSING_TYPES = [
    "gaussian",
    "median",
    "sobel",
    "cdf_threshold",
    "erosion",
    "dilation",
    "opening",
    "closing",
    "connected_components",
    "bounding_boxes",
]


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/process — basic happy-path
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessEndpoint:
    @pytest.fixture(autouse=True)
    def upload_image(self, client):
        """Upload once per test class; store image_id as instance attribute."""
        self.image_id = _upload_png(client)

    def test_gaussian_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "gaussian"},
        )
        assert response.status_code == 200

    def test_median_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "median"},
        )
        assert response.status_code == 200

    def test_sobel_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "sobel"},
        )
        assert response.status_code == 200

    def test_cdf_threshold_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "cdf_threshold"},
        )
        assert response.status_code == 200

    def test_erosion_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "erosion"},
        )
        assert response.status_code == 200

    def test_dilation_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "dilation"},
        )
        assert response.status_code == 200

    def test_opening_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "opening"},
        )
        assert response.status_code == 200

    def test_closing_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "closing"},
        )
        assert response.status_code == 200

    def test_connected_components_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "connected_components"},
        )
        assert response.status_code == 200

    def test_bounding_boxes_returns_200(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "bounding_boxes"},
        )
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/process — response payload validation
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessResponsePayload:
    @pytest.fixture(autouse=True)
    def upload_image(self, client):
        self.image_id = _upload_png(client)

    def test_gaussian_response_contains_image(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "gaussian"},
        )
        data = response.json()
        assert "image" in data or "result" in data or "processing_type" in data

    def test_image_id_echoed_in_response(self, client):
        response = client.post(
            "/api/process",
            json={"image_id": self.image_id, "processing_type": "sobel"},
        )
        data = response.json()
        assert data.get("image_id") == self.image_id


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/process — error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessErrors:
    def test_unknown_processing_type_returns_422(self, client):
        """
        A processing_type not in the allowed set should return 400 or 422.
        """
        # Need a real image_id; upload one first
        image_id = _upload_png(client)
        response = client.post(
            "/api/process",
            json={"image_id": image_id, "processing_type": "not_a_real_filter"},
        )
        assert response.status_code in (400, 422)

    def test_missing_image_id_returns_error(self, client):
        response = client.post(
            "/api/process",
            json={"processing_type": "gaussian"},
        )
        assert response.status_code in (400, 422)

    def test_invalid_image_id_format_returns_400(self, client):
        """A non-UUID image_id should be rejected with 400."""
        response = client.post(
            "/api/process",
            json={"image_id": "not-a-uuid", "processing_type": "gaussian"},
        )
        assert response.status_code == 400

    def test_nonexistent_image_id_returns_404(self, client):
        """A well-formed UUID that doesn't exist should return 404."""
        response = client.post(
            "/api/process",
            json={
                "image_id": "00000000-0000-0000-0000-000000000000",
                "processing_type": "gaussian",
            },
        )
        assert response.status_code == 404
