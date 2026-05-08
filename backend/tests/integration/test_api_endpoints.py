import uuid

import numpy as np

import api.features as features_api
import api.process as process_api
import api.upload as upload_api


def test_upload_endpoint_with_mocked_loader(client, monkeypatch, tmp_path):
    saved = {}

    monkeypatch.setattr(upload_api, "get_upload_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        upload_api,
        "load_image",
        lambda _path: (
            np.ones((8, 8), dtype=np.float32),
            {
                "file_type": "png_jpg",
                "shape": [8, 8],
                "dtype_str": "float32",
                "ndim": 2,
                "is_3d": False,
                "intensity_min": 1.0,
                "intensity_max": 1.0,
                "spacing": [1.0, 1.0, 1.0],
                "modality": "unknown",
                "extra_meta": {},
            },
        ),
    )

    def mock_save_metadata(image_id, metadata):
        saved["image_id"] = image_id
        saved["metadata"] = metadata

    monkeypatch.setattr(upload_api, "save_metadata", mock_save_metadata)

    resp = client.post(
        "/api/upload",
        files={"file": ("scan.png", b"fake-png-data", "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "image_id" in body
    assert body["shape"] == [8, 8]
    assert saved["image_id"] == body["image_id"]


def test_features_endpoint_with_mocked_loader(client, monkeypatch):
    image_id = str(uuid.uuid4())
    updated = {}

    monkeypatch.setattr(features_api, "load_metadata", lambda _image_id: {"is_3d": False})
    monkeypatch.setattr(features_api, "find_uploaded_file", lambda _image_id: "/tmp/fake-image")

    async def fake_async_load_image(_path):
        return np.arange(64, dtype=np.float32).reshape(8, 8), {"is_3d": False}

    monkeypatch.setattr(features_api, "async_load_image", fake_async_load_image)
    monkeypatch.setattr(features_api, "update_cache", lambda _id, data: updated.update(data))

    resp = client.post("/api/features", json={"image_id": image_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_id"] == image_id
    assert "mean" in body
    assert "features" in updated


def test_process_endpoint_with_mocked_loader(client, monkeypatch):
    image_id = str(uuid.uuid4())
    updated = {}

    monkeypatch.setattr(process_api, "load_metadata", lambda _image_id: {"is_3d": False})
    monkeypatch.setattr(process_api, "find_uploaded_file", lambda _image_id: "/tmp/fake-image")

    async def fake_async_load_image(_path):
        return np.arange(100, dtype=np.float32).reshape(10, 10), {"is_3d": False}

    monkeypatch.setattr(process_api, "async_load_image", fake_async_load_image)
    monkeypatch.setattr(process_api, "update_cache", lambda _id, data: updated.update(data))

    resp = client.post(
        "/api/process",
        json={"image_id": image_id, "processing_type": "gaussian", "sigma": 1.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_id"] == image_id
    assert body["processing_type"] == "gaussian"
    assert body["result_image"].startswith("data:image/png;base64,")
    assert updated["last_processing"]["type"] == "gaussian"
