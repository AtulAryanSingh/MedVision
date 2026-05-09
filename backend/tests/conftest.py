"""
tests/conftest.py

Shared pytest fixtures for the MedVision backend test suite.

Fixtures provided
-----------------
sample_2d_array     – 64×64 float32 gradient array (no disk I/O)
sample_3d_volume    – 8×32×32 float32 volume (no disk I/O)
png_file            – minimal PNG written to tmp_path
nifti_file          – minimal NIfTI-1 (.nii.gz) written to tmp_path
client              – FastAPI TestClient with auth dependency bypassed
anon_client         – FastAPI TestClient with real auth (no bypass)
auth_headers        – {"Authorization": "Bearer <signed-test-JWT>"} dict
"""

from datetime import datetime, timedelta, timezone
import os
import sys

import jwt
import numpy as np
import pytest

# ── JWT secret must be set before any app import so that api/deps.py picks it
# up at module-load time (it reads os.environ at import).  Using setdefault
# means a value already set in the environment (e.g. from a .env in CI) wins.
_DEFAULT_TEST_JWT_SECRET = "medvision-pytest-secret-do-not-use-in-production"
_ACTIVE_TEST_JWT_SECRET = os.environ.setdefault("JWT_SECRET", _DEFAULT_TEST_JWT_SECRET)

# ── Ensure the backend package root is importable from any working directory
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Array / volume fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_2d_array() -> np.ndarray:
    """64×64 float32 array with a smooth gradient — representative of a 2-D slice."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64)).astype(np.float32) * 255.0


@pytest.fixture
def sample_3d_volume() -> np.ndarray:
    """8×32×32 float32 volume in (Z, Y, X) layout — minimal 3-D test volume."""
    rng = np.random.default_rng(42)
    return rng.random((8, 32, 32)).astype(np.float32) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# On-disk file fixtures (generated synthetically — no large binaries committed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def png_file(tmp_path, sample_2d_array):
    """Write a small grayscale PNG to *tmp_path* and return its path."""
    import cv2

    path = str(tmp_path / "test.png")
    arr_u8 = np.clip(sample_2d_array, 0, 255).astype(np.uint8)
    cv2.imwrite(path, arr_u8)
    return path


@pytest.fixture
def nifti_file(tmp_path, sample_3d_volume):
    """
    Write a minimal NIfTI-1 (.nii.gz) file to *tmp_path* and return its path.

    nibabel stores arrays as (X, Y, Z) internally, so the (Z, Y, X) test
    volume is transposed before saving.  The affine encodes 1.5×1.5×2.0 mm
    voxel spacing via the diagonal.
    """
    import nibabel as nib

    # Transpose (Z=8, Y=32, X=32) → (X=32, Y=32, Z=8) for nibabel convention
    arr_nib = np.transpose(sample_3d_volume, (2, 1, 0)).astype(np.float32)
    affine = np.diag([1.5, 1.5, 2.0, 1.0])  # 1.5×1.5×2.0 mm voxels
    img = nib.Nifti1Image(arr_nib, affine)
    path = str(tmp_path / "test.nii.gz")
    nib.save(img, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Auth fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def auth_headers():
    """
    Return an Authorization header dict containing a valid signed test JWT.

    The token is signed with the active JWT secret from os.environ so
    api/deps.get_current_user will accept it in both local and CI runs.
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=24)
    payload = {"sub": "testuser", "exp": expire}
    token = jwt.encode(payload, _ACTIVE_TEST_JWT_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI TestClient fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    FastAPI TestClient with:
      • get_current_user dependency overridden (no real JWT required)
      • upload and cache directories redirected to *tmp_path* so tests do not
        pollute or depend on the real backend/data/ directories

    The override is cleaned up after each test via monkeypatch teardown.
    """
    import api as api_pkg
    import api.upload as upload_mod
    import api.process as process_mod
    import api.features as features_mod

    from fastapi.testclient import TestClient
    from main import app as _app
    from api.deps import get_current_user

    upload_dir = str(tmp_path / "uploads")
    cache_dir = str(tmp_path / "cache")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Patch directory helpers in api/__init__ (used by find_uploaded_file,
    # save_metadata, load_metadata, update_cache)
    monkeypatch.setattr(api_pkg, "get_upload_dir", lambda: upload_dir)
    monkeypatch.setattr(api_pkg, "get_cache_dir", lambda: cache_dir)

    # Patch the names that were imported-by-value into each router module
    monkeypatch.setattr(upload_mod, "get_upload_dir", lambda: upload_dir)
    monkeypatch.setattr(process_mod, "find_uploaded_file",
                        api_pkg.find_uploaded_file)
    monkeypatch.setattr(features_mod, "find_uploaded_file",
                        api_pkg.find_uploaded_file)

    # Bypass JWT auth for all protected endpoints
    _app.dependency_overrides[get_current_user] = lambda: {"sub": "testuser"}

    with TestClient(_app, raise_server_exceptions=True) as c:
        yield c

    _app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def anon_client():
    """
    FastAPI TestClient *without* any auth override — used by auth tests that
    need to exercise the real JWT validation path.
    """
    from fastapi.testclient import TestClient
    from main import app as _app

    with TestClient(_app, raise_server_exceptions=False) as c:
        yield c
