import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api.deps import get_current_user
from main import app


@pytest.fixture()
def client():
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
