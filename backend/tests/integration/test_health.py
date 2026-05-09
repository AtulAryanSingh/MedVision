"""
tests/integration/test_health.py

Integration tests for the unprotected health / root endpoints.

GET /        – should return 200 with status "ok"
GET /health  – should return 200 with status "ok"
"""

class TestHealthEndpoint:
    def test_root_returns_200(self, anon_client):
        response = anon_client.get("/")
        assert response.status_code == 200

    def test_root_status_ok(self, anon_client):
        response = anon_client.get("/")
        data = response.json()
        assert data["status"] == "ok"

    def test_root_contains_version(self, anon_client):
        response = anon_client.get("/")
        data = response.json()
        assert "version" in data

    def test_health_returns_200(self, anon_client):
        response = anon_client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, anon_client):
        response = anon_client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_contains_modules(self, anon_client):
        response = anon_client.get("/health")
        data = response.json()
        assert "modules" in data
        assert isinstance(data["modules"], list)
        assert len(data["modules"]) > 0
