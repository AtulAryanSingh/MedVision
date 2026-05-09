"""
tests/integration/test_auth.py

Integration tests for POST /api/auth/login.

Covers
------
• Correct credentials → 200 + JWT token returned
• Wrong password → 401
• Non-existent user → 401
• Missing fields → 422 (FastAPI form validation)
• Protected endpoint without token → 401
• Protected endpoint with invalid token → 401
• Protected endpoint with valid token → passes auth (delegates to health)
"""

class TestLogin:
    def test_valid_credentials_return_200(self, anon_client, monkeypatch):
        """
        Monkeypatch db.get_user and db.verify_password so the test does not
        depend on the real SQLite database or a pre-existing user.
        """
        import db

        fake_user = {"username": "testuser", "hashed_password": "hashed"}
        monkeypatch.setattr(db, "get_user", lambda _: fake_user)
        monkeypatch.setattr(db, "verify_password", lambda _plain, _hashed: True)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "correctpassword"},
        )
        assert response.status_code == 200

    def test_valid_credentials_return_token(self, anon_client, monkeypatch):
        import db

        fake_user = {"username": "testuser", "hashed_password": "hashed"}
        monkeypatch.setattr(db, "get_user", lambda _: fake_user)
        monkeypatch.setattr(db, "verify_password", lambda _plain, _hashed: True)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "correctpassword"},
        )
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 10  # a non-trivial JWT

    def test_wrong_password_returns_401(self, anon_client, monkeypatch):
        import db

        fake_user = {"username": "testuser", "hashed_password": "hashed"}
        monkeypatch.setattr(db, "get_user", lambda _: fake_user)
        monkeypatch.setattr(db, "verify_password", lambda _plain, _hashed: False)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_unknown_user_returns_401(self, anon_client, monkeypatch):
        import db

        monkeypatch.setattr(db, "get_user", lambda _: None)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "ghost", "password": "anything"},
        )
        assert response.status_code == 401

    def test_missing_username_returns_422(self, anon_client):
        response = anon_client.post(
            "/api/auth/login",
            data={"password": "nouser"},
        )
        assert response.status_code == 422

    def test_missing_password_returns_422(self, anon_client):
        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser"},
        )
        assert response.status_code == 422


class TestProtectedEndpointAuth:
    def test_no_token_returns_401(self, anon_client):
        """Calling a protected endpoint without a token should yield 401."""
        response = anon_client.get("/api/preview/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, anon_client):
        response = anon_client.get(
            "/api/preview/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": "Bearer this.is.not.valid"},
        )
        assert response.status_code == 401

    def test_valid_token_passes_auth(self, anon_client, auth_headers):
        """
        A correctly signed token should pass auth (endpoint may still return
        404 because no image exists, but must NOT return 401 or 403).
        """
        response = anon_client.get(
            "/api/preview/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert response.status_code != 401
        assert response.status_code != 403
