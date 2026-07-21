"""Integration tests for the FastAPI app.

These spin up the real app (which loads the SentenceTransformer embedding
model and a persistent Chroma store). In CI, the model should be
pre-cached so this runs fully; in fully offline/sandboxed environments
without the model cached, these tests are skipped rather than failing
the whole suite.
"""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():
    try:
        from backend.main import app
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not initialize app (likely no network for model download): {exc}")

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_register_and_login(client):
    resp = client.post("/api/auth/register", json={"username": "testuser", "password": "testpass123"})
    assert resp.status_code == 200

    resp = client.post("/api/auth/login", json={"username": "testuser", "password": "testpass123"})
    assert resp.status_code == 200
    assert "token" in resp.json()


def test_login_wrong_password_rejected(client):
    client.post("/api/auth/register", json={"username": "testuser2", "password": "correctpass"})
    resp = client.post("/api/auth/login", json={"username": "testuser2", "password": "wrongpass"})
    assert resp.status_code == 401


def test_upload_and_chat_flow(client, tmp_path):
    # Register and login first
    client.post("/api/auth/register", json={"username": "testuser3", "password": "testpass123"})
    login_resp = client.post("/api/auth/login", json={"username": "testuser3", "password": "testpass123"})
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    doc_path = tmp_path / "sample.txt"
    doc_path.write_text("The company reported revenue of $482 million in Q3, up 14% year-over-year.")

    with doc_path.open("rb") as f:
        resp = client.post("/api/upload", files={"file": ("sample.txt", f, "text/plain")}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["chunks_created"] >= 1

    resp = client.post(
        "/api/chat", json={"session_id": "test-session", "message": "What was Q3 revenue?"}, headers=headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["route"] in {"retrieval", "tool", "direct", "hybrid"}


def test_metrics_endpoint_shape(client):
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "documents_uploaded" in body
    assert "queries_by_route" in body
