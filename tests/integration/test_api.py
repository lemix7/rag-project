import pytest
from fastapi.testclient import TestClient

from rag.api import dependencies
from rag.core.config import Settings


@pytest.fixture
def test_settings():
    return Settings(
        openai_api_key="test-key",
        api_key="test-api-key",
        vector_store_type="chroma",
        chroma_persist_dir="/tmp/test_chroma_api",
        chroma_collection="test_api",
        log_format="console",
    )


@pytest.fixture
def client(test_settings):
    from rag.api.app import create_app

    dependencies.reset_dependencies()
    dependencies.set_settings(test_settings)

    app = create_app(settings=test_settings)
    with TestClient(app) as c:
        yield c

    dependencies.reset_dependencies()


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_supported_types_with_auth(client):
    response = client.get(
        "/api/v1/documents/supported-types",
        headers={"X-API-Key": "test-api-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert ".pdf" in data["supported_types"]


def test_query_requires_auth(client):
    response = client.post(
        "/api/v1/query",
        json={"question": "test question"},
    )
    assert response.status_code == 401


def test_query_rejects_invalid_key(client):
    response = client.post(
        "/api/v1/query",
        json={"question": "test question"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_upload_rejects_unsupported_type(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.exe", b"fake content", "application/octet-stream")},
        headers={"X-API-Key": "test-api-key"},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_query_validates_input(client):
    response = client.post(
        "/api/v1/query",
        json={"question": "", "k": 0},
        headers={"X-API-Key": "test-api-key"},
    )
    assert response.status_code == 422
