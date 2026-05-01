"""Tests for the Tourism SuperAI API."""

import pytest
from fastapi.testclient import TestClient

from src.tourism_superai.api import app


@pytest.fixture
def client():
    """Test client for the API."""
    return TestClient(app)


def test_health(client):
    """Test the health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"