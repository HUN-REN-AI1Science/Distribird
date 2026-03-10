"""Tests for FastAPI endpoints."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure test credentials match the defaults in config.py
os.environ.setdefault("DISTRIBIRD_AUTH_USERNAME", "demo")
os.environ.setdefault("DISTRIBIRD_AUTH_PASSWORD", "changeme")

from distribird.api.routes import app
from distribird.models import (
    ConfidenceLevel,
    DistributionFamily,
    FittedPrior,
    ParameterInput,
    PipelineResult,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_result():
    return PipelineResult(
        parameter=ParameterInput(name="test", description="test param"),
        prior=FittedPrior(
            parameter_name="test",
            family=DistributionFamily.NORMAL,
            params={"mu": 5.0, "sigma": 1.0},
            confidence=ConfidenceLevel.HIGH,
            is_informative=True,
        ),
        papers_found=3,
        values_extracted=5,
    )


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_parameter_requires_auth(client):
    resp = client.post(
        "/api/v1/parameter",
        json={"name": "test", "description": "test param"},
    )
    assert resp.status_code == 401


@patch("distribird.api.routes.run_parameter", new_callable=AsyncMock)
def test_process_parameter(mock_run, client, mock_result):
    mock_run.return_value = mock_result
    resp = client.post(
        "/api/v1/parameter",
        json={"name": "test", "description": "test param"},
        auth=("demo", "changeme"),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["parameter"] == "test"
    assert data["distribution"] == "normal"
