"""Tests for export modules."""

import ast
import json

import pytest

from distribird.export.json_export import export_json, export_single_json
from distribird.export.python_export import export_python, export_single_python
from distribird.export.r_export import export_r, export_single_r
from distribird.models import (
    BatchResult,
    ConfidenceLevel,
    DistributionFamily,
    FittedPrior,
    LiteratureEvidence,
    ParameterInput,
    PipelineResult,
)


@pytest.fixture
def sample_result():
    prior = FittedPrior(
        parameter_name="max_lai",
        family=DistributionFamily.TRUNCATED_NORMAL,
        params={"mu": 5.0, "sigma": 1.5, "a": 0.0, "b": 12.0},
        confidence=ConfidenceLevel.HIGH,
        is_informative=True,
        reason="AIC-selected from 10 values",
        evidence=[
            LiteratureEvidence(title="Paper 1", doi="10.1234/a", year=2020),
        ],
        n_sources=1,
    )
    return PipelineResult(
        parameter=ParameterInput(name="max_lai", description="Max LAI", unit="m2/m2"),
        prior=prior,
        search_queries=["maize LAI"],
        papers_found=5,
        values_extracted=10,
    )


@pytest.fixture
def sample_batch(sample_result):
    return BatchResult(results=[sample_result], metadata={"n_parameters": 1})


class TestJsonExport:
    def test_single_json_valid(self, sample_result):
        output = export_single_json(sample_result)
        data = json.loads(output)
        assert data["parameter"] == "max_lai"
        assert data["distribution"] == "truncated_normal"
        assert "mu" in data["params"]

    def test_batch_json_valid(self, sample_batch):
        output = export_json(sample_batch)
        data = json.loads(output)
        assert "distribird_version" in data
        assert len(data["parameters"]) == 1


class TestRExport:
    def test_single_r(self, sample_result):
        code = export_single_r(sample_result)
        assert "rtruncnorm" in code
        assert "max_lai" in code

    def test_batch_r(self, sample_batch):
        code = export_r(sample_batch)
        assert "library(truncnorm)" in code
        assert "n <-" in code


class TestPythonExport:
    def test_single_python(self, sample_result):
        code = export_single_python(sample_result)
        assert "truncnorm" in code
        assert "max_lai" in code

    def test_batch_python_syntax(self, sample_batch):
        code = export_python(sample_batch)
        # Verify it's valid Python syntax
        ast.parse(code)

    def test_batch_python_content(self, sample_batch):
        code = export_python(sample_batch)
        assert "from scipy import stats" in code
        assert "N_SAMPLES" in code
