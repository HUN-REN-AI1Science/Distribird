"""Tests for Pydantic models."""

from litopri.models import (
    BatchResult,
    ConfidenceLevel,
    ConstraintSpec,
    DistributionFamily,
    ExtractedValue,
    FittedPrior,
    LiteratureEvidence,
    ParameterInput,
    PipelineResult,
)


def test_parameter_input_minimal():
    p = ParameterInput(name="test", description="A test parameter")
    assert p.name == "test"
    assert p.unit == ""
    assert p.constraints.lower_bound is None


def test_parameter_input_full():
    p = ParameterInput(
        name="lai",
        description="Leaf area index",
        unit="m2/m2",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=12),
        domain_context="crop modeling",
    )
    assert p.constraints.upper_bound == 12


def test_extracted_value():
    ev = ExtractedValue(reported_value=5.5, uncertainty=0.3)
    assert ev.reported_value == 5.5
    assert ev.reported_range is None


def test_literature_evidence():
    le = LiteratureEvidence(title="Test Paper", doi="10.1234/test")
    assert le.relevance_score == 0.0
    assert le.extracted_values == []


def test_fitted_prior_display_name():
    fp = FittedPrior(
        parameter_name="test",
        family=DistributionFamily.NORMAL,
        params={"mu": 5.0, "sigma": 1.0},
        confidence=ConfidenceLevel.HIGH,
        is_informative=True,
    )
    assert "normal" in fp.display_name()
    assert "5" in fp.display_name()


def test_pipeline_result():
    fp = FittedPrior(
        parameter_name="test",
        family=DistributionFamily.UNIFORM,
        params={"lower": 0, "upper": 1},
        confidence=ConfidenceLevel.NONE,
        is_informative=False,
    )
    pr = PipelineResult(
        parameter=ParameterInput(name="test", description="test"),
        prior=fp,
        papers_found=5,
        values_extracted=3,
    )
    assert pr.papers_found == 5


def test_batch_result():
    br = BatchResult(results=[], metadata={"n": 0})
    assert len(br.results) == 0


def test_model_serialization():
    p = ParameterInput(
        name="test",
        description="desc",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=10),
    )
    data = p.model_dump()
    p2 = ParameterInput(**data)
    assert p2.name == p.name
    assert p2.constraints.upper_bound == 10
