"""Tests for parameter context enrichment."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litopri.agent.enrich import enrich_parameter, enrich_parameter_context, research_model
from litopri.config import Settings
from litopri.models import ConstraintSpec, EnrichedContext, ParameterInput


@pytest.fixture
def settings():
    return Settings(llm_base_url="http://localhost:4000", llm_api_key="test")


@pytest.fixture
def parameter():
    return ParameterInput(
        name="allocation_ratio_root_leaf",
        description="Ratio of carbon allocated to roots vs leaves",
        unit="dimensionless",
        domain_context="Biome-BGCMuSo maize crop modeling",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=10),
    )


def _mock_llm_response(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    return mock_response


class TestResearchModel:
    @patch("litopri.agent.enrich.OpenAI")
    def test_research_model(self, mock_openai_cls, settings):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        response_data = {
            "model_name": "Biome-BGCMuSo",
            "model_summary": "A biogeochemical model for terrestrial ecosystems.",
            "scientific_domain": "ecosystem modeling",
            "key_processes": ["photosynthesis", "carbon allocation"],
        }
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            json.dumps(response_data)
        )

        result = research_model("Biome-BGCMuSo maize crop modeling", settings)
        assert "Biome-BGCMuSo" in result
        assert "biogeochemical" in result

        # Verify MODEL_RESEARCH prompt was used (messages[0] is system, [1] is user)
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "Biome-BGCMuSo maize crop modeling" in user_content


class TestEnrichParameter:
    @patch("litopri.agent.enrich.OpenAI")
    def test_enrich_parameter(self, mock_openai_cls, parameter, settings):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        response_data = {
            "parameter_meaning": "Controls belowground vs aboveground partitioning.",
            "common_terminology": ["root:shoot ratio", "carbon partitioning"],
            "typical_range": "0.5-2.0",
            "enriched_description": "Root to leaf carbon allocation ratio in maize",
            "search_hints": ["maize root shoot ratio", "carbon partitioning"],
        }
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            json.dumps(response_data)
        )

        result = enrich_parameter(parameter, "Biome-BGCMuSo model summary", settings)

        assert isinstance(result, EnrichedContext)
        assert "root:shoot ratio" in result.common_terminology
        assert result.enriched_description == "Root to leaf carbon allocation ratio in maize"
        assert len(result.search_hints) == 2


class TestEnrichParameterContext:
    @patch("litopri.agent.enrich.enrich_parameter")
    @patch("litopri.agent.enrich.research_model")
    def test_cache_miss_populates(
        self, mock_research, mock_enrich, parameter, settings
    ):
        mock_research.return_value = "Model summary text"
        mock_enrich.return_value = EnrichedContext(
            model_summary="Model summary text",
            common_terminology=["term1"],
        )

        cache: dict[str, str] = {}
        result = enrich_parameter_context(parameter, settings, model_cache=cache)

        mock_research.assert_called_once_with(
            "Biome-BGCMuSo maize crop modeling", settings
        )
        assert cache["Biome-BGCMuSo maize crop modeling"] == "Model summary text"
        assert result.common_terminology == ["term1"]

    @patch("litopri.agent.enrich.enrich_parameter")
    @patch("litopri.agent.enrich.research_model")
    def test_cache_hit(self, mock_research, mock_enrich, parameter, settings):
        mock_enrich.return_value = EnrichedContext(
            model_summary="Cached summary",
            common_terminology=["cached_term"],
        )

        cache = {"Biome-BGCMuSo maize crop modeling": "Cached summary"}
        result = enrich_parameter_context(parameter, settings, model_cache=cache)

        mock_research.assert_not_called()
        mock_enrich.assert_called_once()
        assert result.common_terminology == ["cached_term"]


class TestEnrichmentInPipeline:
    @pytest.mark.asyncio
    @patch("litopri.agent.extract.extract_all_values")
    @patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
    @patch("litopri.agent.search.generate_search_queries")
    async def test_enrichment_disabled(
        self, mock_queries, mock_search, mock_extract, settings
    ):
        from litopri.agent.pipeline import run_parameter
        from litopri.models import ExtractedValue, LiteratureEvidence

        settings.enable_context_enrichment = False
        settings.enable_deliberation = False
        settings.llm_web_search = False
        settings.search_refinement_max = 0
        settings.cross_enrichment_max = 0
        settings.extraction_refinement_max = 0

        mock_queries.return_value = ["query"]
        mock_search.return_value = [
            LiteratureEvidence(
                title="Paper",
                doi="10.1234/test",
                abstract="Value was 5.0.",
                year=2022,
                extracted_values=[ExtractedValue(reported_value=5.0)],
            )
        ]
        mock_extract.return_value = [
            LiteratureEvidence(
                title="Paper",
                extracted_values=[ExtractedValue(reported_value=5.0)],
            )
        ]

        param = ParameterInput(
            name="test_param",
            description="Test",
            unit="unit",
            domain_context="test",
            constraints=ConstraintSpec(lower_bound=0, upper_bound=100),
        )

        result = await run_parameter(param, settings)

        assert result.enrichment is None

    @pytest.mark.asyncio
    @patch("litopri.agent.extract.extract_all_values")
    @patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
    @patch("litopri.agent.search.generate_search_queries")
    @patch("litopri.agent.enrich.enrich_parameter_context")
    async def test_enrichment_failure_fallback(
        self, mock_enrich, mock_queries, mock_search, mock_extract, settings
    ):
        from litopri.agent.pipeline import run_parameter
        from litopri.models import ExtractedValue, LiteratureEvidence

        settings.enable_context_enrichment = True
        settings.enable_deliberation = False
        settings.llm_web_search = False
        settings.search_refinement_max = 0
        settings.cross_enrichment_max = 0
        settings.extraction_refinement_max = 0

        mock_enrich.side_effect = RuntimeError("LLM unavailable")

        mock_queries.return_value = ["query"]
        mock_search.return_value = [
            LiteratureEvidence(
                title="Paper",
                doi="10.1234/test",
                abstract="Value was 5.0.",
                year=2022,
                extracted_values=[ExtractedValue(reported_value=5.0)],
            )
        ]
        mock_extract.return_value = [
            LiteratureEvidence(
                title="Paper",
                extracted_values=[ExtractedValue(reported_value=5.0)],
            )
        ]

        param = ParameterInput(
            name="test_param",
            description="Test",
            unit="unit",
            domain_context="test",
            constraints=ConstraintSpec(lower_bound=0, upper_bound=100),
        )

        result = await run_parameter(param, settings)

        assert result.enrichment is None
        assert any("enrichment failed" in w.lower() for w in result.warnings)
        assert result.prior is not None
