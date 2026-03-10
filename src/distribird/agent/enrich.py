"""Parameter context enrichment via LLM pre-research."""

from __future__ import annotations

import logging

from openai import OpenAI

from distribird.agent.extract import _llm_json_call
from distribird.config import Settings
from distribird.models import EnrichedContext, ParameterInput

logger = logging.getLogger(__name__)

_WEB_SEARCH_EXTRA_BODY: dict[str, object] = {
    "web_search_options": {"search_context_size": "high"},
}


def research_model(domain_context: str, settings: Settings) -> str:
    """Use LLM to identify and summarize the model from domain_context."""
    from distribird.agent.prompts import MODEL_RESEARCH

    prompt = MODEL_RESEARCH.format(domain_context=domain_context)

    extra_body = _WEB_SEARCH_EXTRA_BODY if settings.llm_web_search else None
    logger.info(
        "[LLM:model_research] domain=%r model=%s web_search=%s",
        domain_context,
        settings.llm_model,
        settings.llm_web_search,
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    raw = _llm_json_call(
        client,
        settings.llm_model,
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        extra_body=extra_body,
    )

    if isinstance(raw, dict):
        summary = raw.get("model_summary", "")
        name = raw.get("model_name", "")
        domain = raw.get("scientific_domain", "")
        processes = raw.get("key_processes", [])
        result = f"{name}: {summary} Domain: {domain}. Key processes: {', '.join(processes)}."
        logger.info("[LLM:model_research] summary=%r", result[:120])
        return result

    logger.warning("[LLM:model_research] unexpected response type: %s", type(raw).__name__)
    return ""


def enrich_parameter(
    parameter: ParameterInput,
    model_summary: str,
    settings: Settings,
) -> EnrichedContext:
    """Use LLM to enrich a parameter with scientific context."""
    from distribird.agent.prompts import PARAMETER_ENRICHMENT

    prompt = PARAMETER_ENRICHMENT.format(
        model_summary=model_summary,
        name=parameter.name,
        description=parameter.description,
        unit=parameter.unit,
        domain_context=parameter.domain_context,
        lower_bound=parameter.constraints.lower_bound,
        upper_bound=parameter.constraints.upper_bound,
    )

    logger.info(
        "[LLM:enrich_param] param=%r model=%s",
        parameter.name,
        settings.llm_model,
    )

    extra_body = _WEB_SEARCH_EXTRA_BODY if settings.llm_web_search else None
    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    raw = _llm_json_call(
        client,
        settings.llm_model,
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        extra_body=extra_body,
    )

    if isinstance(raw, dict):
        ctx = EnrichedContext(
            model_summary=model_summary,
            parameter_meaning=raw.get("parameter_meaning", ""),
            common_terminology=raw.get("common_terminology", []),
            typical_range=raw.get("typical_range", ""),
            enriched_description=raw.get("enriched_description", ""),
            search_hints=raw.get("search_hints", []),
            application_context=raw.get("application_context", ""),
            context_keywords=raw.get("context_keywords", []),
        )
        logger.info(
            "[LLM:enrich_param] param=%r terms=%r hints=%r",
            parameter.name,
            ctx.common_terminology,
            ctx.search_hints,
        )
        return ctx

    logger.warning("[LLM:enrich_param] unexpected response type: %s", type(raw).__name__)
    return EnrichedContext(model_summary=model_summary)


def enrich_parameter_context(
    parameter: ParameterInput,
    settings: Settings,
    model_cache: dict[str, str] | None = None,
) -> EnrichedContext:
    """Orchestrate model research + parameter enrichment.

    Uses model_cache to avoid redundant LLM calls for the same domain_context.
    """
    if model_cache is None:
        model_cache = {}

    domain = parameter.domain_context
    if domain not in model_cache:
        logger.info("[enrich] cache miss for domain=%r, calling research_model", domain)
        model_cache[domain] = research_model(domain, settings)
    else:
        logger.info("[enrich] cache hit for domain=%r", domain)

    model_summary = model_cache[domain]
    return enrich_parameter(parameter, model_summary, settings)
