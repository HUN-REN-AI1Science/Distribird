"""LLM-based value extraction from paper abstracts."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time

from openai import OpenAI

from distribird.agent import diagnostics
from distribird.config import Settings
from distribird.models import (
    ConstraintSpec,
    EnrichedContext,
    ExtractedValue,
    LiteratureEvidence,
    ParameterInput,
)

logger = logging.getLogger(__name__)

# Per-task token accumulator. `run_parameter_graph` calls `reset_token_accumulator()`
# at the start of each parameter's pipeline; every `_llm_json_call` invocation
# inside (including subtasks, since asyncio copies ContextVars per task) adds
# its response.usage into the dict. The dict reference is preserved so the
# graph can read final totals back without another ContextVar lookup.
_token_accumulator: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "distribird_token_accumulator", default=None
)


def reset_token_accumulator() -> dict[str, int]:
    """Install a fresh per-task token accumulator.

    Returns the dict (mutated in place by `_record_usage`).
    """
    acc: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_calls": 0,
    }
    _token_accumulator.set(acc)
    return acc


def _usage_dict(response: object) -> dict[str, int]:
    """Extract a plain token-usage dict from an LLM response (empty if absent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _record_usage(response: object) -> None:
    """Record one LLM response's `usage` block into the current task's accumulator.

    No-op if no accumulator is installed.
    """
    acc = _token_accumulator.get()
    if acc is None:
        return
    usage = _usage_dict(response)
    if not usage:
        return
    acc["prompt_tokens"] += usage["prompt_tokens"]
    acc["completion_tokens"] += usage["completion_tokens"]
    acc["total_tokens"] += usage["total_tokens"]
    acc["n_calls"] += 1


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _try_repair_json(text: str) -> str | None:
    """Attempt to repair truncated JSON by closing open brackets/braces.

    Returns repaired text if successful, None otherwise.
    """
    # Try parsing as-is first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Count unmatched delimiters and close them
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]") and stack and ch == stack[-1]:
            stack.pop()

    if not stack:
        return None

    # Truncate trailing partial value (incomplete string, number, etc.)
    # by finding the last complete key-value separator
    trimmed = text.rstrip()
    for end in (trimmed.rfind(","), trimmed.rfind('"'), trimmed.rfind("}")):
        if end > 0:
            candidate = trimmed[: end + 1].rstrip().rstrip(",")
            candidate += "".join(reversed(stack))
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    # Last resort: just close everything
    candidate = text.rstrip().rstrip(",") + "".join(reversed(stack))
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


_JSON_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a precise data extraction assistant. "
        "You MUST respond with ONLY valid JSON — no markdown, no explanation, "
        "no code fences, no preamble. Your entire response must be parseable "
        "by json.loads()."
    ),
}


def _is_reasoning_model(model: str) -> bool:
    """GPT-5 / o-series reasoning models reject `temperature != 1` and
    `web_search_options`. Detect by name prefix so callers can keep passing
    their preferred temperature for non-reasoning models."""
    m = (model or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _llm_json_call(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 2,
    extra_body: dict[str, object] | None = None,
    label: str = "llm",
) -> object:
    """Call LLM and parse JSON response, retrying on parse failures.

    Returns the parsed JSON object (dict or list).
    Raises json.JSONDecodeError after max_retries exhausted.

    ``label`` is a purpose tag (e.g. "value_extraction", "query_gen") recorded in
    the debug trace so each LLM call can be attributed to its pipeline stage.
    """
    # Prepend system message to enforce JSON-only output
    full_messages = [_JSON_SYSTEM_MSG] + list(messages)

    reasoning = _is_reasoning_model(model)
    if reasoning and isinstance(extra_body, dict) and "web_search_options" in extra_body:
        filtered = {k: v for k, v in extra_body.items() if k != "web_search_options"}
        extra_body = filtered or None

    tracing = diagnostics.enabled()
    started = time.time() if tracing else 0.0
    attempts: list[dict[str, object]] = []

    def _emit(status: str, parsed: object, usage: dict[str, int]) -> None:
        if not tracing:
            return
        diagnostics.record(
            "llm_call",
            {
                "label": label,
                "model": model,
                "reasoning_model": reasoning,
                "temperature": None if reasoning else temperature,
                "extra_body": extra_body,
                "messages": full_messages,
                "n_attempts": len(attempts),
                "attempts": attempts,
                "status": status,
                "parsed": parsed,
                "usage": usage,
                "latency_s": round(time.time() - started, 4),
            },
        )

    for attempt in range(max_retries + 1):
        create_kwargs: dict[str, object] = {
            "model": model,
            "messages": full_messages,
            "extra_body": extra_body,
        }
        if not reasoning:
            create_kwargs["temperature"] = temperature
        response = client.chat.completions.create(**create_kwargs)  # type: ignore[call-overload]
        _record_usage(response)
        usage = _usage_dict(response) if tracing else {}
        raw_text = response.choices[0].message.content or ""
        text = _strip_code_fences(raw_text)
        if tracing:
            attempts.append({"attempt": attempt, "raw_response": raw_text})
        try:
            parsed = json.loads(text)
            _emit("ok", parsed, usage)
            return parsed
        except json.JSONDecodeError:
            # Try to repair truncated JSON before retrying
            repaired = _try_repair_json(text)
            if repaired is not None:
                logger.info("[LLM:json_retry] repaired truncated JSON")
                parsed = json.loads(repaired)
                _emit("repaired", parsed, usage)
                return parsed
            if attempt < max_retries:
                logger.info(
                    "[LLM:json_retry] attempt %d/%d failed, retrying",
                    attempt + 1,
                    max_retries,
                )
                full_messages = list(full_messages) + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. "
                            "Return ONLY valid JSON, no other text."
                        ),
                    },
                ]
            else:
                _emit("parse_failed", None, usage)
                raise
    return None


def _effective_description(parameter: ParameterInput, enrichment: EnrichedContext | None) -> str:
    """Return enriched description if available, otherwise the original."""
    if enrichment and enrichment.enriched_description:
        return enrichment.enriched_description
    return parameter.description


def _build_context_block(enrichment: EnrichedContext | None) -> str:
    """Build a context summary for prompts from enrichment data."""
    if enrichment is None:
        return ""
    parts = []
    if enrichment.application_context:
        parts.append(f"Application context: {enrichment.application_context}")
    if enrichment.context_keywords:
        parts.append(f"Context keywords: {', '.join(enrichment.context_keywords)}")
    if enrichment.typical_range:
        parts.append(f"Expected range: {enrichment.typical_range}")
    return "\n".join(parts)


def _paper_text(paper: LiteratureEvidence) -> str:
    """Return the best available text for a paper (full text > abstract)."""
    if paper.full_text:
        return paper.full_text
    return paper.abstract


def extract_values_from_paper(
    paper: LiteratureEvidence,
    parameter: ParameterInput,
    settings: Settings,
    enrichment: EnrichedContext | None = None,
) -> list[ExtractedValue]:
    """Extract parameter values from a paper using LLM."""
    from distribird.agent.prompts import VALUE_EXTRACTION

    text = _paper_text(paper)
    if not text:
        return []

    source_type = "full_text" if paper.full_text else "abstract"
    constraint = parameter.constraints
    prompt = VALUE_EXTRACTION.format(
        name=parameter.name,
        description=_effective_description(parameter, enrichment),
        unit=parameter.unit,
        lower_bound=constraint.lower_bound,
        upper_bound=constraint.upper_bound,
        context_block=_build_context_block(enrichment),
        title=paper.title,
        abstract=text,
    )

    logger.info(
        "[LLM:extract] param=%r paper=%r source=%s model=%s",
        parameter.name,
        paper.title[:80],
        source_type,
        settings.llm_model,
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    try:
        raw = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature_precise,
            label="value_extraction",
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("[LLM:extract] failed paper=%r error=%s", paper.title[:80], e)
        return []

    if not isinstance(raw, list):
        return []

    values = _parse_extracted_items(raw, constraint, paper.title, enrichment)

    logger.info(
        "[LLM:extract] param=%r paper=%r values_extracted=%d",
        parameter.name,
        paper.title[:80],
        len(values),
    )
    return values


def _parse_extracted_items(
    raw: list[object],
    constraint: ConstraintSpec,
    paper_title: str,
    enrichment: EnrichedContext | None = None,
) -> list[ExtractedValue]:
    """Parse raw JSON items into ExtractedValue list with bounds and plausibility checking."""
    values = []
    rejected: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ev = ExtractedValue(
            reported_value=_parse_number(item.get("reported_value")),
            reported_range=_parse_range(item.get("reported_range")),
            uncertainty=_parse_number(item.get("uncertainty")),
            sample_size=item.get("sample_size"),
            context=item.get("context", ""),
        )
        if not _passes_bounds_check(ev, constraint):
            logger.info(
                "[LLM:extract] excluded out-of-bounds value=%s paper=%r",
                ev.reported_value,
                paper_title[:80],
            )
            rejected.append({"value": ev.reported_value, "reason": "out_of_bounds"})
            continue
        if not _passes_plausibility_check(ev, enrichment):
            logger.info(
                "[LLM:extract] excluded implausible value=%s (outside typical range) paper=%r",
                ev.reported_value,
                paper_title[:80],
            )
            rejected.append({"value": ev.reported_value, "reason": "implausible"})
            continue
        values.append(ev)

    if diagnostics.enabled():
        diagnostics.record(
            "extraction",
            {
                "paper": paper_title[:200],
                "n_raw_items": len(raw),
                "accepted": [v.model_dump() for v in values],
                "rejected": rejected,
            },
        )
    return values


def extract_values_batch(
    papers: list[LiteratureEvidence],
    parameter: ParameterInput,
    settings: Settings,
    batch_size: int = 5,
    enrichment: EnrichedContext | None = None,
) -> list[list[ExtractedValue]]:
    """Extract values from multiple papers in batched LLM calls.

    Returns a list of ExtractedValue lists, one per input paper (same order).
    Falls back to per-paper extraction if the batch call fails.
    """
    from distribird.agent.prompts import BATCH_VALUE_EXTRACTION

    constraint = parameter.constraints

    # Build numbered paper block (use best available text)
    lines = []
    for i, paper in enumerate(papers):
        lines.append(f"[{i}] Title: {paper.title}")
        text = _paper_text(paper) or "(no abstract)"
        lines.append(f"    Abstract: {text}")
    papers_block = "\n".join(lines)

    prompt = BATCH_VALUE_EXTRACTION.format(
        name=parameter.name,
        description=_effective_description(parameter, enrichment),
        unit=parameter.unit,
        lower_bound=constraint.lower_bound,
        upper_bound=constraint.upper_bound,
        context_block=_build_context_block(enrichment),
        papers_block=papers_block,
    )

    logger.info(
        "[LLM:batch_extract] param=%r papers=%d",
        parameter.name,
        len(papers),
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    try:
        raw = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature_precise,
            label="batch_value_extraction",
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("[LLM:batch_extract] batch failed, falling back to per-paper: %s", e)
        return [extract_values_from_paper(p, parameter, settings, enrichment) for p in papers]

    if not isinstance(raw, dict):
        logger.warning(
            "[LLM:batch_extract] expected dict, got %s; falling back",
            type(raw).__name__,
        )
        return [extract_values_from_paper(p, parameter, settings, enrichment) for p in papers]

    # Parse results by index
    results: list[list[ExtractedValue]] = []
    for i, paper in enumerate(papers):
        paper_raw = raw.get(str(i), [])
        if not isinstance(paper_raw, list):
            paper_raw = []
        results.append(_parse_extracted_items(paper_raw, constraint, paper.title, enrichment))

    logger.info(
        "[LLM:batch_extract] param=%r extracted=%s",
        parameter.name,
        [len(r) for r in results],
    )
    return results


def extract_values_web_assisted(
    papers: list[LiteratureEvidence],
    parameter: ParameterInput,
    settings: Settings,
    batch_size: int = 3,
    max_papers: int = 10,
    enrichment: EnrichedContext | None = None,
) -> list[LiteratureEvidence]:
    """Use web-search-capable LLM to look up specific papers online and extract values.

    Sorts papers by relevance, takes top max_papers, batches in groups of batch_size,
    and calls the LLM with web_search_options to find full text content.
    Returns papers that gained new extracted values.
    """
    from distribird.agent.prompts import WEB_ASSISTED_EXTRACTION

    constraint = parameter.constraints

    # Sort by relevance and cap
    sorted_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)
    selected = sorted_papers[:max_papers]

    logger.info(
        "[LLM:web_extract] param=%r papers=%d (of %d)",
        parameter.name,
        len(selected),
        len(papers),
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    extra_body: dict[str, object] = {"web_search_options": {"search_context_size": "high"}}

    papers_with_new_values: list[LiteratureEvidence] = []

    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]

        # Build paper blocks with DOI for web lookup
        lines = []
        for i, paper in enumerate(chunk):
            lines.append(f"[{i}] Title: {paper.title}")
            lines.append(f"    DOI: {paper.doi or 'unknown'}")
            lines.append(f"    Year: {paper.year or 'unknown'}")
            abstract_snippet = (paper.abstract or "")[:300]
            lines.append(f"    Abstract snippet: {abstract_snippet}")
        papers_block = "\n".join(lines)

        prompt = WEB_ASSISTED_EXTRACTION.format(
            name=parameter.name,
            description=_effective_description(parameter, enrichment),
            unit=parameter.unit,
            lower_bound=constraint.lower_bound,
            upper_bound=constraint.upper_bound,
            context_block=_build_context_block(enrichment),
            papers_block=papers_block,
        )

        try:
            raw = _llm_json_call(
                client,
                settings.llm_model,
                [{"role": "user", "content": prompt}],
                temperature=settings.llm_temperature_precise,
                extra_body=extra_body,
                label="web_assisted_extraction",
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("[LLM:web_extract] batch failed: %s", e)
            continue

        if not isinstance(raw, dict):
            continue

        for i, paper in enumerate(chunk):
            paper_raw = raw.get(str(i), [])
            if not isinstance(paper_raw, list):
                continue
            values = _parse_extracted_items(paper_raw, constraint, paper.title, enrichment)
            if values:
                # Store source_url in context if provided
                for item_raw, val in zip(paper_raw, values):
                    if isinstance(item_raw, dict) and item_raw.get("source_url"):
                        val.context = f"{val.context} [source: {item_raw['source_url']}]"
                paper.extracted_values = values
                papers_with_new_values.append(paper)

    logger.info(
        "[LLM:web_extract] param=%r papers_with_values=%d",
        parameter.name,
        len(papers_with_new_values),
    )
    return papers_with_new_values


def extract_all_values(
    papers: list[LiteratureEvidence],
    parameter: ParameterInput,
    settings: Settings,
    batch_size: int = 5,
    enrichment: EnrichedContext | None = None,
) -> list[LiteratureEvidence]:
    """Extract values from all papers and update them in-place.

    Papers with full text are extracted individually (too large for batching).
    Abstract-only papers use batched extraction for efficiency.
    Returns papers that had at least one value extracted.
    """
    logger.info("[LLM:extract_all] param=%r papers=%d", parameter.name, len(papers))

    # Split: full-text papers get per-paper extraction, abstract-only use batch
    fulltext_papers = [p for p in papers if p.full_text]
    abstract_papers = [p for p in papers if not p.full_text]

    if fulltext_papers:
        logger.info(
            "[LLM:extract_all] extracting from %d full-text papers",
            len(fulltext_papers),
        )
        for paper in fulltext_papers:
            values = extract_values_from_paper(paper, parameter, settings, enrichment)
            if values:
                paper.extracted_values = values

    # Batch-extract from abstract-only papers
    if abstract_papers:
        for start in range(0, len(abstract_papers), batch_size):
            chunk = abstract_papers[start : start + batch_size]
            batch_results = extract_values_batch(
                chunk, parameter, settings, batch_size, enrichment
            )
            for paper, values in zip(chunk, batch_results):
                if values:
                    paper.extracted_values = values

    papers_with_values = [p for p in papers if p.extracted_values]

    logger.info(
        "[LLM:extract_all] param=%r papers_with_values=%d/%d (fulltext=%d, abstract=%d)",
        parameter.name,
        len(papers_with_values),
        len(papers),
        len(fulltext_papers),
        len(abstract_papers),
    )
    return papers_with_values


def extract_consensus_values(
    papers: list[LiteratureEvidence],
    parameter: ParameterInput,
    settings: Settings,
    enrichment: EnrichedContext | None = None,
) -> list[ExtractedValue]:
    """Fallback: synthesize consensus values across papers when per-paper extraction fails.

    Asks the LLM to look across all abstracts and determine the standard/default
    or implied value for the parameter.
    """
    from distribird.agent.prompts import CONSENSUS_EXTRACTION

    lines = []
    for i, paper in enumerate(papers):
        text = _paper_text(paper) or "(no abstract)"
        lines.append(f"[{i}] Title: {paper.title}\n    Abstract: {text}")
    papers_block = "\n\n".join(lines)

    prompt = CONSENSUS_EXTRACTION.format(
        name=parameter.name,
        description=_effective_description(parameter, enrichment),
        unit=parameter.unit,
        domain_context=parameter.domain_context,
        context_block=_build_context_block(enrichment),
        papers_block=papers_block,
    )

    logger.info(
        "[LLM:consensus_extract] param=%r papers=%d",
        parameter.name,
        len(papers),
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    try:
        raw = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature_precise,
            label="consensus_extraction",
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("[LLM:consensus_extract] failed: %s", e)
        return []

    if not isinstance(raw, dict):
        return []

    consensus_value = _parse_number(raw.get("consensus_value"))
    consensus_range = _parse_range(raw.get("consensus_range"))
    uncertainty = _parse_number(raw.get("uncertainty"))
    context = raw.get("context", "")
    evidence_type = raw.get("evidence_type", "consensus")
    confidence = raw.get("confidence", "low")

    if consensus_value is None and consensus_range is None:
        logger.info("[LLM:consensus_extract] no consensus value found")
        return []

    constraint = parameter.constraints
    ev = ExtractedValue(
        reported_value=consensus_value,
        reported_range=consensus_range,
        uncertainty=uncertainty,
        context=f"[{evidence_type}, {confidence} confidence] {context}",
    )

    if not _passes_bounds_check(ev, constraint):
        logger.info(
            "[LLM:consensus_extract] consensus value out of bounds: %s",
            consensus_value,
        )
        return []

    logger.info(
        "[LLM:consensus_extract] param=%r value=%s range=%s uncertainty=%s type=%s",
        parameter.name,
        consensus_value,
        consensus_range,
        uncertainty,
        evidence_type,
    )
    return [ev]


def _parse_number(raw: object) -> float | None:
    """Try to extract a float from a value that might be a string like 'SD=0.4'."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        nums = re.findall(r"[-+]?\d*\.?\d+", raw)
        if nums:
            return float(nums[0])
    return None


def _parse_range(raw: object) -> tuple[float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (ValueError, TypeError):
            return None
    if isinstance(raw, str):
        nums = re.findall(r"[-+]?\d*\.?\d+", raw)
        if len(nums) >= 2:
            return (float(nums[0]), float(nums[1]))
    return None


def _parse_typical_range(typical_range: str) -> tuple[float, float] | None:
    """Parse the enrichment typical_range string to extract numeric bounds.

    Handles formats like "0°C - 5°C", "0.5-2.0 (dimensionless)", "typically 0 to 12 m2/m2".
    Uses min/max of all extracted numbers to capture the full range even when the
    text contains repeated values (e.g., "0°C is standard... explores 0°C to 5°C").
    Returns (low, high) or None if unparseable.
    """
    if not typical_range:
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+", typical_range)
    if len(nums) >= 2:
        try:
            floats = [float(n) for n in nums]
            low, high = min(floats), max(floats)
            if low < high:
                return (low, high)
        except ValueError:
            pass
    return None


def _passes_bounds_check(value: ExtractedValue, constraint: ConstraintSpec) -> bool:
    """Check if extracted value is within physical bounds."""
    v = value.reported_value
    if v is None:
        return True  # No value to check
    if constraint.lower_bound is not None and v < constraint.lower_bound:
        return False
    if constraint.upper_bound is not None and v > constraint.upper_bound:
        return False
    return True


def _passes_plausibility_check(
    value: ExtractedValue,
    enrichment: EnrichedContext | None,
    margin_factor: float = 2.0,
) -> bool:
    """Soft plausibility check using enrichment typical_range.

    Excludes values that are far outside the expected range.
    Uses a margin (default 2x the range width) to allow some slack while
    catching clearly wrong values (e.g., germination temperature vs. development temperature).
    """
    if enrichment is None or not enrichment.typical_range:
        return True
    v = value.reported_value
    if v is None:
        return True
    bounds = _parse_typical_range(enrichment.typical_range)
    if bounds is None:
        return True
    low, high = bounds
    width = max(high - low, 1.0)  # minimum width of 1.0 to avoid zero-width ranges
    margin = width * margin_factor
    return (low - margin) <= v < (high + margin)
