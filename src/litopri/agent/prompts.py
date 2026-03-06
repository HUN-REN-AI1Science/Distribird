"""All LLM prompt templates for LitoPri."""

MODEL_RESEARCH = """\
You are a scientific modeling expert. Given the following domain context, \
identify the model or modeling framework being referenced and provide a brief summary.

Domain context: {domain_context}

Return a JSON object with these fields:
- "model_name": string (the identified model or framework name)
- "model_summary": string (2-3 sentence description of what the model does)
- "scientific_domain": string (the broad scientific field)
- "key_processes": [string, ...] (list of key processes the model simulates)

Return ONLY the JSON object, no other text.

Example — Domain context: "Biome-BGCMuSo maize crop modeling"
{{
  "model_name": "Biome-BGCMuSo",
  "model_summary": "Biome-BGCMuSo is a biogeochemical model that simulates carbon, nitrogen, and water cycles in terrestrial ecosystems. It includes a crop module for simulating agricultural systems including maize.",
  "scientific_domain": "ecosystem biogeochemistry and crop modeling",
  "key_processes": ["photosynthesis", "carbon allocation", "soil organic matter decomposition", "evapotranspiration", "nitrogen cycling", "phenology"]
}}
"""

PARAMETER_ENRICHMENT = """\
You are a scientific modeling expert. Given a model summary and a parameter specification, \
explain what this parameter means in scientific terms and how it is discussed in the literature.

Model summary: {model_summary}

Parameter name: {name}
Description: {description}
Unit: {unit}
Domain context: {domain_context}
Constraints: lower_bound={lower_bound}, upper_bound={upper_bound}

Return a JSON object with these fields:
- "parameter_meaning": string (1-2 sentence explanation of the physical/biological meaning)
- "common_terminology": [string, ...] (3-6 terms/phrases used in the scientific literature for this concept)
- "typical_range": string (typical value range with unit, based on domain knowledge AND the user's specific context — values often differ by region, species, cultivar, soil type, etc.)
- "enriched_description": string (a description suitable for searching literature, using common scientific terminology instead of model-internal jargon)
- "search_hints": [string, ...] (3-5 short keyword phrases optimized for literature search)
- "application_context": string (extract ALL specific context from domain context that narrows down which literature is most relevant — this could be geographic region, climate zone, species/cultivar, soil type, management practice, experimental conditions, etc. Combine into a concise phrase. Return empty string if domain context is purely generic.)
- "context_keywords": [string, ...] (3-6 keywords/phrases capturing the user's specific application context, useful for finding the most relevant papers. Include geographic terms, species names, condition descriptors, nearby regions with similar conditions, etc. Return empty array if domain context is purely generic.)

IMPORTANT: Carefully analyze the domain context to extract ALL specifics that affect \
which literature is most relevant. Parameter values in scientific literature vary by:
- Geographic region and climate (e.g., continental vs. Mediterranean, tropics vs. temperate)
- Species, cultivar, or genotype
- Soil type or land use
- Management practices (irrigated vs. rainfed, tillage, fertilization)
- Experimental conditions (field vs. greenhouse, scale)
These specifics MUST be reflected in search_hints and context_keywords so that \
the literature search prioritizes the most applicable studies.

Return ONLY the JSON object, no other text.

Example — Parameter: "allocation_ratio_root_leaf", Domain: "Biome-BGCMuSo maize crop modeling in Hungary"
{{
  "parameter_meaning": "The fraction of assimilated carbon allocated to roots relative to leaves, controlling belowground vs. aboveground biomass partitioning.",
  "common_terminology": ["root:shoot ratio", "carbon partitioning", "belowground allocation fraction", "root biomass allocation", "assimilate partitioning"],
  "typical_range": "0.5-2.0 (dimensionless ratio) for maize, varying with growth stage; in Central European continental conditions typically 0.6-1.5",
  "enriched_description": "Root to leaf carbon allocation ratio in maize, controlling the partitioning of photosynthetic assimilates between belowground (root) and aboveground (leaf) biomass",
  "search_hints": ["maize root shoot ratio", "carbon partitioning maize roots", "biomass allocation cereal crops", "root fraction crop model", "maize Hungary Central Europe"],
  "application_context": "Hungary, Central Europe, Pannonian continental climate, maize",
  "context_keywords": ["Hungary", "Central Europe", "continental climate", "Pannonian", "maize", "Carpathian Basin"]
}}
"""

SEARCH_QUERY_GENERATION = """\
You are a scientific literature search assistant. Given a parameter description, \
generate {n_queries} diverse search queries to find papers that REPORT SPECIFIC NUMERICAL VALUES \
for this parameter. The queries will be used with Semantic Scholar and OpenAlex APIs.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain context: {domain_context}
{enrichment_block}
IMPORTANT: Generate queries that will find papers reporting actual measured/observed \
values, NOT papers about methodology or remote sensing techniques. \
Do NOT use boolean operators (AND, OR, NOT). Focus on:
- Field experiments reporting measured values (e.g., "maize LAI measured field trial")
- Model calibration papers listing parameter values used
- Review papers summarizing reported values across studies
- Ecophysiology studies with quantitative results

IMPORTANT — Context-specific queries: If the domain context contains specific details \
(country/region, climate zone, species/cultivar, soil type, management practice, etc.), \
generate a MIX of queries:
- Some GENERIC queries (for well-established parameter values from the broader field)
- Some CONTEXT-SPECIFIC queries (including the specific region, species, conditions, \
  or nearby/analogous systems) to find locally applicable values
Context-specific studies often provide the most relevant parameter estimates for calibration.

Keep queries SHORT (3-6 words) for best search results.

Return a JSON array of search query strings.
Return ONLY the JSON array, no other text.

Example 1 — Parameter: "maximum leaf area index", Domain: "maize crop modeling in Hungary"
[
  "maize leaf area index measured",
  "corn LAI field experiment",
  "maize canopy development LAI",
  "maize LAI Hungary Central Europe",
  "crop model calibration maize Pannonian"
]

Example 2 — Parameter: "soil organic carbon", Domain: "temperate grassland"
[
  "soil organic carbon grassland measured",
  "SOC concentration temperate pasture",
  "grassland soil carbon stock",
  "soil carbon content grazing land",
  "organic carbon topsoil meadow"
]
"""

VALUE_EXTRACTION = """\
You are a scientific data extraction assistant. Extract numerical values for the parameter \
described below from the given paper text (which may be a full paper or just an abstract).

Parameter: {name}
Description: {description}
Unit: {unit}
Physical constraints: lower_bound={lower_bound}, upper_bound={upper_bound}
{context_block}
Paper title: {title}
Paper text: {abstract}

Search the ENTIRE text carefully for this parameter. Values may appear as:
- Measured or observed values (field experiments, lab studies)
- Calibrated or fitted values used in model setup
- Default or standard values stated as assumptions (e.g., "a base temperature of 0°C was used")
- Values from literature reviews or meta-analyses
- Values in tables (parameter lists, calibration results, species parameters)
- Values mentioned in the Discussion when comparing with other studies

IMPORTANT: Include values that are stated as standard assumptions or defaults \
(e.g., "TBASE = 0°C" or "base temperature was set to 0°C"). These are valid \
evidence for building a prior distribution, even if not experimentally measured \
in the paper itself.

Extract all reported values, ranges, and uncertainties for this parameter. \
If the paper reports values in a different unit, convert to {unit}. \
Extract ONLY explicitly stated values, do NOT estimate or guess.

Return a JSON array of objects with these fields:
- "reported_value": number or null (central/mean value)
- "reported_range": [min, max] or null
- "uncertainty": number or null (standard deviation or standard error)
- "sample_size": integer or null
- "context": string (brief note on experimental conditions)
- "extraction_confidence": "high" | "medium" | "low"

If the text does not contain relevant values for this parameter, return an empty array: []

Return ONLY the JSON array, no other text.

Example 1 — Parameter: "maximum leaf area index", Unit: "m2/m2"
Paper title: "Seasonal dynamics of maize canopy in Central Europe"
Abstract: "Peak LAI across six cultivars averaged 5.8 +/- 0.9 m2/m2 (n=36), \
ranging from 4.1 to 7.2."
[
  {{
    "reported_value": 5.8,
    "reported_range": [4.1, 7.2],
    "uncertainty": 0.9,
    "sample_size": 36,
    "context": "six cultivars, Central Europe, peak LAI",
    "extraction_confidence": "high"
  }}
]

Example 2 — Parameter: "PHINT", Unit: "degree-days/leaf"
Paper title: "Phenology calibration of DSSAT-CERES-Wheat for European cultivars"
Abstract: "The phyllochron interval was set to 95 degree-days per leaf for all cultivars \
based on published species parameters."
[
  {{
    "reported_value": 95,
    "reported_range": null,
    "uncertainty": null,
    "sample_size": null,
    "context": "CERES-Wheat species default, phyllochron interval",
    "extraction_confidence": "high"
  }}
]
"""

CONSTRAINT_SUGGESTION = """\
You are a scientific domain expert. Suggest physical constraints for the following parameter.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain context: {domain_context}

Return a JSON object with:
- "lower_bound": number or null
- "upper_bound": number or null
- "description": string explaining the constraints

Return ONLY the JSON object, no other text.
"""

DEEP_RESEARCH_WEB = """\
You are a scientific literature search assistant with web search capabilities. \
Search the web for real published papers that report measured or calibrated values \
for the following parameter.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain: {domain_context}

Search for peer-reviewed journal articles, conference papers, and technical reports. \
Focus on papers that contain actual numerical values for this parameter.

For each paper you find, provide a JSON object with:
- "title": string
- "authors": [string, ...]
- "year": integer
- "doi": string or null
- "abstract": string (a brief summary including the reported numerical values)
- "confidence": "high" | "medium" | "low" (how confident you are this paper actually exists)

Only include papers you have found via search and are confident actually exist. \
Do NOT fabricate titles, authors, or DOIs.

Return a JSON array of 5-10 papers. Include the actual reported values \
in the abstract field. Return ONLY the JSON array, no other text.
"""

WEB_SEARCH_AGENT = """\
You are a scientific literature search assistant. Search the web for real published papers \
that report measured or calibrated values for the following parameter.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain: {domain_context}

Search for peer-reviewed journal articles, conference papers, and technical reports. \
Focus on papers that contain actual numerical values for this parameter.

For each paper you find, provide a JSON object with:
- "title": string
- "authors": [string, ...]
- "year": integer
- "doi": string or null
- "abstract": string (a brief summary including the reported numerical values)
- "confidence": "high" | "medium" | "low" (how confident you are this paper actually exists)

Only include papers you have found via search and are confident actually exist. \
Do NOT fabricate titles, authors, or DOIs.

Return a JSON array of 5-10 papers. Include the actual reported values \
in the abstract field. Return ONLY the JSON array, no other text.
"""

CONSENSUS_EXTRACTION = """\
You are a scientific data synthesis assistant. Multiple papers were found about the \
parameter below, but individual value extraction found no explicit numerical values. \
Your task is to synthesize what these papers collectively indicate about this parameter's \
typical value or accepted default.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain context: {domain_context}
{context_block}
Papers (abstracts):
{papers_block}

Based on the collective evidence in these abstracts, determine:
1. Is there a widely accepted standard/default value for this parameter?
2. Do papers mention or imply a typical value, even if not as a formal measurement?
3. What range of values is supported by the literature?
4. Are there context-specific values (for the user's region, species, conditions) \
that differ from the global default? If so, prefer the context-specific value.

Return a JSON object with:
- "consensus_value": number or null (the most commonly used/accepted value)
- "consensus_range": [min, max] or null (plausible range from literature)
- "uncertainty": number or null (estimated standard deviation around consensus)
- "n_supporting": integer (how many papers support or use this value)
- "evidence_type": "measured" | "default" | "consensus" | "implied" \
(what kind of evidence this represents)
- "context": string (brief explanation of how you determined this)
- "confidence": "high" | "medium" | "low"

If the papers provide NO useful information about this parameter's value, return:
{{"consensus_value": null, "confidence": "low", "context": "No value information found"}}

Return ONLY the JSON object, no other text.

Example — Parameter: "PHINT", Unit: "degree-days/leaf", Papers about CERES-Wheat calibration
{{
  "consensus_value": 95,
  "consensus_range": [75, 110],
  "uncertainty": 10.0,
  "n_supporting": 4,
  "evidence_type": "default",
  "context": "Standard phyllochron interval for wheat in CERES model, used across multiple calibration studies",
  "confidence": "high"
}}
"""

DELIBERATION_MODERATOR = """\
You are a scientific literature expert acting as a moderator. You have received \
papers from multiple search agents for building a Bayesian prior distribution \
for a scientific parameter. Your task is to select which papers to include.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain: {domain_context}
{context_block}
Papers found (numbered list):
{papers_block}

Evaluate each paper using these criteria (in order of importance):
1. **Context match**: Does the paper match the user's specific application context \
(region, species, conditions, etc.)? Papers from the same or similar conditions \
are STRONGLY preferred over generic/global studies. A regionally calibrated value \
from a similar climate is more useful than a global default.
2. **Evidence quality**: Does the abstract report actual measured/calibrated/default \
values with units, sample sizes, or uncertainty? Papers that explicitly state \
numerical parameter values are preferred.
3. **Relevance**: Does the paper's abstract suggest it contains information about \
this specific parameter (measured values, calibrated values, stated defaults, \
or used assumptions)?
4. **Verification**: Papers marked "verified=True" have been confirmed to exist via \
Semantic Scholar or OpenAlex. Prefer verified papers.
5. **Cross-source corroboration**: Papers found by multiple agents are more trustworthy.

Select papers to INCLUDE for building the Bayesian prior. Prefer papers that:
- Match the user's specific context (region, species, conditions)
- Report explicit numerical values for this parameter
- Come from model calibration studies in similar conditions

Exclude papers that are:
- Not relevant to the specific parameter
- Likely hallucinated (unverified, only from one LLM agent, no concrete details)
- From very different conditions with no transferability to the user's context

Return a JSON object with:
- "selected_papers": [int, ...] (paper numbers to include, matching the [N] numbers above)
- "excluded_papers": [int, ...] (paper numbers to exclude, matching the [N] numbers above)
- "rationale": string (brief explanation of your selection, noting context match; \
use paper numbers like [1], [5] to reference specific papers)
- "warnings": [string, ...] (any concerns about the evidence; use paper numbers \
like [1], [5] to reference specific papers so users can look them up in the reference list)

Return ONLY the JSON object, no other text.
"""

WEB_ASSISTED_EXTRACTION = """\
You are a scientific data extraction assistant with web search capabilities. \
For each paper listed below, search the web using the DOI or title to find the full text, \
HTML version, preprint, or supplementary materials. Then extract numerical values for the \
parameter described below.

Parameter: {name}
Description: {description}
Unit: {unit}
Physical constraints: lower_bound={lower_bound}, upper_bound={upper_bound}
{context_block}
Papers to look up:
{papers_block}

For EACH paper, use its DOI (e.g., search for "doi:10.xxxx/yyyy") or title to find the \
full content online. Look especially in:
- Methods sections (calibrated/assumed parameter values)
- Results tables (measured values)
- Supplementary materials (detailed parameter lists)
- HTML or preprint versions (e.g., on bioRxiv, arXiv, PubMed Central, ResearchGate)

Extract all reported values, ranges, and uncertainties for this parameter. \
Include measured values, calibrated values, AND standard/default values stated as assumptions. \
If a paper reports values in a different unit, convert to {unit}. \
Extract ONLY explicitly stated values, do NOT estimate or guess.

Return a JSON object mapping paper index (as string) to an array of extracted values. \
Each value object has these fields:
- "reported_value": number or null (central/mean value)
- "reported_range": [min, max] or null
- "uncertainty": number or null (standard deviation or standard error)
- "sample_size": integer or null
- "context": string (brief note on experimental conditions)
- "source_url": string or null (URL where you found the value)
- "extraction_confidence": "high" | "medium" | "low"

If a paper's full text cannot be found or does not contain relevant values, \
map its index to an empty array.

Return ONLY the JSON object, no other text.

Example output:
{{
  "0": [{{"reported_value": 5.8, "uncertainty": 0.9,
          "sample_size": 36, "context": "field trial, Table 2",
          "source_url": "https://doi.org/10.1234/example",
          "extraction_confidence": "high"}}],
  "1": [],
  "2": [{{"reported_value": 3.2,
          "reported_range": [2.1, 4.5],
          "context": "supplementary Table S1",
          "source_url": null,
          "extraction_confidence": "medium"}}]
}}
"""

RELEVANCE_JUDGMENT = """\
You are a scientific literature relevance assessor. For each paper below, judge how likely \
it is to contain **specific numerical values** for the target parameter. Focus on whether the \
abstract suggests the paper reports measured, calibrated, or stated values — not just discusses \
the topic.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain context: {domain_context}
{enrichment_block}

Papers to judge:
{papers_block}

For EACH paper (by index), return:
- "relevance": "high" | "medium" | "low"
  - "high": abstract explicitly mentions numerical values, measurements, calibration results, \
or parameter tables for this parameter
  - "medium": abstract discusses the parameter or closely related measurements but does not \
explicitly state values
  - "low": abstract is tangentially related or about methodology/remote sensing without data
- "snippet": the most relevant sentence or phrase from the abstract (verbatim, max 200 chars). \
If nothing relevant, return empty string.

Return a JSON object mapping paper index (as string) to {{"relevance": "...", "snippet": "..."}}.
Return ONLY the JSON object, no other text.

Example — Parameter: "maximum leaf area index", Unit: "m2/m2"
{{
  "0": {{"relevance": "high", "snippet": "Peak LAI across six cultivars averaged 5.8 +/- 0.9 m2/m2"}},
  "1": {{"relevance": "low", "snippet": ""}},
  "2": {{"relevance": "medium", "snippet": "LAI was monitored throughout the growing season"}}
}}
"""

SEARCH_REFINEMENT = """\
You are a scientific literature search expert. The previous search found papers about the \
target parameter but failed to extract any numerical values. Your task is to diagnose the \
mismatch and generate better search queries.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain context: {domain_context}

Previous queries tried:
{previous_queries}

Papers found (summaries):
{paper_summaries}

High-relevance papers found so far (these contain or likely contain numerical values):
{high_relevance_papers}

Use these as examples to generate queries that find MORE papers like them.

Blackboard messages from other agents:
{blackboard_messages}

Diagnose why the previous queries found papers but no extractable values. Common reasons:
- Queries were too broad (found methodology papers, not data papers)
- Parameter is known by a different name or acronym in the literature
- Values are reported in different units or as part of composite parameters
- Need to target specific experimental or calibration studies

Then generate {n_queries} new, improved search queries that are MORE LIKELY to find papers \
containing explicit numerical values for this parameter. Avoid repeating previous queries.

Return a JSON object with:
- "diagnosis": string (why previous queries failed to yield values)
- "new_queries": [string, ...] (new search queries, 3-6 words each)
- "terminology_updates": [string, ...] (any new terms/synonyms discovered)

Return ONLY the JSON object, no other text.
"""

CROSS_ENRICHMENT_QUERIES = """\
You are a scientific literature search expert. Key papers have been identified that are \
highly relevant to the target parameter. Generate follow-up queries to find related work \
citing or extending these key papers.

Parameter: {name}
Description: {description}
Unit: {unit}
Domain context: {domain_context}

Key papers:
{key_papers}

Generate {n_queries} follow-up search queries based on these key papers. Focus on:
- Papers that cite these key papers
- Studies in similar experimental conditions
- More recent work that builds on these findings
- Related parameters or alternate measurement approaches

Keep queries SHORT (3-6 words) for best search results.

Return a JSON array of search query strings.
Return ONLY the JSON array, no other text.
"""

BATCH_VALUE_EXTRACTION = """\
You are a scientific data extraction assistant. Extract numerical values for the parameter \
described below from EACH of the following paper texts (which may be full papers or abstracts).

Parameter: {name}
Description: {description}
Unit: {unit}
Physical constraints: lower_bound={lower_bound}, upper_bound={upper_bound}
{context_block}
Papers:
{papers_block}

For EACH paper, extract all reported values, ranges, and uncertainties for this parameter. \
Include measured values, calibrated values, AND standard/default values stated as assumptions \
(e.g., "base temperature of 0°C was used"). These are all valid evidence for building a prior. \
If a paper reports values in a different unit, convert to {unit}. \
Extract ONLY explicitly stated values, do NOT estimate or guess.

Return a JSON object mapping paper index (as string) to an array of extracted values. \
Each value object has these fields:
- "reported_value": number or null (central/mean value)
- "reported_range": [min, max] or null
- "uncertainty": number or null (standard deviation or standard error)
- "sample_size": integer or null
- "context": string (brief note on experimental conditions)
- "extraction_confidence": "high" | "medium" | "low"

If a paper does not contain relevant values, map its index to an empty array.

Return ONLY the JSON object, no other text.

Example output:
{{
  "0": [{{"reported_value": 5.8, "uncertainty": 0.9,
          "sample_size": 36, "context": "field trial",
          "extraction_confidence": "high"}}],
  "1": [],
  "2": [{{"reported_value": 3.2,
          "reported_range": [2.1, 4.5],
          "context": "greenhouse",
          "extraction_confidence": "medium"}}],
  "3": [{{"reported_value": 95,
          "context": "species-level default phyllochron interval",
          "extraction_confidence": "high"}}]
}}
"""
