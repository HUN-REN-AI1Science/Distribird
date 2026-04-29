# BullshitBench — Real-LLM Run

Real end-to-end pipeline runs (LLM enrichment + Semantic Scholar + extraction + validity check)

## Summary — 4/4 verdicts match expected

| Parameter | Category | Expected | Verdict | Match | Papers | Values | Time |
|---|---|---|---|---|---|---|---|
| `mumblesnort_factor` | nonsense | likely_invalid | likely_invalid | OK | 0 | 0 | 50.8s |
| `fake_quantum_correction_xyz` | nonsense | likely_invalid | likely_invalid | OK | 0 | 0 | 46.9s |
| `biome_bgcmuso_carbon_pool_calibration_weight_v3` | theoretical | suspicious | suspicious | OK | 8 | 0 | 560.2s |
| `specific_leaf_area` | real | valid | valid | OK | 80 | 4 | 1954.2s |

## Per-case details

### `mumblesnort_factor` — nonsense
- **Description:** A fabricated coefficient that does not exist in any literature
- **Domain:** general scientific testing
- **Expected:** `likely_invalid`
- **Verdict:** `likely_invalid` — MATCHES expected
- **Reason:** No literature found across 5 refined queries
- **Empirical:** False
- **Pipeline:** 0 papers, 0 values, prior `truncated_normal` (confidence: none, informative: False)
- **LLM recognized:** False (confidence: none)
- **LLM empirically measured:** False
- **Terminology:** fabricated parameter, dummy variable, placeholder coefficient
- **Signals:**
  ```json
  {
    "papers_found": 0,
    "values_extracted": 0,
    "n_queries_tried": 5,
    "is_recognized_parameter": false,
    "recognition_confidence": "none",
    "empirically_measured": false,
    "n_terminology": 3,
    "prior_is_informative": false,
    "prior_confidence": "none"
  }
  ```
- **Warnings:**
  - The requested parameter 'mumblesnort_factor' is fabricated and does not exist in scientific literature.
  - All provided papers were excluded because they are completely unrelated to the target parameter.
  - No informative evidence found; using uninformative prior.
  - Parameter validity: LIKELY INVALID — No literature found across 5 refined queries
- **Elapsed:** 50.8s

### `fake_quantum_correction_xyz` — nonsense
- **Description:** A made-up quantum correction term with no scientific basis
- **Domain:** theoretical physics
- **Expected:** `likely_invalid`
- **Verdict:** `likely_invalid` — MATCHES expected
- **Reason:** No literature found across 5 refined queries
- **Empirical:** False
- **Pipeline:** 0 papers, 0 values, prior `truncated_normal` (confidence: none, informative: False)
- **LLM recognized:** False (confidence: none)
- **LLM empirically measured:** False
- **Terminology:** radiative correction, loop correction, quantum correction, higher-order correction, renormalization constant
- **Signals:**
  ```json
  {
    "papers_found": 0,
    "values_extracted": 0,
    "n_queries_tried": 5,
    "is_recognized_parameter": false,
    "recognition_confidence": "none",
    "empirically_measured": false,
    "n_terminology": 5,
    "prior_is_informative": false,
    "prior_confidence": "none"
  }
  ```
- **Warnings:**
  - The requested parameter is fictitious and has no physical equivalent, meaning no valid scientific literature can be used to build an empirical Bayesian prior for it.
  - No informative evidence found; using uninformative prior.
  - Parameter validity: LIKELY INVALID — No literature found across 5 refined queries
- **Elapsed:** 46.9s

### `biome_bgcmuso_carbon_pool_calibration_weight_v3` — theoretical
- **Description:** Internal calibration weight from Biome-BGCMuSo model version 3.x; purely a model-internal tuning parameter
- **Domain:** Biome-BGCMuSo crop modeling
- **Expected:** `suspicious`
- **Verdict:** `suspicious` — MATCHES expected
- **Reason:** The parameter is a model-internal calibration weight specific to a particular model version and is not an empirically measured scientific quantity.
- **Empirical:** False
- **Pipeline:** 8 papers, 0 values, prior `truncated_normal` (confidence: none, informative: False)
- **LLM recognized:** False (confidence: none)
- **LLM empirically measured:** False
- **Terminology:** calibration parameter, tuning weight, scaling factor, empirical adjustment factor, model coefficient
- **Signals:**
  ```json
  {
    "papers_found": 8,
    "values_extracted": 0,
    "n_queries_tried": 15,
    "is_recognized_parameter": false,
    "recognition_confidence": "none",
    "empirically_measured": false,
    "n_terminology": 5,
    "prior_is_informative": false,
    "prior_confidence": "none"
  }
  ```
- **LLM probe:** verdict=`suspicious`, reason='The parameter is a model-internal calibration weight specific to a particular model version and is not an empirically measured scientific quantity.'
- **Warnings:**
  - None of the abstracts explicitly mention the exact parameter 'biome_bgcmuso_carbon_pool_calibration_weight_v3' or its numerical value.
  - The parameter is likely an internal tuning weight that may only be found in the supplementary materials, model code, or detailed methodology sections of papers [1], [2], [3], and [4].
  - Search refinement round 1: generated 5 new queries.
  - The specific parameter 'biome_bgcmuso_carbon_pool_calibration_weight_v3' may be an obsolete or highly specific internal tuning weight from version 3.x, whereas most recent literature covers versions 4.0 to 6.2 ([1], [5]).
  - None of the abstracts explicitly report numerical values for this specific v3.x calibration weight, so full-text review of the model description papers (e.g., [5]) will be necessary.
  - Search refinement round 2: generated 5 new queries.
  - None of the abstracts explicitly mention the specific 'biome_bgcmuso_carbon_pool_calibration_weight_v3' parameter or its numerical value.
  - The parameter is likely an internal tuning weight that may only be found in the supplementary materials, model code, or detailed methodology sections of papers like [2], [3], or [6].
  - No informative evidence found; using uninformative prior.
  - Parameter validity: suspicious — The parameter is a model-internal calibration weight specific to a particular model version and is not an empirically measured scientific quantity.
- **Elapsed:** 560.2s

### `specific_leaf_area` — real
- **Description:** Leaf area per unit dry mass of leaves
- **Domain:** maize crop modeling
- **Expected:** `valid`
- **Verdict:** `valid` — MATCHES expected
- **Reason:** Recognized parameter with literature-backed prior
- **Empirical:** True
- **Pipeline:** 80 papers, 4 values, prior `truncated_normal` (confidence: medium, informative: True)
- **LLM recognized:** True (confidence: high)
- **LLM empirically measured:** True
- **Terminology:** Specific leaf area (SLA), Leaf mass per area (LMA), Specific leaf weight (SLW), Leaf area-to-mass ratio
- **Signals:**
  ```json
  {
    "papers_found": 80,
    "values_extracted": 3,
    "n_queries_tried": 15,
    "is_recognized_parameter": true,
    "recognition_confidence": "high",
    "empirically_measured": true,
    "n_terminology": 4,
    "prior_is_informative": true,
    "prior_confidence": "medium"
  }
  ```
- **Warnings:**
  - Crop modeling papers like [3], [4], and [5] might use default specific leaf area values from model documentation rather than measuring them directly in the field.
  - Papers [2] and [6] might report specific leaf area as an intermediate variable rather than the main focus, requiring careful extraction.
  - Search refinement round 1: generated 5 new queries.
  - Papers [3] and [4] lack DOIs and abstracts, which may make full-text retrieval difficult, though their titles are highly relevant.
  - None of the provided abstracts contain explicit numerical values for SLA, meaning full-text review will be required to extract the actual parameter values.
  - Used web-assisted extraction to look up paper content online.
  - Search refinement round 2: generated 5 new queries.
  - Most selected papers do not explicitly state numerical SLA values in their abstracts, requiring full-text review.
  - Some papers (like [3] and [4]) may report Leaf Mass per Area (LMA) or Leaf Dry Matter Content instead of SLA; LMA is the inverse of SLA and will require conversion.
  - Paper [5] is a global vegetation model, so its maize SLA parameter might be a generic crop functional type default rather than a specifically calibrated value for a local context.
- **Elapsed:** 1954.2s
