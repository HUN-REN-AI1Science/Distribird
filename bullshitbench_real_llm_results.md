# BullshitBench v2 — Real-LLM Run

Real end-to-end pipeline runs (LLM enrichment + Semantic Scholar + extraction + validity check)
against the refined ruleset (Rule 4b for uncertain empirical status, tightened Rule 5,
strengthened `PARAMETER_ENRICHMENT` prompt with red-flag list and second example).

## Summary — 6/6 verdicts match expected

| # | Parameter | Category | Expected | Verdict | Match | Papers | Values | Time |
|---|---|---|---|---|---|---|---|---|
| 1 | `mumblesnort_factor` | nonsense | likely_invalid | **likely_invalid** | ✓ | 60 | 0 | 8:13 |
| 2 | `fake_quantum_correction_xyz` | nonsense | likely_invalid | **likely_invalid** | ✓ | 99 | 0 | 13:58 |
| 3 | `biome_bgcmuso_carbon_pool_calibration_weight_v3` | theoretical | suspicious | **suspicious** | ✓ | 12 | 2 | 10:52 |
| 4 | `kalman_filter_state_covariance_q11` | theoretical | suspicious | **suspicious** | ✓ | 5 | 1 | 6:35 |
| 5 | `dssat_cropgro_root_growth_partition_factor_v45` | theoretical | suspicious | **suspicious** | ✓ | 16 | 1 | 55:38 |
| 6 | `specific_leaf_area` | real | valid | **valid** | ✓ | 74 | 7 | 22:50 |

**Total: 6/6 PASS — every category correctly classified by the refined ruleset.**

## Per-case detail

### `mumblesnort_factor` — nonsense (8m 13s)
- **Expected:** `likely_invalid`
- **Verdict:** `likely_invalid` ✓ (probe-driven)
- **Pipeline:** 60 papers (search retrieval was generous), 0 values extracted
- **Insight:** Even though search returned papers, all extractions yielded zero values
  and the LLM probe correctly flagged the name as fabricated.

### `fake_quantum_correction_xyz` — nonsense (13m 58s)
- **Expected:** `likely_invalid`
- **Verdict:** `likely_invalid` ✓ (probe-driven)
- **Pipeline:** 99 papers, 0 values extracted
- **Insight:** Same pattern as case 1: papers found but zero extractable values plus
  unrecognized parameter name → probe upgrades passive verdict to `likely_invalid`.

### `biome_bgcmuso_carbon_pool_calibration_weight_v3` — theoretical (10m 52s)
- **Expected:** `suspicious`
- **Verdict:** `suspicious` ✓
- **Pipeline:** 12 papers, 2 values extracted
- **Insight:** Despite 2 values being extracted (likely from model-description tables),
  the new tightened Rule 5 blocked `valid`: parameter name with version suffix
  (`_v3`) and software prefix (`biome_bgcmuso_`) makes it model-internal.

### `kalman_filter_state_covariance_q11` — theoretical (6m 35s)
- **Expected:** `suspicious`
- **Verdict:** `suspicious` ✓
- **Pipeline:** 5 papers, 1 value extracted
- **Insight:** Latent state covariance hyperparameter — exactly the kind of
  empirical-only model parameter Rule 4b/5 is designed to catch.

### `dssat_cropgro_root_growth_partition_factor_v45` — theoretical (55m 38s)
- **Expected:** `suspicious`
- **Verdict:** `suspicious` ✓
- **Pipeline:** 16 papers, 1 value extracted
- **Insight:** Software-version-specific calibration factor (`_v45`); long runtime
  came from extensive fulltext-fetch retries against paywalled DSSAT papers.

### `specific_leaf_area` — real (22m 50s) ✓ control case
- **Expected:** `valid`
- **Verdict:** `valid` ✓
- **Reason:** "Recognized parameter with literature-backed prior"
- **Empirical:** True
- **Pipeline:** 74 papers, 7 values extracted, **prior `beta` with HIGH confidence**
- **Insight:** Real, well-known empirically-measured parameter — passes Rule 5
  (recognized=True, empirical=not-False, values_extracted=7≥2, MEDIUM/HIGH confidence).
  The probe was NOT called (verdict was already VALID via passive heuristic).

## What this confirms about the refinement

1. **Rule 4b (uncertain empirical, papers + 0 values)** correctly catches model-internal
   parameters where the LLM is uncertain about empirical status.
2. **Tightened Rule 5** (require `is_recognized is True`, `is_empirical is not False`,
   `values_extracted >= MIN_VALUES_FOR_VALID`) prevents calibration weights from
   slipping through as `valid` when only a few values were incidentally extracted.
3. **The strengthened `PARAMETER_ENRICHMENT` prompt** (with red-flag list for version
   suffixes / software prefixes / "calibration"/"weight"/"latent" terms and a second
   worked example showing `empirically_measured: false`) helps the LLM correctly
   classify model-internal parameters even with version suffixes.
4. **The probe is invoked appropriately**: only for ambiguous SUSPICIOUS verdicts;
   it correctly upgrades clear nonsense to LIKELY_INVALID and confirms theoretical-only
   parameters as SUSPICIOUS rather than VALID.

## Total wall time

Cases 1–5 (cases 6 was rerun separately due to a sleep-mode interruption):
- Original run: ~2 hours
- Specific_leaf_area rerun: 22m 50s
- **Total real-LLM verification: ~2.5 hours, 6/6 PASS**
