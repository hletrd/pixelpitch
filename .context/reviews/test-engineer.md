# test-engineer Review (Cycle 51)

**Date:** 2026-04-29
**HEAD:** 3b35dcc

## Inventory

- `tests/test_parsers_offline.py` (2155 LOC): comprehensive, runs as a script (no pytest).
- `tests/test_sources.py` (111 LOC): live-fetch, optional.
- `tests/fixtures/`: HTML fixtures.
- Round-trip tests for matched_sensors verbatim and `;`-guard added cycle 50.

## Findings

### F51-T-01: No round-trip test for whitespace in `matched_sensors` tokens — LOW / MEDIUM
- **File:** `tests/test_parsers_offline.py`
- **Detail:** Pairs with code-reviewer F51-01. If `parse_existing_csv` is updated to strip
  whitespace (recommended fragility-defense), an asymmetry test would catch regressions.
  Currently no test asserts that hand-edited CSV with `IMX455; IMX571` yields exactly
  `["IMX455", "IMX571"]` after parse.
- **Fix:** Add a minimal test that parses a synthetic CSV row containing `IMX455; IMX571 ; IMX989`
  in the matched_sensors column via `parse_existing_csv`, and asserts the parsed list is
  `["IMX455", "IMX571", "IMX989"]`. Companion to the F51-01 strip fix.
- **Confidence:** HIGH (test gap is real; prerequisite is the F51-01 strip fix)
- **Severity:** LOW

## No other test gaps identified this cycle.
