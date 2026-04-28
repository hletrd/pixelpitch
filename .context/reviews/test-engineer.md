# Test Engineer — Cycle 54

**HEAD:** `93851b0`

## Inventory

- `tests/test_parsers_offline.py` (2283 lines): per-source HTML
  fixture parsers, matched_sensors round-trip + parse-tolerance,
  year + id parse-tolerance.
- `tests/test_sources.py` (111 lines): network-dependent smoke
  tests; not in the gate.

## Findings

### F54-T01 — No test asserts `_load_per_source_csvs` behavior — LOW

- **File:** `tests/test_parsers_offline.py` (gap)
- **Severity:** LOW | **Confidence:** HIGH
- **Description:** `_load_per_source_csvs` clears row ids and trusts
  matched_sensors verbatim. There is no test that exercises this
  function. Specifically, no test asserts:
  - Per-row `id` is cleared after load.
  - `matched_sensors` is preserved (current behavior) OR refreshed
    (expected behavior under F54-01 fix).
  - Missing per-source CSV does not raise.
- **Fix:** Add a small unit test that writes a temp CSV with a known
  matched_sensors column and asserts the loaded `SpecDerived`
  reflects the chosen semantics.

### F54-T02 — No test for stale matched_sensors merge scenario — LOW

- **File:** `tests/test_parsers_offline.py` (gap)
- **Severity:** LOW | **Confidence:** MEDIUM
- **Description:** Tests cover preservation when new_spec
  matched_sensors is None (C46), but no test exercises the case
  where new_spec has stale matched_sensors that should be replaced
  with current `sensors.json` matches.
- **Fix:** Test will be relevant once F54-01 lands.

## Final sweep

No regressions. New findings are scoped to F54-01.
