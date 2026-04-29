# Test-Engineer Review (Cycle 58, orchestrator cycle 11)

**Date:** 2026-04-29
**HEAD:** `aef726b`

## Coverage check

- `parse_existing_csv` — 25+ assertions; F57-01 area
  recomputation now pinned.
- `merge_camera_data` — 12+ assertions; matched_sensors
  preservation pinned.
- `_safe_year`, `_safe_int_id` — boundary parsing pinned for
  garbage / inf / nan / range overflow.
- `match_sensors` — indirectly covered via round-trip and
  refresh tests; **direct unit test still missing** (F57-03
  deferred).
- `_load_per_source_csvs` — cache-fallback and refresh paths
  pinned.

## New findings

### F58-TE-01: no test for `--limit` validation in `source` command — LOW

- **File:** `tests/test_parsers_offline.py` (gap)
- **Detail:** F58-CR-01 (negative `--limit` accepted silently)
  has no test pinning the new validation behavior. After the
  fix, a test should assert that `--limit -1` and `--limit 0`
  exit with a non-zero status and a clear error message.
- **Severity:** LOW. **Confidence:** HIGH.
- **Fix:** call `main()` with a patched `sys.argv` that
  includes `--limit -1`, capture the `SystemExit`, assert
  non-zero exit code. Subprocess-style would be too heavy
  for the offline test gate.

### F58-TE-02 (deferred): direct unit tests for `match_sensors` — LOW

- Carry-over of F57-03. Indirect coverage via round-trip
  tests is sufficient for current scope.
- **Disposition:** keep deferred.

## Summary

One new actionable test gap (F58-TE-01), conditional on the
F58-CR-01 fix. Deferred carry-over preserved.
