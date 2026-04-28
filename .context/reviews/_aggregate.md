# Aggregate Review (Cycle 55) — Deduplicated, Merged Findings

**Date:** 2026-04-29
**HEAD:** `f08c3c4` (after C54-01)
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic,
verifier, test-engineer, tracer, architect, debugger,
document-specialist, designer.

## Cycle 1–54 Status

All previous fixes confirmed still working at HEAD `f08c3c4`. Both
gates pass:

- `flake8 .` → 0 errors (also enforced in CI).
- `python3 -m tests.test_parsers_offline` → all sections green
  (including the new C54-01 `_load_per_source_csvs refresh
  against sensors.json` section).

No regressions.

## Cycle 55 New Findings

### F55-01 (consensus): `_load_per_source_csvs` drops cached matched_sensors when sensors.json fails to load — LOW

- **Flagged by:** code-reviewer (F55-CR-02 partial), critic
  (F55-CRIT-01), verifier (F55-V-01), test-engineer (F55-TE-01),
  tracer (F55-T-01), architect (F55-A-01), debugger (F55-D-01),
  document-specialist (F55-DOC-01).
- **File:** `pixelpitch.py:1074-1086`
- **Detail:** When `load_sensors_database()` returns `{}` (file
  missing or invalid JSON), `_load_per_source_csvs` resets every
  parsed row's matched_sensors to `None`, discarding the cache. By
  contrast, `merge_camera_data`'s existing-only branch (line 615-628)
  leaves matched_sensors untouched when sensors_db is empty. The two
  paths disagree on cache fallback semantics.
- **Fix:** In `_load_per_source_csvs`, when sensors_db is empty,
  leave the parsed matched_sensors untouched (keep the cache as a
  fallback). When sensors_db is non-empty, refresh as today. Update
  the docstring. Add a unit test that mocks `load_sensors_database`
  to return `{}` and asserts the cache is preserved.
- **Severity:** LOW (sensors.json failure is rare; in CI the file
  is always present).
- **Confidence:** MEDIUM.

### F55-02 (test gap): no test for `match_sensors` boundary tolerance — LOW

- **Flagged by:** verifier, test-engineer.
- **File:** `tests/test_parsers_offline.py` (gap).
- **Detail:** `match_sensors` uses `<= 2%` size, `<= 5%` mpix. No
  test pins the boundary behavior.
- **Severity:** LOW (test gap). **Confidence:** HIGH.

### F55-03 (test gap): no direct BOM test for `parse_existing_csv` — LOW

- **Flagged by:** test-engineer.
- **File:** `tests/test_parsers_offline.py` (gap).
- **Detail:** Add regression test prepending `﻿` to a CSV and
  asserting has_id detection still works.
- **Severity:** LOW. **Confidence:** HIGH.

### F55-04 (latent contract): `merge_camera_data` mutates input `existing_specs` items — LOW

- **Flagged by:** code-reviewer (F55-CR-03).
- **File:** `pixelpitch.py:622-628`
- **Detail:** `existing_spec.matched_sensors = ...` mutates the
  caller's list items. No current caller observes the mutation, but
  the contract is brittle.
- **Severity:** LOW (latent). **Confidence:** HIGH.
- **Disposition:** Defer; gated on a future caller actually
  reusing `existing_specs`.

### F55-05 (parse robustness): hand-edited blank-leading-cell defeats `has_id` detection — LOW

- **Flagged by:** code-reviewer (F55-CR-01).
- **File:** `pixelpitch.py:371-372`
- **Detail:** A spreadsheet user inserting a blank column before id
  defeats `header[0] == "id"`.
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** Defer; user-edited CSVs are out of scope per
  current data flow (CSVs are produced by write_csv).

### F55-06 (cleanup): `_load_per_source_csvs` and `merge_camera_data` should share a refresh helper — LOW

- **Flagged by:** architect (F55-A-01).
- **Files:** `pixelpitch.py:615-628`, `pixelpitch.py:1074-1086`
- **Disposition:** Bundled into F55-01 fix. After resolving F55-01,
  consider extracting `_refresh_matched_sensors`.

### F55-DOC-02: README does not mention smartphone / cinema pages — LOW

- **Flagged by:** document-specialist.
- **File:** `README.md`
- **Severity:** LOW. **Confidence:** MEDIUM.

## Cross-Agent Agreement Matrix

| Finding   | Flagged By                                                                                  | Severity |
|-----------|---------------------------------------------------------------------------------------------|----------|
| F55-01    | code-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, document-specialist | LOW (HIGH consensus on existence) |
| F55-02    | verifier, test-engineer                                                                     | LOW |
| F55-03    | test-engineer                                                                               | LOW |
| F55-04    | code-reviewer                                                                               | LOW |
| F55-05    | code-reviewer                                                                               | LOW |
| F55-06    | architect (gated on F55-01)                                                                 | LOW |
| F55-DOC-02 | document-specialist                                                                        | LOW |

## AGENT FAILURES

No agents failed.

## Summary statistics

- 11 reviewer perspectives executed.
- 7 findings produced this cycle (1 actionable
  correctness/consistency, 2 actionable test gaps, 4 cosmetic /
  cleanup / deferral candidates).
- 0 new HIGH/CRITICAL findings.
