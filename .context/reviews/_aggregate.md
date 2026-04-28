# Aggregate Review (Cycle 54) — Deduplicated, Merged Findings

**Date:** 2026-04-29
**HEAD:** `93851b0`
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic,
verifier, test-engineer, tracer, architect, debugger, document-specialist,
designer.

## Cycle 1–53 Status

All previous fixes confirmed still working. Both gates pass at HEAD
`93851b0`:

- `flake8 .` → 0 errors (also enforced in CI).
- `python3 -m tests.test_parsers_offline` → all sections green.

No regressions detected.

## Cycle 54 New Findings

### F54-01 (consensus): `_load_per_source_csvs` does not refresh `matched_sensors` against current `sensors.json` — LOW

- **Flagged by:** code-reviewer, critic, verifier, tracer, architect
  (as F54-A01), debugger (as F54-D01), document-specialist (as
  F54-DOC-01).
- **File:** `pixelpitch.py:1028-1053`
- **Repro:** Per-source CSV is written with `matched_sensors=["S0"]`;
  `sensors.json` is later edited (S0 renamed/removed); next
  `python pixelpitch.py html dist` run leaves the stale `S0` in
  the merged output because `_load_per_source_csvs` parses the file
  verbatim and `merge_camera_data` only re-matches **existing-only**
  cameras (not those re-introduced via per-source CSVs).
- **Fix options:**
  1. Refresh: after `parse_existing_csv`, call `derive_spec(d.spec,
     sensors_db)` (with lazy-load) to re-compute matched_sensors.
  2. Drop on load: set `d.matched_sensors = None` and let
     `merge_camera_data` fall back to existing CSV matches (which
     may also be stale, so this is weaker).
  Option 1 is preferred. Update the `_load_per_source_csvs`
  docstring to declare per-source CSVs as caches.
- **Severity:** LOW (informational column, self-healing on next
  per-source fetch).
- **Confidence:** MEDIUM (architectural inconsistency, no observed
  user impact yet).

### F54-T01: no test asserts `_load_per_source_csvs` semantics — LOW

- **Flagged by:** test-engineer.
- **File:** `tests/test_parsers_offline.py` (gap).
- **Detail:** No coverage for per-source CSV load behavior:
  id-clearing, matched_sensors-handling, missing-file tolerance.
- **Fix:** Add a temp-file based unit test exercising
  `_load_per_source_csvs`. Should be authored alongside F54-01 fix
  to lock in the chosen semantics.
- **Severity:** LOW (test gap).
- **Confidence:** HIGH.

### F54-02 (cosmetic, low confidence): `merge_camera_data` overwrites a valid new id with a None existing id — LOW

- **Flagged by:** code-reviewer.
- **File:** `pixelpitch.py:524`
- **Detail:** `new_spec.id = existing_spec.id` blindly copies the id
  even when `existing_spec.id is None`. Mitigated by sequential
  reassignment at line 623-624.
- **Severity:** LOW (mitigated).
- **Confidence:** LOW.

### F54-DOC-01: `_load_per_source_csvs` docstring "caches" claim contradicts implementation — LOW (cosmetic)

- **Flagged by:** document-specialist.
- **File:** `pixelpitch.py:1031-1033`
- **Fix:** Update docstring to describe whichever semantics F54-01
  picks. Gate-bound to F54-01.
- **Severity:** LOW (cosmetic).

### F54-DOC-02: `merge_camera_data` docstring does not document matched_sensors preservation — LOW (cosmetic)

- **Flagged by:** document-specialist.
- **File:** `pixelpitch.py:475-497`
- **Fix:** Add a paragraph describing the C46 behavior (preserve
  existing matched_sensors when new is None; treat [] as
  authoritative).
- **Severity:** LOW (cosmetic).

## Cross-Agent Agreement Matrix

| Finding   | Flagged By                                                                            | Highest Severity |
|-----------|---------------------------------------------------------------------------------------|------------------|
| F54-01    | code-reviewer, critic, verifier, tracer, architect, debugger, document-specialist     | LOW (HIGH consensus on existence) |
| F54-T01   | test-engineer                                                                         | LOW (test gap)   |
| F54-02    | code-reviewer                                                                         | LOW (cosmetic)   |
| F54-DOC-01| document-specialist                                                                   | LOW (cosmetic)   |
| F54-DOC-02| document-specialist                                                                   | LOW (cosmetic)   |

## AGENT FAILURES

No agents failed.

## Summary statistics

- 11 reviewer perspectives executed.
- 5 findings produced this cycle (1 actionable correctness/architecture,
  1 actionable test gap, 1 cosmetic id-overwrite, 2 doc/code mismatches
  gate-bound to F54-01).
- 0 new HIGH/CRITICAL findings.
- 0 deferred items added.
