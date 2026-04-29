# Aggregate Review (Cycle 57) — Deduplicated, Merged Findings

**Date:** 2026-04-29
**HEAD:** `01c31d8` (after C56-01)
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic,
verifier, test-engineer, tracer, architect, debugger,
document-specialist, designer.

## Cycle 1–56 Status

All previous fixes confirmed still working at HEAD `01c31d8`. Both
gates pass:

- `flake8 .` → 0 errors (also enforced in CI).
- `python3 -m tests.test_parsers_offline` → all sections green
  (including the C54-01 `_load_per_source_csvs refresh against
  sensors.json`, C55-01 `_load_per_source_csvs cache fallback when
  sensors.json missing` and `parse_existing_csv BOM has_id
  detection`, and C56-01 `_load_per_source_csvs size-less row drops
  cache (sensors_db non-empty)` sections).

No regressions. Cycle 56's three findings (F56-01, F56-02, F56-03)
are fixed and verified.

## Cycle 57 New Findings

### F57-01 (BUG): `parse_existing_csv` accepts `area` column inconsistent with `width*height` — LOW

- **Flagged by:** code-reviewer (F57-CR-01), critic (F57-CRIT-02),
  verifier (F57-V-02 — reproduced), tracer (F57-T-01),
  architect (F57-A-01), debugger (F57-D-01), test-engineer
  (F57-TE-01), document-specialist (F57-DOC-01).
- **File:** `pixelpitch.py:413-426`
- **Detail:** When `parse_existing_csv` reads a CSV row that has
  both width and height present, it uses the `area` column verbatim
  rather than recomputing `area = width * height`. A hand-edited CSV
  whose width/height were corrected but whose area column was left
  stale will round-trip the wrong area through to the next CSV /
  HTML emit. `derive_spec` enforces `area = width * height` for
  fresh Specs, so `parse_existing_csv` is the only inconsistent
  path.
- **Repro:** Confirmed by verifier — input
  `1,Foo,dslr,,23.6,15.6,999.0,24.0,3.85,2020,` parses as
  `area=999.0` while `width*height=368.16`.
- **Fix:** In `parse_existing_csv`, when `width` and `height` are
  both present and finite-positive, recompute
  `area = width * height` (matching `derive_spec`); fall back to
  the `area_str` column only when `size` is missing. Update the
  docstring to document the area trust contract. Add a test
  asserting the parse-time recomputation.
- **Severity:** LOW. **Confidence:** HIGH.
- **Cross-agent agreement:** 8 reviewers (CRITICAL high signal).

### F57-02 (DOC): `match_sensors` megapixel-disagree branch undocumented — LOW

- **Flagged by:** code-reviewer (F57-CR-02), critic (F57-CRIT-03).
- **File:** `pixelpitch.py:242-251`
- **Detail:** When megapixels and sensor_megapixels are both
  present but disagree, the sensor is silently rejected. Behaviour
  is intentional (rejection > size-only match when both are known)
  but no comment explains it. Future refactors could accidentally
  add an `else: matches.append(...)` and break sensor matching.
- **Severity:** LOW. **Confidence:** HIGH.
- **Fix:** One-line comment.

### F57-03 (gap, deferred): direct unit tests for `match_sensors` — LOW

- **Flagged by:** test-engineer (F57-TE-02).
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** Defer; indirect coverage via the round-trip
  and refresh tests is sufficient.

### F57-04 (theoretical, deferred): semicolon inside sensor name fragments parse-back — LOW

- **Flagged by:** debugger (F57-D-06).
- **File:** `pixelpitch.py:441-443`, `pixelpitch.py:1003-1010`
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Defer until sensors.json actually contains a
  `;` in a name; pre-emptive code adds complexity for a
  hypothetical case.

## Carry-over deferred (no action this cycle)

- F32 monolith, F55-CRIT-03 / F56-CRIT-02 / F57-CRIT-01 test monolith.
- F49-04 perf, F55-PR-01..03 perf, F56-PR-04 / F57-PR-01..03 informational.
- C10-07 redirects, C10-08 debug port (F57-SR-old).
- F35..F40 UI carry-overs (re-confirmed by designer).
- F55-A-02 / F56-A-02 / F57-A-02 category list duplication.
- F55-04 (existing_specs in-place mutation), F55-05 (hand-edited
  blank-leading-cell defeats has_id).
- F56-DOC-03 / F57-DOC-03 (`deferred.md` size).
- F57-CR-03 (cosmetic comment redundancy in
  `_load_per_source_csvs`).

## Cross-Agent Agreement Matrix

| Finding   | Flagged By                                                                    | Severity |
|-----------|-------------------------------------------------------------------------------|----------|
| F57-01    | code-reviewer, critic, verifier, tracer, architect, debugger, test-engineer, document-specialist | LOW (high signal — 8 agents) |
| F57-02    | code-reviewer, critic                                                         | LOW |
| F57-03    | test-engineer                                                                 | LOW (defer) |
| F57-04    | debugger                                                                      | LOW (defer) |

## AGENT FAILURES

No agents failed.

## Summary statistics

- 11 reviewer perspectives executed.
- 4 findings produced this cycle:
  - 1 actionable bug (F57-01) flagged by 8/11 reviewers.
  - 1 actionable doc/comment (F57-02).
  - 2 deferred per repo policy (F57-03 indirect coverage, F57-04
    theoretical).
- 0 new HIGH/CRITICAL findings.
- 0 regressions vs cycle 56.
