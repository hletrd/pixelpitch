# Aggregate Review (Cycle 59, orchestrator cycle 12) — Deduplicated, Merged Findings

**Date:** 2026-04-29
**HEAD:** `fa0ae66` (after C58-01 plan-completed marker)
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic,
verifier, test-engineer, tracer, architect, debugger,
document-specialist, designer.

## Cycle 1-58 Status

All previous fixes confirmed still working at HEAD `fa0ae66`.
Both gates pass:

- `flake8 .` -> 0 errors (also enforced in CI).
- `python3 -m tests.test_parsers_offline` -> all sections green.

No regressions. Cycle 58's findings (F58-01 `--limit` validation
+ F58-02 help-text doc + F58-03 test) are fixed and verified.

## Cycle 59 New Findings

### F59-01 (defensive-parity, LOW): `write_csv` width/height columns lack the isfinite/positive guards used for area/mpix/pitch

- **Flagged by:** code-reviewer (F59-CR-01), critic
  (F59-CRIT-01), verifier (reproduced), test-engineer
  (F59-TE-01 paired test gap), debugger (F59-D-01),
  architect (F59-A-01), document-specialist (F59-DOC-01).
- **File:** `pixelpitch.py:1018-1019`
- **Detail:** Lines 1020-1022 already guard `area_str`,
  `mpix_str`, and `pitch_str` against non-finite (`inf`/`nan`)
  and non-positive (<=0) values. Lines 1018-1019 only check
  truthiness of `derived.size`, so a hypothetical
  `(0.0, 0.0)`, `(-1.0, -1.0)`, `(inf, inf)`, or
  `(nan, nan)` SpecDerived.size tuple would write "0.00,0.00",
  "-1.00,-1.00", "inf,inf", or "nan,nan" to the CSV. Today
  upstream guards (`derive_spec` line 900, `parse_existing_csv`
  line 430-433) prevent the pathological tuple from reaching
  `write_csv`, so this is a defensive-parity gap rather than
  a live bug. Hardening at the write boundary co-locates the
  contract enforcement and survives future regressions in
  derive_spec or new direct-SpecDerived construction sites.
- **Repro (verifier):** synthetic
  `SpecDerived(size=(0.0, 0.0), area=0.0, ...)` written via
  `write_csv` produces a CSV row containing `0.00,0.00` for
  the width/height cells (area cell correctly empty).
- **Fix:** mirror the area/mpix/pitch guard:

  ```python
  if (derived.size
      and isfinite(derived.size[0]) and derived.size[0] > 0
      and isfinite(derived.size[1]) and derived.size[1] > 0):
      width_str = f"{derived.size[0]:.2f}"
      height_str = f"{derived.size[1]:.2f}"
  else:
      width_str = ""
      height_str = ""
  ```
- **Severity:** LOW. **Confidence:** HIGH.
- **Cross-agent agreement:** 7 reviewers (HIGH signal).

### F59-02 (test gap, LOW): no test pins write_csv width/height non-finite/non-positive guards

- **Flagged by:** test-engineer (F59-TE-01).
- **File:** `tests/test_parsers_offline.py` (gap)
- **Detail:** F59-01 fix needs a regression test pinning the
  validation behavior. Add a test section
  `write_csv width/height non-finite/non-positive guards`
  with sub-tests for `inf`, `nan`, `0.0`, negative, and
  sanity cases.
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Schedule alongside F59-01.

### F59-03 (DOC, LOW): write_csv docstring does not document float-cell contract

- **Flagged by:** document-specialist (F59-DOC-01).
- **File:** `pixelpitch.py:1000-1001`
- **Detail:** Expand the write_csv docstring to document the
  no-inf/no-nan/no-zero/no-negative float-cell contract for
  all five numeric columns (width, height, area, mpix,
  pitch). Pair with F59-01 fix.
- **Severity:** LOW. **Confidence:** HIGH.

### F59-04 (deferred, informational): per-source CSV "missing" log line wording

- **Flagged by:** code-reviewer (F59-CR-02), critic
  (F59-CRIT-02), debugger (F59-D-02).
- **File:** `pixelpitch.py:1085`
- **Detail:** "missing" wording could be softer ("no cached
  CSV at ..."). Informational, no behavior change.
- **Disposition:** Defer (informational).

## Carry-over deferred (no action this cycle)

- F32 monolith, F55-CRIT-03 / F56-CRIT-02 / F57-CRIT-01 /
  F58-CRIT-02 test monolith.
- F49-04 perf, F55-PR-01..03 perf, F56-PR-04 / F57-PR-01..03
  / F59-PR-01 informational.
- C10-07 redirects, C10-08 debug port.
- F35..F40 UI carry-overs (re-confirmed by designer).
- F55-A-02 / F56-A-02 / F57-A-02 / F58-A-02 category list
  duplication / argparse drift.
- F55-04 (existing_specs in-place mutation), F55-05
  (hand-edited blank-leading-cell defeats has_id).
- F56-DOC-03 / F57-DOC-03 / F58-DOC-02 (`deferred.md` size).
- F57-CR-03, F57-D-06, F58-CR-03 (informational).
- F58-04 (`--out --limit` typo tolerance), F58-05 (argparse
  refactor), F58-06 (boundary tests).

## Cross-Agent Agreement Matrix

| Finding | Flagged By | Severity |
|---------|------------|----------|
| F59-01  | code-reviewer, critic, verifier, test-engineer, debugger, architect, document-specialist | LOW (high signal — 7 agents) |
| F59-02  | test-engineer | LOW |
| F59-03  | document-specialist | LOW |
| F59-04  | code-reviewer, critic, debugger | LOW (defer) |

## AGENT FAILURES

No agents failed.

## Summary statistics

- 11 reviewer perspectives executed.
- 4 findings produced this cycle:
  - 1 actionable defensive-parity bug (F59-01) flagged by
    7/11 reviewers.
  - 1 actionable test gap (F59-02), pairs with F59-01.
  - 1 actionable doc fix (F59-03), pairs with F59-01.
  - 1 deferred informational (F59-04 log wording).
- 0 new HIGH/CRITICAL findings.
- 0 regressions vs cycle 58.
