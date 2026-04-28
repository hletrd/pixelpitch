# Code Reviewer Report — Cycle 53

**Date:** 2026-04-29
**HEAD:** `1c968dd` (cycle 52 docs commit)
**Scope:** Whole repository.

## Status of prior fixes

All cycles 1–52 fixes verified still in place.

- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  (matched_sensors + year + id parse-tolerance).

## New finding F53-01: `_safe_int_id` returns unbounded big-int on extreme float strings — LOW

- **File:** `pixelpitch.py:318-337`
- **Repro:** `_safe_int_id("1e308")` → 309-digit Python integer.
- **Why it's a problem:** `parse_existing_csv` stores the result on
  `SpecDerived.id`. `merge_camera_data` carries this through
  (`new_spec.id = existing_spec.id` at line 516) before sequential
  reassignment. Asymmetric with `_safe_year`, which has both an
  `isfinite` guard AND a 1900-2100 range guard. `_safe_int_id` has
  only the `isfinite` guard.
- **Failure scenario:** Excel rewrites a small integer column as
  scientific notation (`1.0E+308`) → next CI run parses, propagates a
  309-digit id through merge. Sequential reassignment in `main()`
  overwrites it before write_csv, so committed CSV is safe — but
  any code path that reads `spec.id` between parse and reassignment
  sees garbage. Defense-in-depth class with F52-01/F52-02.
- **Fix:** Add a sanity range guard to `_safe_int_id`: reject ids
  outside `[0, 10**6]`. The merge step regenerates ids in `[0, N)`
  where N is the camera count (~1000), so 10**6 is a comfortable
  upper bound. Mirror `_safe_year`'s post-conversion range check.
- **Confidence:** MEDIUM
- **Severity:** LOW (recoverable; sequential reassignment masks
  most failure modes)

## New finding F53-02: no test for `_safe_year` / `_safe_int_id` non-finite-from-string edge — LOW

- **File:** `tests/test_parsers_offline.py` (gap)
- **Detail:** Current year/id tolerance tests cover `"abc"`, `""`,
  `"2023.0"`, ` 2023 `. They do not exercise `"nan"`, `"inf"`,
  `"-inf"`, `"1e308"`. The implementation handles them via
  `isfinite`/range guards, but absence of tests means a future
  refactor could silently regress.
- **Fix:** Extend the year-tolerance and id-tolerance test sections
  with rows for those scientific-notation edges.
- **Confidence:** HIGH
- **Severity:** LOW (test gap; defense-in-depth)

## Sweep for commonly-missed issues

- Reviewed every `*.py` under `pixelpitch.py`, `models.py`,
  `sources/*.py`, `tests/*.py`. No new logic bugs found beyond F53-01
  and F53-02.
- `pixelpitch.py:1370` lines, still under the F32 threshold (1500
  lines) recorded in `deferred.md`.
- `_safe_float` already enforces `isfinite`, so all numeric columns
  except `record_id` are non-finite-safe.

## Confidence summary

| Finding | Confidence | Severity |
|---------|------------|----------|
| F53-01  | MEDIUM     | LOW      |
| F53-02  | HIGH       | LOW      |
