# Debugger Review (Cycle 57)

**Reviewer:** debugger
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## Latent failure mode sweep

### F57-D-01: `parse_existing_csv` accepts hand-edited area inconsistent with width*height — LOW

- **File:** `pixelpitch.py:413-426`
- **Failure mode:** Stale area persists across deploys when a user
  edits width/height but leaves area; emits an inconsistent CSV row
  next round-trip.
- **Severity:** LOW. **Confidence:** HIGH.
- **Same as:** F57-CR-01.

### F57-D-02: `_safe_float("inf")` rejected, `_safe_float("Infinity")` rejected — VERIFIED

- Tested by the existing C40 finite test. OK.

### F57-D-03: BOM detection on first column with `id` header — VERIFIED

- C55-01 test pins it. OK.

### F57-D-04: `_load_per_source_csvs` size-less branch drops cache — VERIFIED

- C56-01 test pins it. OK.

### F57-D-05: empty CSV / single-line CSV → no rows + no exception

- Tested via `parse_existing_csv` empty/header-only sections. OK.

### F57-D-06: matched_sensors with semicolons inside sensor name — LOW (theoretical)

- **File:** `pixelpitch.py:441-443`, `pixelpitch.py:1003-1010`
- **Detail:** Parse uses `;` as delimiter. write_csv comments warn
  "if a future entry violates this, drop the offending element and
  warn rather than silently fragmenting on parse-back" but does
  not implement the drop+warn — currently emits the name with
  semicolons, which fragments on parse-back.
- **Severity:** LOW. **Confidence:** HIGH (the comment exists but
  the behaviour does not).
- **Disposition:** Defer until sensors.json actually has a `;` in
  a name. Currently no such entry exists; pre-emptive code adds
  complexity for a hypothetical case.

## Carry-over

- F49-04 perf, F55-PR-01..03, etc. — re-defer.

## Confidence summary

- 1 LOW actionable (F57-D-01 = F57-CR-01).
- 1 LOW deferred (F57-D-06 semicolon-in-sensor-name).
