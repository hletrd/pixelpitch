# Code Review (Cycle 45) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-44 fixes, focusing on NEW issues

## Previous Findings Status

C44-01 (FORMAT_TO_MM dead code) — COMPLETED. Dict and fmt code removed from cined.py.
C44-02 (fmt/fmt_m dead code) — COMPLETED. Format extraction regex removed.

## New Findings

### CR45-01: GSMArena _select_main_lens regex split breaks on decimal MP values — data corruption

**File:** `sources/gsmarena.py, line 82`
**Severity:** HIGH | **Confidence:** HIGH

The `re.split(r'(?=\b\d+(?:\.\d+)?\s*MP\b)', raw)` in `_select_main_lens` uses `\b` before `\d+`. The word boundary `\b` matches between the digit "0" and the decimal point "." in a number like "12.2 MP", causing the split to produce `['12.', '2 MP, f/1.7, ...']` instead of `['12.2 MP, f/1.7, ...']`. The function then selects "2 MP" as the main lens, producing mpix=2.0 instead of 12.2.

This affects any phone with a decimal-megapixel main camera (e.g., Google Pixel 7 at 12.2 MP, Samsung Galaxy S21 FE at 12.0 MP with decimal representation, various older phones). The bug causes three-fold data corruption:

1. **mpix wrong**: 12.2 becomes 2.0
2. **sensor type lost**: The fractional-inch format (e.g., `1/2.55"`) falls after the split point and loses its quote suffix, so `TYPE_FRACTIONAL_RE` no longer matches it
3. **derived.size wrong**: Without spec.type, derive_spec cannot compute sensor dimensions

The root cause is that `\b` (word boundary) fires between a digit and a period. The regex `(?=\b\d+(?:\.\d+)?\s*MP\b)` lookbehind hits `\b` at position before "0" in "12.2", then matches `\d+` = "0", which consumes "0" but not ".2", then the remaining "2 MP" starts a new split segment.

**Fix:** Remove the leading `\b` from the regex. The `(?=\d+(?:\.\d+)?\s*MP\b)` pattern will correctly match at the start of a decimal number because `\d+` is greedy and will consume all digits before the optional decimal part. Alternatively, use a negative lookbehind `(?<!\.)` to prevent matching a digit that's part of a decimal number.

---

## Summary

- CR45-01 (HIGH): GSMArena _select_main_lens regex split breaks on decimal MP values — data corruption
