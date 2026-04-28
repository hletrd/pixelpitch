# Critic Review (Cycle 45) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## Previous Findings Status

CRIT44-01 (FORMAT_TO_MM dead code) — COMPLETED. Removed.

## New Findings

### CRIT45-01: GSMArena _select_main_lens regex split corrupts decimal MP data — high-severity data integrity bug

**File:** `sources/gsmarena.py, line 82`
**Severity:** HIGH | **Confidence:** HIGH

The `re.split(r'(?=\b\d+(?:\.\d+)?\s*MP\b)', raw)` regex uses `\b` (word boundary) before `\d+`. This causes a split at the boundary between a digit and a decimal point, breaking "12.2 MP" into "12." and "2 MP". When `_select_main_lens` sorts and picks the lowest-priority (main) lens, it selects the corrupted fragment "2 MP, f/1.7, ..." instead of the correct "12.2 MP, f/1.7, ...".

This is a high-severity data integrity bug because:
1. It silently corrupts mpix values (12.2 -> 2.0)
2. It causes TYPE_FRACTIONAL_RE to fail matching because the quote suffix on the sensor format (e.g., `1/2.55"`) gets severed from the numeric prefix
3. Without spec.type, derive_spec cannot compute sensor dimensions from the type lookup

The bug affects all phones whose main camera has a decimal-megapixel value, including the entire Google Pixel line (Pixel 1-6 at 12.2 MP), various Samsung phones with 12.0 MP decimal representation, and many older smartphones.

**Fix:** Remove the leading `\b` from the split regex. Change to `r'(?=\d+(?:\.\d+)?\s*MP\b)'`. This allows the greedy `\d+` to consume the full integer part of a decimal number before the optional `.\\d+` group matches the fractional part, preventing the split from occurring inside a decimal number.

---

## Summary

- CRIT45-01 (HIGH): GSMArena _select_main_lens regex split corrupts decimal MP data
