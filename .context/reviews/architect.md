# Architect Review (Cycle 45) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## Previous Findings Status

ARCH44-01 (FORMAT_TO_MM false coupling) — COMPLETED. Removed.
ARCH44-02 (CineD format extraction wasted computation) — COMPLETED. Removed.

## New Findings

### ARCH45-01: GSMArena _select_main_lens regex split uses word boundary that breaks on decimal numbers — architectural fragility

**File:** `sources/gsmarena.py, _select_main_lens, line 82`
**Severity:** HIGH | **Confidence:** HIGH

The `_select_main_lens` function splits camera lens entries using a regex with `\b` (word boundary). This is architecturally fragile because it assumes MP values are always integers at word boundaries. In reality, decimal MP values (12.2 MP, 10.7 MP) are common in smartphone specs. The word boundary `\b` fires between a digit and a decimal point, causing the split to occur inside the decimal number rather than before it.

This is a parsing/architectural issue: the function's contract ("pick the main wide lens") is violated when the input contains decimal MP values, because the split corrupts the data before the selection logic runs. The architecture has no validation step to detect corrupted fragments (e.g., a fragment that starts with a bare number not followed by "MP").

**Fix:** Remove `\b` from the start of the split regex. Consider adding a validation step after splitting to reject fragments that don't start with a complete "N MP" pattern (i.e., the first token must be a number immediately followed by "MP").

---

## Summary

- ARCH45-01 (HIGH): GSMArena _select_main_lens regex split uses word boundary that breaks on decimal numbers
