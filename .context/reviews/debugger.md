# Debugger Review (Cycle 39) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## Previous Findings Status

DBG38-01 fixed. Template renders "unknown" for 0.0 pitch/mpix.

## New Findings

### DBG39-01: C38-01 fix incomplete — negative/NaN pitch still renders as numeric

**File:** `templates/pixelpitch.html`, lines 84-88
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** When a camera has `pitch=-1.0` (from corrupted CSV data through `_safe_float`):
1. Template renders "-1.0 µm" in the table cell (guard: `!= 0.0` → True)
2. JS `isInvalidData` returns true for `pitch < 0`
3. Row is hidden by default
4. User unchecks "Hide possibly invalid data" → sees "-1.0 µm" as if legitimate

**Failure mode for NaN pitch:**
1. Template renders "nan µm" (guard: `!= 0.0` → True, since NaN != 0.0)
2. JS `isInvalidData` catches `isNaN(parseFloat(...))` → hidden
3. But HTML source contains "nan µm" — malformed

This is the same class of bug as DBG38-01, but the C38-01 fix only addressed 0.0, not the broader set of invalid values.

**Fix:** Change `!= 0.0` to `> 0` in template guards.

---

## Summary

- DBG39-01 (MEDIUM): C38-01 fix incomplete — negative/NaN pitch still renders as numeric in template
