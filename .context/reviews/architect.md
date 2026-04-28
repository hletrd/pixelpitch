# Architect Review (Cycle 27) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes

## Previous Findings Status

ARCH26-01 (MPIX_RE centralization) addressed in C26. All previously identified architectural concerns remain deferred (LOW severity).

## New Findings

### ARCH27-01: PITCH_UM_RE incomplete as single source of truth — missing "um" variant

**File:** `sources/__init__.py`, line 66
**Severity:** LOW | **Confidence:** HIGH

The C25-01 centralization established `sources/__init__.py` as the single source of truth for shared regex patterns. The C26-01 fix completed MPIX_RE centralization. However, PITCH_UM_RE is still incomplete as a single source of truth — it does not match "um" (lowercase ASCII), which is handled by `sources/gsmarena.py`'s local `PITCH_RE`.

Current centralization status:
- `TYPE_FRACTIONAL_RE` — fully centralized (all local variants replaced) ✓
- `SIZE_MM_RE` — fully centralized ✓
- `PITCH_UM_RE` — NOT fully centralized (GSMArena still uses local PITCH_RE with "um") ✗
- `MPIX_RE` — fully centralized (C26-01 fix) ✓

For true DRY completeness, PITCH_UM_RE should include all variants that local patterns handle, and GSMArena should import the shared pattern instead of maintaining its own.

**Fix:** Add "um" to the shared PITCH_UM_RE alternation. Optionally, update GSMArena to import the shared pattern.

---

## Summary

- ARCH27-01 (LOW): PITCH_UM_RE incomplete as single source of truth — missing "um"
