# Architect Review (Cycle 26) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes

## Previous Findings Status

ARCH25-01 (DRY violation for SIZE_RE/PITCH_RE) addressed in C25-01. All previously identified architectural concerns remain deferred (LOW severity).

## New Findings

### ARCH26-01: MPIX_RE not centralized — incomplete DRY resolution from C25-01

**File:** `pixelpitch.py` line 42 vs `sources/__init__.py` line 67
**Severity:** MEDIUM | **Confidence:** HIGH

The C25-01 fix centralized SIZE_RE and PITCH_RE (imported from `sources/__init__.py`), following the pattern already used for TYPE_FRACTIONAL_RE. However, MPIX_RE was not similarly centralized, despite the C25-01 aggregate review explicitly flagging it.

Current state after C25-01 fix:
- `TYPE_FRACTIONAL_RE` — centralized (imported from sources) ✓
- `SIZE_MM_RE` — centralized (imported from sources) ✓
- `PITCH_UM_RE` — centralized (imported from sources) ✓
- `MPIX_RE` — NOT centralized (local definition in pixelpitch.py) ✗

The `sources/__init__.py` exports `MPIX_RE` in `__all__` (line 89) and it is a superset of the local pattern. This is the same class of DRY violation that was fixed for SIZE_RE and PITCH_RE.

**Fix:** Import `MPIX_RE` from `sources` in `pixelpitch.py`, replacing the local definition on line 42. Update `extract_specs()` to use the imported pattern. This completes the DRY centralization that C25-01 started.

---

## Summary

- ARCH26-01 (MEDIUM): MPIX_RE not centralized — incomplete DRY resolution from C25-01
