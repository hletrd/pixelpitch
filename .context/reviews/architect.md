# Architect Review (Cycle 25) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes

## Previous Findings Status

All previously identified architectural concerns remain deferred (LOW severity). ARCH24-01 (TYPE_FRACTIONAL_RE evolution) addressed by adding space+inch and bare 1-inch support.

## New Findings

### ARCH25-01: Duplicate regex patterns with different robustness — DRY violation

**File:** `pixelpitch.py` lines 42-43 vs `sources/__init__.py` lines 65-66
**Severity:** MEDIUM | **Confidence:** HIGH

The project has two sets of regex patterns for the same conceptual data (sensor dimensions, pixel pitch):

1. `pixelpitch.py`: `SIZE_RE`, `PITCH_RE` — used by Geizhals parsing (more restrictive)
2. `sources/__init__.py`: `SIZE_MM_RE`, `PITCH_UM_RE` — used by other source modules (more robust)

This violates DRY. As the sources have evolved, the shared patterns in `sources/__init__.py` have been improved to handle more edge cases (Unicode ×, Greek mu, spaces, multiple suffix variants), while the Geizhals-specific patterns in `pixelpitch.py` have not been updated to match.

The `TYPE_FRACTIONAL_RE` pattern was already centralized (imported from `sources/__init__.py` into `pixelpitch.py`), but `SIZE_RE` and `PITCH_RE` were not similarly centralized.

**Fix:** Import `SIZE_MM_RE` and `PITCH_UM_RE` from `sources` in `pixelpitch.py` (replacing the local `SIZE_RE` and `PITCH_RE`), or centralize all patterns in `sources/__init__.py` and import them. This follows the same pattern already used for `TYPE_FRACTIONAL_RE`.

---

## Summary

- ARCH25-01 (MEDIUM): Duplicate regex patterns with different robustness — DRY violation
