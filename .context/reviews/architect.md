# Architect Review (Cycle 36) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

ARCH35-01 (no input validation layer) partially addressed by negative-value guards in pixel_pitch and openmvg. ARCH35-02 (BOM literal) fixed.

## New Findings

### ARCH36-01: Validation layer still incomplete — NaN/inf not rejected at any stage

**Files:** `pixelpitch.py`, `sources/openmvg.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The C35-01 fix added negative-value guards in `pixel_pitch` and `openmvg.fetch`, but these use `> 0` and `<= 0` comparisons which do not catch NaN or inf. The validation layer remains incomplete because:

1. `float('nan') > 0` is `False` — NaN IS caught by `> 0` but NOT by `<= 0`
2. `float('inf') > 0` is `True` — inf passes through `> 0` checks
3. `float('nan') <= 0` is `False` — NaN bypasses `<= 0` guards

The architectural fix from C35-01 (add validation guards) should use `math.isfinite()` instead of simple comparison operators. This would provide a complete validation layer that rejects NaN, inf, negative, and zero in a single check.

**Fix:** Use `math.isfinite` in all validation guards.

---

## Summary

- ARCH36-01 (MEDIUM): Validation layer incomplete — NaN/inf not rejected at any stage
