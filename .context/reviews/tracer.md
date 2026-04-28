# Tracer Review (Cycle 40) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## Previous Findings Status

TR39-01 fixed. Template renders "unknown" for negative/NaN pitch/mpix.

## New Findings

### TR40-01: pitch=0.0 sentinel data flow — `pixel_pitch` -> `derive_spec` -> `selectattr` -> wrong table section

**Files:** `pixelpitch.py` (pixel_pitch, derive_spec), `templates/pixelpitch.html` (selectattr)
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace for pitch=0.0 misclassification:**

1. `Spec(name="Cam", size=(5.0, 3.7), pitch=None, mpix=0.0)` — mpix=0.0 is invalid
2. `derive_spec`: `spec.pitch is None`, so compute from area+mpix
3. `pixel_pitch(18.5, 0.0)` → returns 0.0 (sentinel for invalid inputs)
4. `derive_spec` sets `pitch = 0.0` without checking if 0.0 is a valid result
5. Template: `selectattr('pitch', 'ne', None)` — `0.0 != None` is True → camera in "with pitch" table
6. Template cell: `spec.pitch > 0` → `0.0 > 0` is False → shows "unknown"
7. Result: Camera with "unknown" pitch appears in the "with pitch" table

**Comparison with pitch=None:**
- `pitch=None` → `selectattr('pitch', 'ne', None)` → False → goes to "without pitch" section (correct)
- `pitch=0.0` → `selectattr('pitch', 'ne', None)` → True → goes to "with pitch" section (incorrect)

**Root cause:** `pixel_pitch()` uses 0.0 as a sentinel for "invalid", but `derive_spec` treats 0.0 as a valid computed result. This is a contract mismatch between the two functions.

**Fix:** In `derive_spec`, convert `pixel_pitch()`'s 0.0 return to None.

---

## Summary

- TR40-01 (MEDIUM): pitch=0.0 sentinel flows through derive_spec -> selectattr -> camera in wrong table section
