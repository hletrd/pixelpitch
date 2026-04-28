# Tracer Review (Cycle 31) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## Previous Findings Status

TR30-01 (GSMArena per-phone try/except) fixed in C30. All previous fixes stable.

## New Findings

### TR31-01: merge_camera_data spec/derived pitch inconsistency — causal trace

**File:** `pixelpitch.py`, lines 413-432
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. openMVG provides camera "X" with `Spec(pitch=None, size=(5.0,3.7), mpix=10.0)`
2. `derive_spec()` at line 693-698: `spec.pitch` is None, `area=18.5`, `mpix=10.0`, so `derived.pitch = pixel_pitch(18.5, 10.0)` ~= 1.36
3. `derived.pitch` is 1.36 (computed), `spec.pitch` is None (no direct measurement)
4. Existing CSV has same camera with `spec.pitch=2.0` (Geizhals measurement), `derived.pitch=2.0`
5. `merge_camera_data()` processes this camera:
   - Line 417-418: `new_spec.spec.pitch is None` (True) AND `existing_spec.spec.pitch is not None` (True) → `new_spec.spec.pitch = 2.0` — spec.pitch CORRECTED to 2.0
   - Line 431-432: `new_spec.pitch is None` (False, it's 1.36) → condition NOT met → derived.pitch stays at 1.36 — WRONG
6. Template at pixelpitch.html line 84: displays `derived.pitch` (1.36), NOT `spec.pitch` (2.0)
7. `write_csv()` at line 796: writes `derived.pitch` (1.36) to CSV
8. Next build: `parse_existing_csv()` reads 1.36 from CSV, computes derived.pitch=1.36
9. The authoritative 2.0 measurement is permanently lost

**Competing hypothesis:** Is this scenario realistic? Yes. Geizhals directly reports pixel pitch for many cameras. openMVG does not report pitch (it's None in the Spec). When the two sources merge, the computed pitch from area+mpix can differ from the direct measurement.

**Fix:** After all Spec field preservation, if `spec.pitch` was restored from existing but `derived.pitch` does not match, set `derived.pitch = spec.pitch`. This ensures the authoritative measurement takes precedence over the computed approximation.

---

## Summary

- TR31-01 (MEDIUM): merge_camera_data spec/derived pitch inconsistency — computed value overwrites direct measurement
