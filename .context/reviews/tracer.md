# Tracer Review (Cycle 34) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-33 fixes, focusing on NEW issues

## Previous Findings Status

TR33-01 (derive_spec 0.0 pitch override) fixed in C33.

## New Findings

### TR34-01: match_sensors ZeroDivisionError — causal trace

**File:** `pixelpitch.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Source parser (e.g., openMVG) produces `Spec(mpix=0.0)` from `round(pw * ph / 1_000_000, 1)` when pixels are 0
2. `derive_spec` correctly handles mpix=0.0 (pixel_pitch returns 0.0)
3. `merge_camera_data` calls `match_sensors(size[0], size[1], spec.mpix, sensors_db)` — line 471
4. `match_sensors`: `if megapixels is not None and sensor_megapixels:` → True (0.0 is not None)
5. `abs(0.0 - 61.2) / 0.0 * 100` → ZeroDivisionError
6. Exception is NOT caught — crashes the entire render pipeline

**Competing hypothesis:** Could mpix=0.0 ever reach match_sensors? The `pixel_pitch` function returns 0.0 for mpix=0.0, and `derive_spec` would set `pitch=0.0`. The Spec would have `mpix=0.0`. During merge, `match_sensors` is called on existing-only cameras (line 470). If a camera has mpix=0.0 in the existing CSV, this path would be reached.

**Fix:** Add `megapixels > 0` guard to the division branch.

---

## Summary

- TR34-01 (MEDIUM): match_sensors ZeroDivisionError with megapixels=0.0 — crash path confirmed
