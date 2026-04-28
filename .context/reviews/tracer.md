# Tracer Review (Cycle 41) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## Previous Findings Status

TR40-01 fixed. `derive_spec` converts computed 0.0 sentinel to None.

## New Findings

### TR41-01: Invalid direct `spec.pitch` values bypass derive_spec validation — data flow trace

**Files:** `pixelpitch.py` (derive_spec, merge_camera_data, write_csv), `templates/pixelpitch.html`
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace for spec.pitch=0.0 direct path:**

1. `Spec(name="Cam", size=(5.0, 3.7), pitch=0.0, mpix=33.0)` — direct pitch=0.0
2. `derive_spec`: `spec.pitch is not None` → `pitch = spec.pitch = 0.0` (NO validation)
3. `derived.pitch = 0.0`
4. Template: `selectattr('pitch', 'ne', None)` — `0.0 != None` is True → camera in "with pitch" table
5. Template cell: `spec.pitch > 0` → `0.0 > 0` is False → shows "unknown"
6. Result: Camera with "unknown" pitch in wrong table section

**Causal trace for spec.pitch=-1.0:**

1. `Spec(name="Cam", size=(5.0, 3.7), pitch=-1.0, mpix=33.0)`
2. `derive_spec`: `spec.pitch is not None` → `pitch = spec.pitch = -1.0` (NO validation)
3. Template: `selectattr('pitch', 'ne', None)` — `-1.0 != None` is True → camera in "with pitch" table
4. Template cell: `spec.pitch > 0` → `-1.0 > 0` is False → shows "unknown"
5. write_csv: `isfinite(-1.0)` is True → writes "-1.00" to CSV
6. parse_existing_csv: `-1.0 <= 0` → rejects → data loss on round-trip

**Root cause:** `derive_spec` validates the computed pitch path (0.0 sentinel to None) but not the direct pitch path. Any invalid value in `spec.pitch` bypasses the sentinel fix.

**Fix:** In `derive_spec`, validate `spec.pitch` the same as computed pitch: reject non-positive and non-finite values.

---

## Summary

- TR41-01 (MEDIUM): Invalid direct spec.pitch values bypass derive_spec validation — 0.0, negative, NaN flow through to selectattr, write_csv, and CSV round-trip
