# Designer Review (Cycle 41) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28

## Previous Findings Status

DES40-01 partially fixed. Computed 0.0 pitch now produces None, but direct 0.0 pitch still misclassified.

## New Findings

### DES41-01: Direct spec.pitch=0.0 cameras still appear in wrong table section

**File:** `templates/pixelpitch.html`, line 183; `pixelpitch.py`, line 759-760
**Severity:** MEDIUM | **Confidence:** HIGH

The C40 fix ensured that *computed* pitch=0.0 (from `pixel_pitch()`) is converted to None, which correctly routes cameras to the "without pitch" section via `selectattr/rejectattr`. However, cameras with *directly set* `spec.pitch=0.0` (e.g., from legacy CSV data) still get `derived.pitch=0.0`, which passes through `selectattr('pitch', 'ne', None)` and appears in the "with pitch" table showing "unknown".

From a UX perspective, this is the same user-facing bug as DES40-01: a camera with "unknown" pitch appears in a table labeled as having pitch data. Users scanning the "with pitch" table expect all cameras to have valid pitch values.

**Fix:** Fix `derive_spec` to reject invalid direct pitch values (same as CR41-01). This ensures consistent behavior regardless of how pitch enters the system.

---

## Summary

- DES41-01 (MEDIUM): Direct spec.pitch=0.0 cameras still appear in wrong table section — same UX bug as DES40-01 but different code path
