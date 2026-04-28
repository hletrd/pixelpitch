# Designer Review (Cycle 40) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28

## Previous Findings Status

DES39-01 fixed. Template uses `> 0` guard for pitch/mpix.

## New Findings

### DES40-01: pitch=0.0 cameras appear in "with pitch" table showing "unknown" — wrong section

**File:** `templates/pixelpitch.html`, line 183
**Severity:** MEDIUM | **Confidence:** HIGH

When `derive_spec` computes `pitch=0.0` (from `pixel_pitch`'s sentinel), the template's `selectattr('pitch', 'ne', None)` includes it in the "with pitch" table. The pitch cell shows "unknown" (due to the `> 0` guard), but the camera is in the wrong section. Users scanning the "with pitch" table expect all cameras to have a valid pixel pitch value.

This is confusing UX — a camera with "unknown" pitch appears in a table labeled as having pitch data, while the "Cameras with Unknown Pixel Pitch" section at the bottom would be the correct place.

**Fix:** Fix `derive_spec` to produce None instead of 0.0 for computed pitch (upstream fix). This ensures `selectattr/rejectattr` correctly routes cameras to the right section.

---

## Summary

- DES40-01 (MEDIUM): pitch=0.0 cameras appear in wrong table section — "unknown" in "with pitch" table
