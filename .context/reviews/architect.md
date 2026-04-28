# Architect Review — Cycle 46

**Date:** 2026-04-28
**Reviewer:** architect

## Previous Findings Status

ARCH45-01 (GSMArena regex word boundary) — COMPLETED. Fix applied.

## New Findings

### ARCH46-01: matched_sensors field has no sentinel for "not checked" vs "checked, empty"

**File:** `pixelpitch.py` (derive_spec, merge_camera_data), `models.py` (SpecDerived)
**Severity:** MEDIUM | **Confidence:** HIGH

The `SpecDerived.matched_sensors` field uses `Optional[List[str]]` with default `None`, but `derive_spec` always initializes it as `[]` (empty list) regardless of whether the sensors database was consulted. This creates a semantic ambiguity: `matched_sensors=[]` can mean either "we checked and found no matches" or "we didn't check at all."

This ambiguity causes data loss in `merge_camera_data`, which preserves fields from existing data only when the new value is `None`. Since `derive_spec` always returns `[]` (never `None`), the preservation check `if new_spec.matched_sensors is None and existing_spec.matched_sensors is not None` never triggers, and existing sensor match data is silently overwritten.

The architectural fix is to use the `None` sentinel correctly:
- `matched_sensors=None` means "not checked" (sensors_db was not available)
- `matched_sensors=[]` means "checked, found nothing" (sensors_db was consulted but no matches)
- `matched_sensors=['IMX455']` means "checked, found matches"

This is consistent with how other `Optional` fields work in the codebase (e.g., `spec.size=None` means "unknown" while `spec.size=(36.0, 24.0)` means "known").

**Fix:** In `derive_spec`, set `matched_sensors=None` when `sensors_db` is falsy, and `matched_sensors=[]` only when `sensors_db` was consulted but found no matches. Then add preservation logic in `merge_camera_data`.

---

## Summary

- ARCH46-01 (MEDIUM): matched_sensors field has no sentinel for "not checked" vs "checked, empty"
