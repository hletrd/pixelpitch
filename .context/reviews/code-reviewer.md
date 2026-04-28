# Code Review (Cycle 41) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-40 fixes, focusing on NEW issues

## Previous Findings Status

C40 findings (derive_spec computed 0.0 sentinel, write_csv isfinite guards) confirmed implemented and working. Gate tests pass. No regressions.

## New Findings

### CR41-01: `derive_spec` preserves invalid direct `spec.pitch` values (0.0, negative, NaN) — incomplete validation

**File:** `pixelpitch.py`, lines 759-760
**Severity:** MEDIUM | **Confidence:** HIGH

The C40 fix added a 0.0-to-None conversion for the *computed* pitch path (when `spec.pitch is None` and pitch is derived from `pixel_pitch()`). However, the *direct* path — when `spec.pitch` is explicitly set to 0.0, negative, or NaN — is completely unguarded:

```python
if spec.pitch is not None:
    pitch = spec.pitch   # <-- no validation
```

This means:
- `Spec(pitch=0.0)` → `derived.pitch = 0.0` — passes through selectattr, wrong table section
- `Spec(pitch=-1.0)` → `derived.pitch = -1.0` — negative pitch in data model
- `Spec(pitch=nan)` → `derived.pitch = nan` — NaN in data model

The template `> 0` guard renders these as "unknown" in the cell, but the camera is still in the wrong section (selectattr includes 0.0 and -1.0 but not NaN).

**Concrete scenario:**
```
Spec(name="Cam", size=(5.0, 3.7), pitch=0.0, mpix=33.0)
→ derive_spec: pitch = spec.pitch = 0.0  (no validation)
→ selectattr('pitch', 'ne', None) includes it
→ Camera in "with pitch" table showing "unknown" — wrong section
```

**Fix:** In `derive_spec`, validate `spec.pitch` the same way as computed pitch:

```python
if spec.pitch is not None:
    pitch = spec.pitch
    if not isfinite(pitch) or pitch <= 0:
        pitch = None
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)
    if pitch == 0.0:
        pitch = None
else:
    pitch = None
```

---

### CR41-02: `write_csv` writes 0.0 and negative mpix/pitch values — `isfinite` guard insufficient

**File:** `pixelpitch.py`, lines 866-868
**Severity:** LOW | **Confidence:** HIGH

The C40 fix added `isfinite()` checks for mpix, pitch, and area in `write_csv`. However, `isfinite(0.0)` returns True and `isfinite(-1.0)` returns True, so these physically invalid values pass through to the CSV:

- `mpix=0.0` → written as "0.0" → `parse_existing_csv` rejects it → data loss on round-trip
- `mpix=-5.0` → written as "-5.0" → `parse_existing_csv` rejects it → data loss on round-trip
- `pitch=0.0` → written as "0.00" → `parse_existing_csv` rejects it → data loss on round-trip
- `pitch=-1.0` → written as "-1.00" → `parse_existing_csv` rejects it → data loss on round-trip

**Fix:** Replace `isfinite()` with positivity checks in `write_csv`:

```python
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and spec.mpix > 0 else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and derived.pitch > 0 else ""
area_str = f"{derived.area:.2f}" if derived.area is not None and derived.area > 0 else ""
```

This is consistent with `parse_existing_csv`'s positivity checks and ensures the CSV round-trip is lossless.

---

### CR41-03: `merge_camera_data` preserves `spec.pitch=0.0` from existing data — re-introduces sentinel

**File:** `pixelpitch.py`, lines 449, 471-473
**Severity:** LOW | **Confidence:** MEDIUM

When merging, `merge_camera_data` preserves `spec.pitch` from existing data if new data has None. If the existing data has `spec.pitch=0.0` (e.g., from an older CSV that predates the positivity check), this 0.0 is preserved and then copied to `derived.pitch` via the consistency check at lines 471-473.

In practice, this is a LOW severity issue because:
1. Source parsers cannot produce `spec.pitch=0.0` (regexes only match positive floats)
2. `parse_existing_csv` now rejects 0.0 pitch from CSV input
3. The only way 0.0 enters is through legacy data or direct API usage

**Fix:** After merging field values, validate pitch just like `derive_spec` should (CR41-01):

```python
if new_spec.spec.pitch is not None and (not isfinite(new_spec.spec.pitch) or new_spec.spec.pitch <= 0):
    new_spec.spec.pitch = existing_spec.spec.pitch if existing_spec.spec.pitch is not None and existing_spec.spec.pitch > 0 else None
```

Or more simply: add a `_validate_pitch` helper that both `derive_spec` and `merge_camera_data` use.

---

## Summary

- CR41-01 (MEDIUM): `derive_spec` preserves invalid direct `spec.pitch` values (0.0, negative, NaN) — computed path fixed but direct path unguarded
- CR41-02 (LOW): `write_csv` writes 0.0/negative mpix/pitch — `isfinite` guard insufficient, needs positivity check
- CR41-03 (LOW): `merge_camera_data` preserves `spec.pitch=0.0` from existing data — re-introduces sentinel
