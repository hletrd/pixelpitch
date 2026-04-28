# Code Review (Cycle 40) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-39 fixes, focusing on NEW issues

## Previous Findings Status

All C39 findings confirmed fixed. Template uses `> 0` guard. `_safe_float` docstring updated. `parse_existing_csv` has positivity checks. Negative/NaN/inf pitch/mpix tests pass. Gate tests pass.

## New Findings

### CR40-01: `derive_spec` propagates `pixel_pitch()` 0.0 sentinel as valid pitch — selectattr misclassification

**File:** `pixelpitch.py`, lines 757-762
**Severity:** MEDIUM | **Confidence:** HIGH

When `spec.pitch` is None but both `area` and `spec.mpix` are known, `derive_spec` calls `pixel_pitch(area, spec.mpix)`. The `pixel_pitch` function returns 0.0 as a sentinel for invalid inputs (negative, zero, NaN, inf). `derive_spec` then sets `derived.pitch = 0.0` without checking if the computed value is valid.

This 0.0 sentinel passes through the template's `selectattr('pitch', 'ne', None)` filter (0.0 != None is True), so the camera appears in the "with pitch" table. The template's `> 0` guard correctly shows "unknown" in the pitch cell, but the camera is in the wrong section — it should be in the "without pitch" section.

**Concrete scenario:**
1. `Spec(name="Cam", size=(5.0, 3.7), pitch=None, mpix=0.0)`
2. `derive_spec` computes: `area=18.5`, `pixel_pitch(18.5, 0.0) = 0.0`
3. `derived.pitch = 0.0`
4. Template: `selectattr('pitch', 'ne', None)` includes it (0.0 != None)
5. Camera appears in "with pitch" table showing "unknown" — wrong section

**Fix:** In `derive_spec`, after computing pitch from `pixel_pitch()`, convert 0.0 to None:
```python
if spec.pitch is not None:
    pitch = spec.pitch
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)
    if pitch == 0.0:  # 0.0 is a sentinel for invalid inputs
        pitch = None
else:
    pitch = None
```

---

### CR40-02: `write_csv` writes inf/nan values to CSV without validation

**File:** `pixelpitch.py`, lines 839-872
**Severity:** LOW | **Confidence:** HIGH

`write_csv` uses Python's default float formatting, which produces "inf" and "nan" strings for non-finite values. While `parse_existing_csv` correctly rejects these on re-read (via `_safe_float`), the CSV file contains non-standard values. Other CSV consumers may not handle them correctly.

**Concrete scenario:**
```
0,Inf Cam,fixed,,35.90,23.90,858.01,inf,5.12,2021,
0,NaN Cam,fixed,,35.90,23.90,858.01,nan,5.12,2021,
```

**Fix:** In `write_csv`, check for non-finite values before writing:
```python
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and isfinite(spec.mpix) else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and isfinite(derived.pitch) else ""
area_str = f"{derived.area:.2f}" if derived.area is not None and isfinite(derived.area) else ""
```

---

## Summary

- CR40-01 (MEDIUM): `derive_spec` propagates 0.0 sentinel from `pixel_pitch` — causes misclassification in template's selectattr
- CR40-02 (LOW): `write_csv` writes inf/nan to CSV without validation
