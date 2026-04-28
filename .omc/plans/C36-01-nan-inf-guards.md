# Plan: Cycle 36 Findings — NaN/inf Input Validation Guards

**Created:** 2026-04-28
**Status:** IN PROGRESS
**Source Reviews:** CR36-01, CR36-02, CR36-03, CRIT36-01, DBG36-01, DBG36-02, V36-02, V36-03, TR36-01, ARCH36-01, TE36-01, TE36-02, TE36-03, TE36-04, DES36-01, DOC36-01

---

## Task 1: Fix `pixel_pitch` to guard against NaN and inf inputs — C36-01 (core)

**Finding:** C36-01 (9-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py` line 184

### Problem

The guard `if mpix <= 0 or area <= 0: return 0.0` does not reject NaN or inf:
- `float('nan') <= 0` is `False` — NaN bypasses the guard
- `float('inf') <= 0` is `False` — inf bypasses the guard
- `pixel_pitch(float('nan'), 10.0)` returns `nan` (not 0.0)
- `pixel_pitch(float('inf'), 10.0)` returns `inf` (not 0.0)

### Implementation

1. Add `import math` (already imported as `from math import sqrt` — need to add `isfinite`)
2. In `pixelpitch.py`, `pixel_pitch()` function, line 184:
   - Change `if mpix <= 0 or area <= 0:` to `if not math.isfinite(area) or not math.isfinite(mpix) or mpix <= 0 or area <= 0:`
3. Update the docstring to mention NaN/inf handling

---

## Task 2: Fix `parse_existing_csv` to reject NaN and inf CSV values — C36-02

**Finding:** C36-02 (5-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py` lines 328-339

### Problem

Python's `float()` accepts `"nan"`, `"inf"`, `"-inf"`, `"NaN"` as valid inputs. The CSV parser uses bare `float()` for width, height, area, mpix, and pitch without checking for finite values.

### Implementation

1. Add a helper function `_safe_float(s: str) -> Optional[float]`:
   ```python
   def _safe_float(s: str) -> Optional[float]:
       """Parse a float string, returning None for NaN/inf/empty."""
       if not s:
           return None
       try:
           val = float(s)
           return val if math.isfinite(val) else None
       except (ValueError, TypeError):
           return None
   ```
2. Replace bare `float()` calls in `parse_existing_csv` with `_safe_float()`:
   - Line 331: `width = float(width_str)` → `width = _safe_float(width_str)`
   - Line 332: `height = float(height_str)` → `height = _safe_float(height_str)`
   - Line 337: `area = float(area_str) if area_str else None` → `area = _safe_float(area_str)`
   - Line 338: `mpix = float(mpix_str) if mpix_str else None` → `mpix = _safe_float(mpix_str)`
   - Line 339: `pitch = float(pitch_str) if pitch_str else None` → `pitch = _safe_float(pitch_str)`
3. Adjust the size construction to handle `_safe_float` returning None:
   ```python
   size = None
   if width is not None and height is not None:
       size = (width, height)
   ```

---

## Task 3: Fix `openmvg.fetch` to reject inf sensor dimensions — C36-03

**Finding:** C36-03
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/openmvg.py` line 96

### Problem

The size guard `sw > 0 and sh > 0` passes for `inf` because `inf > 0` is True.

### Implementation

1. In `sources/openmvg.py`, add `import math` at the top (already has `from __future__ import annotations`)
2. Line 96: Change `size = (sw, sh) if sw and sh and sw > 0 and sh > 0 else None` to:
   ```python
   size = (sw, sh) if sw and sh and math.isfinite(sw) and math.isfinite(sh) and sw > 0 and sh > 0 else None
   ```

---

## Task 4: Add NaN check to JS `isInvalidData` function — C36-04

**Finding:** DES36-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `templates/pixelpitch.html` line 264

### Problem

JS `parseFloat("nan") || 0` evaluates to `0`, which passes all validation checks. NaN-pitched rows are not hidden.

### Implementation

1. In `templates/pixelpitch.html`, `isInvalidData` function, after line 264:
   - Add a NaN check before the `> 10` check:
   ```javascript
   if (isNaN(pitch)) {
     return true;
   }
   ```

---

## Task 5: Add test coverage for NaN/inf handling — C36-01/C36-02 (tests)

**Finding:** TE36-01, TE36-02, TE36-03, TE36-04
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`

### Implementation

1. Add to `test_pixel_pitch()`:
   ```python
   # NaN inputs
   pitch_nan = pp.pixel_pitch(float('nan'), 33.0)
   expect("nan area pitch", pitch_nan, 0.0)
   pitch_nan2 = pp.pixel_pitch(864.0, float('nan'))
   expect("nan mpix pitch", pitch_nan2, 0.0)
   
   # inf inputs
   pitch_inf = pp.pixel_pitch(float('inf'), 33.0)
   expect("inf area pitch", pitch_inf, 0.0)
   pitch_inf2 = pp.pixel_pitch(864.0, float('inf'))
   expect("inf mpix pitch", pitch_inf2, 0.0)
   ```

2. Add `test_parse_existing_csv_nan_inf()`:
   ```python
   def test_parse_existing_csv_nan_inf():
       section("parse_existing_csv NaN/inf rejection")
       import pixelpitch as pp
       
       # NaN values in CSV should be treated as None
       csv_nan = (
           "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
           "megapixels,pixel_pitch_um,year,matched_sensors\n"
           "0,Test,mirrorless,,nan,nan,nan,nan,nan,2021,\n"
       )
       parsed = pp.parse_existing_csv(csv_nan)
       expect("NaN CSV: row count", len(parsed), 1)
       expect("NaN CSV: size is None", parsed[0].size, None)
       expect("NaN CSV: area is None", parsed[0].area, None)
       expect("NaN CSV: mpix is None", parsed[0].spec.mpix, None)
       expect("NaN CSV: pitch is None", parsed[0].pitch, None)
       
       # inf values in CSV should be treated as None
       csv_inf = (
           "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
           "megapixels,pixel_pitch_um,year,matched_sensors\n"
           "0,Test,mirrorless,,inf,inf,inf,inf,inf,2021,\n"
       )
       parsed2 = pp.parse_existing_csv(csv_inf)
       expect("inf CSV: row count", len(parsed2), 1)
       expect("inf CSV: size is None", parsed2[0].size, None)
       expect("inf CSV: area is None", parsed2[0].area, None)
       expect("inf CSV: mpix is None", parsed2[0].spec.mpix, None)
       expect("inf CSV: pitch is None", parsed2[0].pitch, None)
   ```

3. Add to `test_derive_spec_negative_size()`:
   ```python
   # NaN size: size=(nan, 24.0)
   spec_nan = Spec(name='NaN Size Cam', category='fixed', type=None,
                    size=(float('nan'), 24.0), pitch=None, mpix=10.0, year=2020)
   d_nan = pp.derive_spec(spec_nan)
   expect("derive_spec NaN size: no crash", d_nan is not None, True)
   expect("derive_spec NaN size: pitch is 0.0", d_nan.pitch, 0.0)
   ```

4. Add `test_parse_existing_csv_nan_inf` to the `main()` function call list.

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- New NaN/inf tests must pass
- Existing tests must not regress
- `pixel_pitch(float('nan'), 10.0)` must return 0.0
- `pixel_pitch(float('inf'), 10.0)` must return 0.0
- `parse_existing_csv` with "nan"/"inf" must produce None values

---

## Deferred Findings

None. All findings are scheduled for implementation.
