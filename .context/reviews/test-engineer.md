# Test Engineer Review (Cycle 36) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

TE35-01 through TE35-04 all implemented and passing.

## New Findings

### TE36-01: No test for `pixel_pitch` with NaN inputs

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

There is no test verifying that `pixel_pitch` returns 0.0 for NaN inputs. Currently `pixel_pitch(float('nan'), 10.0)` returns `nan` (a bug), but there should be a test that catches this regression once fixed.

**Fix:** Add test:
```python
pitch_nan = pp.pixel_pitch(float('nan'), 33.0)
expect("nan area pitch", pitch_nan, 0.0)

pitch_nan2 = pp.pixel_pitch(864.0, float('nan'))
expect("nan mpix pitch", pitch_nan2, 0.0)
```

---

### TE36-02: No test for `pixel_pitch` with inf inputs

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

There is no test verifying that `pixel_pitch` returns 0.0 for inf inputs. Currently `pixel_pitch(float('inf'), 10.0)` returns `inf` (a bug), but there should be a test that catches this regression once fixed.

**Fix:** Add test:
```python
pitch_inf = pp.pixel_pitch(float('inf'), 33.0)
expect("inf area pitch", pitch_inf, 0.0)

pitch_inf2 = pp.pixel_pitch(864.0, float('inf'))
expect("inf mpix pitch", pitch_inf2, 0.0)
```

---

### TE36-03: No test for `parse_existing_csv` with NaN/inf values

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

There is no test verifying that `parse_existing_csv` rejects or normalizes NaN and inf values. Currently they pass through as `float('nan')` and `float('inf')`.

**Fix:** Add test:
```python
csv_nan = (
    "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
    "megapixels,pixel_pitch_um,year,matched_sensors\n"
    "0,Test,mirrorless,,nan,nan,nan,nan,nan,2021,\n"
)
parsed = pp.parse_existing_csv(csv_nan)
# After fix: NaN values should be treated as None
expect("NaN width rejected", parsed[0].size, None)
expect("NaN mpix rejected", parsed[0].spec.mpix, None)
expect("NaN pitch rejected", parsed[0].pitch, None)
```

---

### TE36-04: No test for `derive_spec` with NaN/inf sensor dimensions

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

There is no test verifying that `derive_spec` handles NaN or inf dimensions gracefully. Currently NaN produces NaN area and NaN pitch; inf produces inf area and inf pitch.

**Fix:** Add test:
```python
spec_nan = Spec(name='NaN Size', category='fixed', type=None,
                size=(float('nan'), 24.0), pitch=None, mpix=10.0, year=2020)
d_nan = pp.derive_spec(spec_nan)
expect("derive_spec NaN size: no crash", d_nan is not None, True)
expect("derive_spec NaN size: pitch is 0.0", d_nan.pitch, 0.0)
```

---

## Summary

- TE36-01 (MEDIUM): No test for `pixel_pitch` with NaN inputs
- TE36-02 (MEDIUM): No test for `pixel_pitch` with inf inputs
- TE36-03 (MEDIUM): No test for `parse_existing_csv` with NaN/inf values
- TE36-04 (LOW): No test for `derive_spec` with NaN/inf dimensions
