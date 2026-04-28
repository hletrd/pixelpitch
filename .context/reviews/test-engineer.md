# Test Engineer Review (Cycle 35) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

TE34-01 (match_sensors megapixels=0.0 test) implemented in C34. TE34-02 (match_sensors width/height=0.0 test) implemented in C34.

## New Findings

### TE35-01: No test for `pixel_pitch` with negative area (ValueError crash)

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

There is no test verifying that `pixel_pitch` or `derive_spec` handles negative area without crashing. The `test_pixel_pitch` function tests zero and negative mpix, but not negative area. Since `pixel_pitch(-864.0, 33.0)` raises `ValueError: expected a nonnegative input`, this crash path is untested and would not be caught by CI.

**Fix:** Add test:
```python
# Edge case: negative area — must not crash
pitch_neg = pp.pixel_pitch(-864.0, 33.0)
expect("negative area pitch", pitch_neg, 0.0)
```

---

### TE35-02: No test for `derive_spec` with negative sensor dimensions (ValueError crash)

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

There is no test verifying `derive_spec` handles negative sensor size without crashing. `derive_spec(spec)` with `spec.size=(-5.0, 3.7), pitch=None, mpix=10.0` crashes with `ValueError` from `pixel_pitch`. This crash path is untested.

**Fix:** Add test:
```python
spec_neg = Spec(name='Neg Size', category='fixed', type=None,
                size=(-5.0, 3.7), pitch=None, mpix=10.0, year=2020)
d_neg = pp.derive_spec(spec_neg)
expect("derive_spec negative size: no crash", d_neg is not None, True)
```

---

### TE35-03: No test for empty strings in matched_sensors from semicolon splitting

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

There is no test for `parse_existing_csv` with leading/trailing semicolons in the matched_sensors field. The split produces empty strings that propagate through the system.

**Fix:** Add test in `test_parse_existing_csv`:
```python
csv_semicolons = '''id,name,category,type,...,matched_sensors
0,Test,mirrorless,...,;IMX455;
'''
parsed = pp.parse_existing_csv(csv_semicolons)
expect("no empty strings in matched_sensors", '' not in parsed[0].matched_sensors, True)
```

---

### TE35-04: No test for openmvg with negative pixel dimensions

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

There is no test verifying that `openmvg.fetch` rejects negative pixel dimensions. Currently, negative pixel dimensions produce a positive mpix (product of two negatives).

**Fix:** Add test in `test_openmvg_negative_dimensions`:
```python
csv_neg_pixels = '''...Canon,EOS Neg,...,-100,-200'''
with unittest.mock.patch.object(openmvg, 'http_get', return_value=csv_neg_pixels):
    specs = openmvg.fetch()
if specs:
    expect("negative pixel dims: mpix is None", specs[0].mpix, None)
```

---

## Summary

- TE35-01 (MEDIUM): No test for `pixel_pitch` with negative area (ValueError crash)
- TE35-02 (MEDIUM): No test for `derive_spec` with negative sensor dimensions (ValueError crash)
- TE35-03 (LOW): No test for empty strings in matched_sensors
- TE35-04 (LOW): No test for openmvg with negative pixel dimensions
