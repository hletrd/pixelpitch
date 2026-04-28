# Test Engineer Review (Cycle 20) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28

## TE20-01: No test for `pixel_pitch` edge cases (zero/negative mpix)
**Severity:** MEDIUM | **Confidence:** HIGH

The `pixel_pitch()` function has no test coverage for mpix=0 or mpix<0. The current test only checks valid positive values. This is why the ZeroDivisionError bug went undetected through 19 cycles.

**Fix:** Add test cases to `test_pixel_pitch()`:
```python
# Zero mpix - should return 0.0, not crash
result = pp.pixel_pitch(864.0, 0.0)
expect("zero mpix pitch", result, 0.0)

# Negative mpix - should return 0.0, not crash
result2 = pp.pixel_pitch(864.0, -1.0)
expect("negative mpix pitch", result2, 0.0)
```

---

## TE20-02: No test for Sony FX name normalization
**Severity:** MEDIUM | **Confidence:** HIGH

The `_parse_camera_name` function has tests for Sony ZV-E10 (ZV normalization) and Roman numerals, but no test for FX-series cameras. This is why the "Fx3" bug went undetected.

**Fix:** Add test cases:
```python
name_fx3 = imaging_resource._parse_camera_name(
    {"Model Name": "Sony FX3"},
    "https://www.imaging-resource.com/cameras/sony-fx3-review/specifications/"
)
expect("IR Sony FX3 name", name_fx3, "Sony FX3")
```

---

## TE20-03: No test for merge field preservation (type/size/pitch)
**Severity:** LOW | **Confidence:** HIGH

The `test_merge_camera_data` tests year preservation but doesn't test whether type, size, or pitch are preserved when new data has None values for these fields.

**Fix:** Add test case where new spec has `type=None` but existing has `type='1/2.3'`.

---

## Summary

- TE20-01 (MEDIUM): No test for pixel_pitch zero/negative mpix
- TE20-02 (MEDIUM): No test for Sony FX name normalization
- TE20-03 (LOW): No test for merge type/size/pitch preservation
