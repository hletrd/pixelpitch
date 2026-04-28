# Test Engineer Review (Cycle 21) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28

## TE21-01: test_merge_field_preservation doesn't verify SpecDerived fields

**Severity:** MEDIUM | **Confidence:** HIGH

The test added in C20-03 validates that `merged[0].spec.type`, `merged[0].spec.size`, and `merged[0].spec.pitch` are preserved (Spec-level fields). But it does NOT check `merged[0].size`, `merged[0].area`, or `merged[0].pitch` (SpecDerived-level fields). These are the fields the template actually reads. This test gap allowed the C21-01 bug (SpecDerived stale fields) to go undetected.

**Fix:** Add assertions for SpecDerived fields alongside existing Spec-level assertions:
```python
expect("merge: preserves derived.size from existing",
       merged_s[0].size, (5.0, 3.7), tol=0.01)
expect("merge: preserves derived.area from existing",
       merged_s[0].area, 18.5, tol=0.01)
expect("merge: preserves derived.pitch from existing",
       merged_p[0].pitch, 2.0, tol=0.01)
```

---

## TE21-02: No test for Sony RX/DSC/HX/WX/TX/QX name normalization

**Severity:** MEDIUM | **Confidence:** HIGH

The C20-02 fix added tests for FX3, FX30, FX6 but no tests for RX, DSC, HX, WX, TX, QX series cameras. This is why the broader naming issue went undetected.

**Fix:** Add test cases:
```python
name_rx100 = imaging_resource._parse_camera_name(
    {"Model Name": "Sony RX100 VII"},
    "https://www.imaging-resource.com/cameras/sony-rx100-vii-review/specifications/"
)
expect("IR Sony RX100 VII name", name_rx100, "Sony RX100 VII")

name_dsc = imaging_resource._parse_camera_name(
    {"Model Name": "Sony DSC-HX400"},
    "https://www.imaging-resource.com/cameras/sony-dsc-hx400-review/specifications/"
)
expect("IR Sony DSC-HX400 name", name_dsc, "Sony DSC-HX400")
```

---

## TE21-03: No test for mpix preservation in merge

**Severity:** LOW | **Confidence:** HIGH

The merge function does not preserve `mpix` when new data has `mpix=None`. There is no test for this scenario.

**Fix:** Add test case where new spec has `mpix=None` but existing has `mpix=33.0`.

---

## Summary

- TE21-01 (MEDIUM): test_merge_field_preservation missing SpecDerived assertions
- TE21-02 (MEDIUM): No test for Sony RX/DSC/HX/WX/TX/QX naming
- TE21-03 (LOW): No test for mpix preservation in merge
