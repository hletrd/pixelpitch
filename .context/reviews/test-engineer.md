# Test Engineer Review (Cycle 33) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

TE30-01 (no test for GSMArena fetch() per-phone resilience) remains deferred — requires network access. TE32-01 (test for write_csv 0.0 value preservation) was implemented in C32.

## New Findings

### TE33-01: No test for derive_spec with spec.pitch=0.0 taking precedence

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

The C32-01 fix added a test for write_csv 0.0 value preservation. But there is no test verifying that `derive_spec` with `spec.pitch=0.0` correctly preserves the 0.0 value instead of computing pitch from area+mpix. The `test_pixel_pitch` function tests `pixel_pitch(864.0, 0.0) == 0.0` but that's the leaf function, not derive_spec.

**Current behavior (bug):** derive_spec with spec.pitch=0.0 computes a non-zero pitch from area+mpix.
**Expected behavior:** derive_spec with spec.pitch=0.0 should produce derived.pitch=0.0.

**Fix:** Add test case in test_pixel_pitch or a new section:
```python
spec_zero_pitch = Spec(name="Zero Pitch Cam", category="fixed", type=None,
                       size=(35.9, 23.9), pitch=0.0, mpix=33.0, year=2021)
d_zp = pp.derive_spec(spec_zero_pitch)
expect("derive_spec spec.pitch=0.0 preserved", d_zp.pitch, 0.0, tol=0.01)
```

This test will currently FAIL, demonstrating the CR33-01 bug.

---

### TE33-02: No test for sorted_by with 0.0 values

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The `sorted_by` function uses truthy checks for sort keys. There is no test verifying that cameras with pitch=0.0, area=0.0, or mpix=0.0 are sorted correctly (i.e., at position 0.0, not at -1).

**Fix:** Add a test case with a mix of None and 0.0 values and verify sort order.

---

### TE33-03: No test for template rendering of 0.0 pitch/mpix values

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The template renders pitch/mpix using Jinja2 truthy checks (`{% if spec.pitch %}`). There is no test verifying that 0.0 values render as "0.0 µm" / "0.0 MP" instead of "unknown". The `test_about_html_rendering` tests template output but only for the about page. No template rendering test exists for the pixelpitch.html template with 0.0 data values.

**Fix:** Add a template rendering test that constructs a SpecDerived with pitch=0.0, mpix=0.0, renders the template, and checks that the HTML contains "0.0" and not "unknown" for those fields.

---

## Summary

- TE33-01 (LOW-MEDIUM): No test for derive_spec with spec.pitch=0.0 taking precedence
- TE33-02 (LOW): No test for sorted_by with 0.0 values
- TE33-03 (LOW): No test for template rendering of 0.0 pitch/mpix values
