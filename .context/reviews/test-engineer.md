# Test Engineer Review (Cycle 22) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28

## TE22-01: No test for year-change log in merge_camera_data

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The `test_merge_field_preservation` test verifies that year values are correctly preserved or overridden, but it does NOT verify the year-change diagnostic log. The existing test `test_create_camera_key_year_mismatch` tests the year-change scenario but only checks the final data values, not the log output.

After the C22-01 bug (year-change `elif` attached to wrong `if`), there is no test that would catch this regression because:
1. The data values are correct (year is properly set)
2. The log is only a diagnostic — not testable via the current test framework

**Fix:** Add a test that captures stdout during merge and verifies the year-change log is printed when years differ and pitch is preserved:

```python
def test_merge_year_change_log():
    import io, contextlib
    # Case: new pitch is None, existing pitch is set, years differ
    existing = [derive("Cam Y", "fixed", (5.0, 3.7), 10.0, 2020, pitch_val=2.0)]
    new = [derive("Cam Y", "fixed", (5.0, 3.7), 10.0, 2021, pitch_val=None)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        merged = pp.merge_camera_data(new, existing)
    expect("year change log printed", "Year changed" in buf.getvalue(), True)
```

---

## TE22-02: No test for Sony DSC hyphen normalisation

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The test for Sony DSC-HX400 expects "Sony DSC HX400" (space between DSC and HX400), which tests the URL-derived name path. But there is no test for the Model Name path where "Sony DSC-HX400" would retain the hyphen. If a DSC-hyphen normalizer is added, it needs a test.

---

## Summary

- TE22-01 (LOW): No test for year-change log in merge
- TE22-02 (LOW): No test for Sony DSC hyphen normalisation
