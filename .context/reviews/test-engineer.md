# Test Engineer Review (Cycle 24) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-23 fixes

## Previous Findings Status

C23-01 (body-category fallback tests) implemented and verified. All test assertions pass.

## New Findings

### TE24-01: No test for TYPE_FRACTIONAL_RE "1/x.y inch" (space+inch) gap

**File:** `tests/test_parsers_offline.py`, `test_gsmarena_unicode_quotes()`
**Severity:** LOW | **Confidence:** MEDIUM

The existing `test_gsmarena_unicode_quotes` tests TYPE_FRACTIONAL_RE with ASCII quotes, Unicode quotes, `-inch` suffix, and bare number (no suffix). But it does NOT test the space+inch format (`1/2.3 inch`) which is a gap in the regex. Adding a test would document the expected behavior and catch regressions if the regex is updated.

**Fix:** Add test case:
```python
m5 = TYPE_FRACTIONAL_RE.search('1/2.3 inch')
expect("space+inch suffix match", m5.group(1) if m5 else None, "1/2.3")
```
(This test would FAIL until the regex is fixed, which is appropriate — it documents the gap.)

### TE24-02: No test for parse_sensor_field with bare 1-inch format

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

`parse_sensor_field('CMOS 1"')` returns `{type: None, size: None, pitch: None}` — the 1-inch sensor type is silently lost. There is no test for this case. If the 1-inch format extraction is added, a test should verify it.

**Fix:** Add test case:
```python
from pixelpitch import parse_sensor_field
result = parse_sensor_field('CMOS 1"')
expect("parse_sensor_field bare 1-inch type", result["type"], "1")
```
(This test would FAIL until parse_sensor_field is fixed.)

### TE24-03: No test for _parse_fields rstrip("</") data mangling

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** LOW

The `_parse_fields` function in `imaging_resource.py` uses `rstrip("</")` which strips individual chars rather than the string `"</"`. No test exists for this. A value ending in `"` (double-quote) or `/` would have those chars silently stripped. This was previously deferred as C3-08.

---

## Summary

- TE24-01 (LOW): No test for TYPE_FRACTIONAL_RE space+inch gap
- TE24-02 (LOW): No test for parse_sensor_field bare 1-inch format
- TE24-03 (LOW): No test for _parse_fields rstrip data mangling (previously deferred as C3-08)
