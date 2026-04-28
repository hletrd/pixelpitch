# Test Engineer Review (Cycle 18) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository test coverage re-review after cycles 1-17 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

All previous test fixes confirmed working. Gate tests pass (98 checks).

## New Findings

### TE18-01: No test for GSMArena SENSOR_FORMAT_RE Unicode curly-quote matching
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The C17-03 fix updated `SENSOR_FORMAT_RE` to match Unicode curly quotes (`″`, U+2033), but no test was added to verify this. If the regex is later changed or broken, the Unicode quote handling would silently regress.

**Fix:** Add a test case:
```python
section("GSMArena SENSOR_FORMAT_RE Unicode quotes")
import re
# ASCII quote
m1 = gsmarena.SENSOR_FORMAT_RE.search('1/1.3"')
expect("ASCII quote match", m1.group(1) if m1 else None, "1/1.3")
# Unicode curly quote (U+2033)
m2 = gsmarena.SENSOR_FORMAT_RE.search('1/1.3″')
expect("Unicode quote match", m2.group(1) if m2 else None, "1/1.3")
```

---

### TE18-02: No test for Pentax KF, K-r, K-x DSLR classification
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The openMVG CSV test includes Pentax KP and Nikon Df but not Pentax KF, K-r, or K-x. After the C17-01 fix, all letter-suffix Pentax models should be tested to prevent regression. Currently only KP is tested.

**Fix:** Add 'Pentax,KF,...', 'Pentax,K-r,...', and 'Pentax,K-x,...' rows to the test CSV and add expect() calls verifying they are classified as "dslr".

---

### TE18-03: No test for SENSOR_TYPE_RE in pixelpitch.py matching sensor types
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The `SENSOR_TYPE_RE` regex in pixelpitch.py (used for Geizhals parsing) has no dedicated test. If this regex is modified or broken, Geizhals sensor type extraction would silently fail. A basic test should verify it matches standard `1/x.y"` patterns.

**Fix:** Add a test verifying `SENSOR_TYPE_RE` matches typical Geizhals sensor type strings.

---

## Summary
- NEW findings: 3 (all LOW)
- TE18-01: No test for GSMArena Unicode quotes — LOW
- TE18-02: No test for Pentax KF/K-r/K-x — LOW
- TE18-03: No test for SENSOR_TYPE_RE in pixelpitch.py — LOW
