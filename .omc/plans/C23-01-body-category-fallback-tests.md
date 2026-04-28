# Plan: Cycle 23 Findings — Body Category Fallback Tests

**Created:** 2026-04-28
**Status:** PENDING
**Source Reviews:** TE23-01 (test-engineer)

---

## Task 1: Add tests for `_body_category` name-based and sensor-format fallback branches — C23-01

**Finding:** C23-01 (test-engineer)
**Severity:** LOW | **Confidence:** LOW
**File:** `sources/imaging_resource.py`, `_body_category()` function
**Test file:** `tests/test_parsers_offline.py`

### Problem
The `_body_category` function has several fallback heuristics that are untested:
1. Name-based action cam detection ("gopro", "insta360", "osmo action" in name)
2. Name-based camcorder detection ("handycam" in name)
3. Sensor format fallbacks for "APS-C", "Micro Four Thirds", "medium format" → "mirrorless"
4. Final fallback to "fixed" when no other match

### Implementation
Add test cases in `test_imaging_resource()` exercising each untested branch:

```python
# Name-based action cam detection
expect("IR body category GoPro name",
       imaging_resource._body_category("", "", "GoPro Hero 12"), "actioncam")
expect("IR body category Insta360 name",
       imaging_resource._body_category("", "", "Insta360 X4"), "actioncam")

# Name-based camcorder detection
expect("IR body category Handycam name",
       imaging_resource._body_category("", "", "Sony Handycam FDR-AX43"), "camcorder")

# Sensor format fallbacks
expect("IR body category APS-C sensor",
       imaging_resource._body_category("", "APS-C", ""), "mirrorless")
expect("IR body category Micro Four Thirds sensor",
       imaging_resource._body_category("", "Micro Four Thirds", ""), "mirrorless")
expect("IR body category Medium Format sensor",
       imaging_resource._body_category("", "Medium Format", ""), "mirrorless")

# Final fallback to "fixed"
expect("IR body category unknown fallback",
       imaging_resource._body_category("", "1/2.3", ""), "fixed")
```

### Verification
- Gate tests (`python3 -m tests.test_parsers_offline`) must pass with new tests
- Each branch produces the expected category

---

## Deferred Findings

None. The single finding is scheduled for implementation above.
