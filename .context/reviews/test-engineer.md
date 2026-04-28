# Test Engineer Review (Cycle 23) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28

## TE23-01: No test for `_body_category` in `sources/imaging_resource.py` edge cases

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** LOW

The `_body_category` function has several fallback heuristics (name-based GoPro/Insta360/action cam detection, sensor format fallbacks). While the existing test covers the "Full-Frame" hyphenated case, the following paths are untested:

- Name-based action cam detection ("gopro", "insta360", "osmo action" in name)
- Name-based camcorder detection ("handycam" in name)
- Sensor format fallback to "mirrorless" for "APS-C", "Micro Four Thirds", "medium format"
- The fallback to "fixed" when no other match is found

However, these are low-risk heuristics that are unlikely to regress since they are simple string checks. Testing would require constructing mock data for each branch.

**Confidence lowered** because these are simple string comparisons with low regression risk.

---

## Summary

- TE23-01 (LOW): No test for `_body_category` name-based and sensor-format fallback branches — low regression risk
