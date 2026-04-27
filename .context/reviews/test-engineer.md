# Test Engineer Review (Cycle 11) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository test review after cycles 1-10 fixes

## Previously Fixed (Cycles 1-10) — Confirmed Resolved
All previous test-related fixes remain working. Gate tests pass cleanly.

## New Findings

### TE11-01: No test for `create_camera_key` year mismatch — duplicate cameras across sources
**File:** `pixelpitch.py`, lines 313-315; `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

`create_camera_key` includes the year in its key. When the same camera comes from two sources with different years (e.g., year=2021 vs year=None), two different keys are produced, and `merge_camera_data` treats them as separate cameras. There is no test covering this scenario.

The existing `test_merge_camera_data` tests overlapping cameras with the SAME year. It does not test cameras with differing years across sources.

**Fix:** Add a test that merges the same camera from two sources where one has a year and the other doesn't, asserting that only one entry appears in the merged result.

---

### TE11-02: No test for `render_html` output — integration gap
**File:** `pixelpitch.py`, lines 738-920; `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

There is no integration test that calls `render_html` (or a subset of it) and verifies the HTML output. The current tests test individual functions (CSV parsing, dedup, merge, etc.) but not the end-to-end rendering pipeline. A regression in template rendering (e.g., a broken template block, a missing variable) would not be caught by the gate tests.

The `test_about_html_rendering` test covers one template, but not the main `pixelpitch.html` template with table rendering.

**Fix:** Add a minimal integration test that renders `pixelpitch.html` with a small set of specs and checks that the HTML contains expected content (table rows, sensor sizes, etc.).

---

### TE11-03: `test_gsmarena_select_main_lens` test has no assertion for single wide lens case
**File:** `tests/test_parsers_offline.py`, lines 661-690
**Severity:** LOW | **Confidence:** LOW

The `_select_main_lens` edge case tests cover: two wide lenses, no wide lenses, empty camera, and no role tag. But there's no test for the simplest case: a single lens with "(wide)" role. While this is covered indirectly by the S25 Ultra fixture test, a dedicated unit test would be more explicit.

Low priority — the S25 Ultra fixture already validates this path.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 2 LOW)
- TE11-01: No test for create_camera_key year mismatch — MEDIUM
- TE11-02: No integration test for render_html — LOW
- TE11-03: No single wide lens unit test — LOW
