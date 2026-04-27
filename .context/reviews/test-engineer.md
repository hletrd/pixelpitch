# Test Engineer Review (Cycle 12) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository test review after cycles 1-11 fixes

## Previously Fixed (Cycles 1-11) — Confirmed Resolved
TE11-01 (year mismatch test) was added in cycle 11 as `test_create_camera_key_year_mismatch`. TE11-02 (render_html integration test) and TE11-03 (single wide lens test) remain as LOW-priority gaps.

## New Findings

### TE12-01: No test for name field whitespace stripping in parse_existing_csv
**File:** `tests/test_parsers_offline.py`; `pixelpitch.py`, lines 277, 291
**Severity:** MEDIUM (test gap) | **Confidence:** HIGH

The type field whitespace test was added in cycle 10, the category field test could be added alongside. But there is no test for name field whitespace in `parse_existing_csv`. Once the name field `.strip()` fix is applied (C12-01), a test should verify it works. This is a test gap that should be filled when the fix is implemented.

**Fix:** Add a test case in `test_parse_existing_csv` that includes a name with leading/trailing whitespace and asserts it is stripped.

---

### TE12-02: No test for `_parse_camera_name` with legacy spec URLs
**File:** `tests/test_parsers_offline.py`; `sources/imaging_resource.py`, line 151
**Severity:** MEDIUM (test gap) | **Confidence:** HIGH

The `_parse_camera_name` function is not directly tested — it's only tested indirectly through the fixture-based `test_imaging_resource` test. There's no test for the Sony branch with different URL formats (modern spec URL vs legacy spec URL). Since C12-02 identifies a bug in the Sony slug extraction for legacy URLs, a test should verify both URL formats produce correct names.

**Fix:** Add unit tests for `_parse_camera_name` with both modern and legacy URL formats.

---

## Summary
- NEW findings: 2 (2 MEDIUM test gaps)
- TE12-01: No test for name whitespace stripping — MEDIUM
- TE12-02: No test for _parse_camera_name URL formats — MEDIUM
