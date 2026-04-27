# Test Engineer Review (Cycle 14) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository test review after cycles 1-13 fixes

## Previously Fixed (Cycles 1-13) — Confirmed Resolved
TE13-01 (load_csv UnicodeDecodeError test) and TE13-02 (Sony fallback test) were added in cycle 13. All passing.

## New Findings

### TE14-01: No test for UTF-8 BOM handling in parse_existing_csv
**File:** `tests/test_parsers_offline.py`; `pixelpitch.py`, lines 250-330
**Severity:** MEDIUM (test gap for a MEDIUM bug) | **Confidence:** HIGH

When the BOM fix is applied (stripping BOM from CSV content before parsing), a test should verify that `parse_existing_csv` correctly handles CSV content with a UTF-8 BOM prefix. Without this test, the fix could regress.

**Fix:** Add a test case that feeds `parse_existing_csv` with BOM-prefixed content and asserts it produces the same result as non-BOM content.

---

### TE14-02: No test for openMVG DSLR category misclassification
**File:** `tests/test_parsers_offline.py`; `sources/openmvg.py`, lines 63-69
**Severity:** MEDIUM (test gap for a MEDIUM bug) | **Confidence:** HIGH

The existing openMVG test (`test_openmvg_csv_parser`) tests Canon EOS 5D and asserts `category="mirrorless"`. This test would need to be updated when the DSLR classification fix is applied. Additionally, a dedicated test for the DSLR heuristic should be added.

**Fix:** When the DSLR heuristic is added to openMVG, update the test to assert `category="dslr"` for Canon EOS 5D and add test cases for other DSLR name patterns.

---

### TE14-03: No test for CineD FORMAT_TO_MM regex/table consistency
**File:** `tests/test_parsers_offline.py`; `sources/cined.py`
**Severity:** LOW (test gap for a LOW bug) | **Confidence:** MEDIUM

The existing `test_cined_format_coverage` only tests entries that are capturable by the regex. It doesn't check for unreachable entries in `FORMAT_TO_MM`. When the fix is applied (extending the regex or removing dead entries), a test should verify that every `FORMAT_TO_MM` key is reachable by the regex.

**Fix:** Extend `test_cined_format_coverage` to verify that all `FORMAT_TO_MM` keys are capturable by the regex alternation.

---

## Summary
- NEW findings: 3 (2 MEDIUM test gaps, 1 LOW test gap)
- TE14-01: No BOM handling test — MEDIUM
- TE14-02: No openMVG DSLR classification test — MEDIUM
- TE14-03: No CineD FORMAT_TO_MM reachability test — LOW
