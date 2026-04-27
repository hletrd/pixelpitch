# Test Engineer Review (Cycle 15) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository test review after cycles 1-14 fixes

## Previously Fixed (Cycles 1-14) — Confirmed Resolved
TE14-01 (BOM handling test), TE14-02 (openMVG DSLR classification test), TE14-03 (CineD FORMAT_TO_MM reachability test) all added in cycle 14. All passing.

## New Findings

### TE15-01: No test for Samsung NX mirrorless misclassification in DSLR regex
**File:** `tests/test_parsers_offline.py`; `sources/openmvg.py`, line 44
**Severity:** MEDIUM (test gap for a MEDIUM bug) | **Confidence:** HIGH

The existing `test_openmvg_csv_parser` tests Canon EOS 5D (dslr) and Sony Alpha 7 III (mirrorless) but does not test Samsung NX cameras. Since the Samsung NX pattern incorrectly matches mirrorless cameras as DSLR, a test should verify that Samsung NX cameras are classified as "mirrorless" (not "dslr").

**Fix:** Add a Samsung NX mirrorless camera to the test CSV and assert `category="mirrorless"`.

---

### TE15-02: No test for Canon EOS xxxD DSLR classification
**File:** `tests/test_parsers_offline.py`; `sources/openmvg.py`, line 37
**Severity:** MEDIUM (test gap for a MEDIUM bug) | **Confidence:** HIGH

The existing test only tests Canon EOS 5D (single-digit xD). The xxxD Rebel series (250D, 800D, 850D, etc.) and xxD series (70D, 80D, 90D) are not tested. When the fix is applied (changing `\dD` to `\d+D`), a test should verify that these cameras are classified as "dslr".

**Fix:** Add a Canon EOS 250D (Rebel) to the test CSV and assert `category="dslr"`.

---

### TE15-03: No test for Geizhals rangefinder misclassification causing triple-duplicates
**File:** `tests/test_parsers_offline.py`; `pixelpitch.py`, lines 740-747, 339
**Severity:** MEDIUM (test gap for a MEDIUM bug) | **Confidence:** HIGH

There is no test that verifies the merge logic correctly handles the case where the same camera appears in 3 Geizhals categories (dslr, mirrorless, rangefinder). The existing `test_create_camera_key_year_mismatch` tests year-based deduplication but not category-based deduplication across Geizhals categories.

**Fix:** Add a test that simulates Geizhals data with the same camera in multiple categories and verifies that only one entry appears in the merged result.

---

### TE15-04: No test for openMVG CSV BOM handling
**File:** `tests/test_parsers_offline.py`; `sources/openmvg.py`, lines 52-56
**Severity:** LOW (test gap for a LOW bug) | **Confidence:** HIGH

The existing test for `parse_existing_csv` BOM handling (TE14-01) is good, but there's no corresponding test for the openMVG fetcher's `DictReader` path. When the BOM defense fix is applied to `openmvg.fetch()`, a test should verify that BOM-prefixed CSV content is handled correctly.

**Fix:** Add a test that mocks `http_get` to return BOM-prefixed CSV and verifies that `openmvg.fetch()` produces correct records.

---

## Summary
- NEW findings: 4 (3 MEDIUM test gaps, 1 LOW test gap)
- TE15-01: No Samsung NX classification test — MEDIUM
- TE15-02: No Canon EOS xxxD classification test — MEDIUM
- TE15-03: No triple-category duplicate test — MEDIUM
- TE15-04: No openMVG CSV BOM test — LOW
