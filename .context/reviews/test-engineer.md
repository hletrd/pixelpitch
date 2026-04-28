# Test Engineer Review (Cycle 17) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository test coverage re-review after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- TE16-01 (sensor_size_from_type tests): Fixed — tests for 1/0, 1/0.0, 1/, 1/-1 all present and passing.
- TE16-02 (merge dedup tests): Fixed — test_merge_camera_data includes "dedup new_specs same key" and "dedup new_specs with existing" tests.
- TE16-03 (Pentax DSLR regex tests): Fixed — Pentax K3 and 645Z test cases in openMVG CSV parser test.
- TE16-04 (http_get exception test): Not yet added, but LOW priority.

## Current Test Coverage Assessment

The gate test suite (`test_parsers_offline.py`) has 98 checks covering:
- IR parsing and name normalization
- Apotelyt parsing and body category
- GSMArena parsing, lens selection, and type/size
- openMVG CSV parsing, BOM handling, DSLR regex
- Merge logic (overlap, preservation, new-only, year mismatch, dedup)
- CSV schema, round-trip, parse_existing_csv edge cases
- deduplicate_specs edge cases
- sensor_size_from_type including invalid inputs
- pixel_pitch, match_sensors, load_sensors_database, load_csv
- CineD FORMAT_TO_MM completeness
- about.html rendering
- Category dedup

## New Findings

### TE17-01: No test for Pentax KP, KF, K-r, K-x DSLR classification
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The openMVG CSV test includes Pentax K3 and 645Z but not Pentax KP, KF, K-r, or K-x. After the C17-01 fix, these should be tested to prevent regression.

**Fix:** Add 'Pentax,KP,...' and 'Pentax,KF,...' rows to the test CSV and add expect() calls verifying they are classified as "dslr".

---

### TE17-02: No test for Nikon Df DSLR classification
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

No test for Nikon Df (letter-suffix D-series DSLR). After the C17-02 fix, this should be tested.

**Fix:** Add a 'Nikon,Df,...' row to the test CSV and verify it is classified as "dslr".

---

### TE17-03: No test for GSMArena Unicode curly-quote sensor format
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

No test verifying that GSMArena's SENSOR_FORMAT_RE handles Unicode right double quotation marks (U+2033). If C17-03 is fixed, a test should verify the fix.

**Fix:** Add a test case with a curly-quote sensor format string.

---

## Summary
- NEW findings: 3 (all LOW)
- TE17-01: No test for Pentax KP/KF/K-r/K-x — LOW
- TE17-02: No test for Nikon Df — LOW
- TE17-03: No test for GSMArena curly-quote format — LOW
