# Test Engineer Review (Cycle 16) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository test coverage re-review after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
All previously-added tests pass.

## New Findings

### TE16-01: No test for `sensor_size_from_type` with invalid inputs (1/0, 1/, negative)
**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

No tests for crash-inducing inputs: `1/0` (ZeroDivisionError), `1/` (ValueError), `1/-1` (nonsensical result). After fixing C16-01, tests should verify these return None instead of crashing.

---

### TE16-02: No test for `merge_camera_data` with duplicate keys in new_specs
**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

No test for the case where `new_specs` contains two entries with the same `create_camera_key`. After fixing C16-02, a test should verify duplicates in `new_specs` are merged into a single entry.

---

### TE16-03: No test for Pentax DSLR regex patterns beyond K-1/K-3
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

No test for Pentax cameras in the openMVG CSV parser test. Adding a "Pentax K3" row to the test CSV would verify the regex covers hyphenless K-mount models.

---

### TE16-04: No test for `http_get` exception handling completeness
**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

No test verifying that `http_get` handles all exception types gracefully (URLError, HTTPError, TimeoutError, and after S16-02 fix, OSError subclasses).

---

## Summary
- NEW findings: 4 (2 MEDIUM, 2 LOW)
- TE16-01: No test for sensor_size_from_type invalid inputs — MEDIUM
- TE16-02: No test for merge_camera_data duplicate new_specs — MEDIUM
- TE16-03: No test for Pentax DSLR regex — LOW
- TE16-04: No test for http_get exception handling — LOW
