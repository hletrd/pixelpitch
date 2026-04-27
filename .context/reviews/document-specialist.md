# Document Specialist Review (Cycle 16) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository doc/code review after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
- C15-06: openMVG docstring now warns about DSLR regex limitations
- All previous doc fixes remain intact

## New Findings

### DS16-01: `sensor_size_from_type` docstring does not mention invalid input handling
**File:** `pixelpitch.py`, lines 139-165
**Severity:** LOW | **Confidence:** HIGH

The docstring describes what the function does for valid inputs but does not mention behavior for invalid inputs (1/0, 1/, etc.). After fixing C16-01, the docstring should be updated to state that invalid fractional types return None.

---

### DS16-02: `merge_camera_data` docstring does not mention duplicate handling among new_specs
**File:** `pixelpitch.py`, lines 349-407
**Severity:** LOW | **Confidence:** HIGH

The function's docstring does not describe how duplicates within new_specs are handled. After fixing C16-02, the docstring should state that entries with duplicate keys in new_specs are merged (first wins or last wins).

---

## Summary
- NEW findings: 2 (2 LOW)
- DS16-01: sensor_size_from_type docstring gap — LOW
- DS16-02: merge_camera_data docstring gap — LOW
