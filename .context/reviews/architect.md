# Architect Review (Cycle 16) — Architectural/Design Risks, Coupling, Layering

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository architecture re-review after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
All previous architectural fixes remain intact.

## New Findings

### A16-01: `merge_camera_data` has no deduplication contract for its input — architectural gap
**File:** `pixelpitch.py`, lines 349-407
**Severity:** MEDIUM | **Confidence:** HIGH

The function's contract is: "merge new camera data with existing camera data." It deduplicates against existing data but assumes new_specs is already deduplicated. This implicit precondition is not documented or enforced. When multiple sources contribute the same camera (same name + category), the function produces duplicates.

Architecturally, the function should either:
1. Document that new_specs must be pre-deduplicated, OR
2. Handle duplicates in new_specs internally

Option 2 is more robust because callers should not need to know about this precondition.

---

### A16-02: `digicamdb` source module is a pure alias — violates DRY at the registry level
**File:** `sources/digicamdb.py`; `pixelpitch.py`, line 985
**Severity:** LOW | **Confidence:** HIGH

The digicamdb module delegates entirely to openmvg.fetch(), creating identical Spec objects. Having both in SOURCE_REGISTRY means the same data could be fetched and stored twice. This is not a DRY violation in code (the module is trivial), but it IS a DRY violation at the data level — two source CSVs with identical content.

---

### A16-03: `sensor_size_from_type` lacks input validation — defensive programming gap
**File:** `pixelpitch.py`, lines 152-165
**Severity:** MEDIUM | **Confidence:** HIGH

The function performs arithmetic on parsed input without validation. From an architectural perspective, any function that processes external data (from HTML parsing) should validate its inputs and fail gracefully rather than crashing. The fix is to add a try/except block.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- A16-01: merge_camera_data input dedup contract gap — MEDIUM
- A16-02: digicamdb registry-level DRY violation — LOW
- A16-03: sensor_size_from_type defensive programming gap — MEDIUM
