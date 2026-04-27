# Architect Review (Cycle 11) — Architectural/Design Risks, Coupling, Layering

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository architecture review after cycles 1-10 fixes

## Previously Noted (Deferred, Still Valid)
- F32: `pixelpitch.py` is a ~1024-line monolith — DEFERRED
- F31: No source Protocol/base class — DEFERRED
- A5-02: Template description blocks DRY violation — DEFERRED

## New Findings

### A11-01: `create_camera_key` couples identity to data quality — architectural issue
**File:** `pixelpitch.py`, lines 313-315
**Severity:** MEDIUM | **Confidence:** HIGH

The camera identity key includes the year, which is an optional data field. This means the identity of a camera changes based on data quality: a camera with `year=2021` from one source and `year=None` from another produces two different identity keys. This is an architectural design flaw — the identity should be based on intrinsic properties (name + category) that are always present, not on optional metadata that may be missing.

The openMVG source always provides `year=None`, so any camera that appears in both openMVG and another source will be duplicated. This affects potentially thousands of cameras.

**Fix:** Remove year from `create_camera_key`. The name+category is sufficient for identity. Year is metadata, not identity.

---

### A11-02: `gsmarena.PHONE_TYPE_SIZE` is a mutable alias to `pixelpitch.TYPE_SIZE` — coupling risk
**File:** `sources/gsmarena.py`, line 58
**Severity:** LOW | **Confidence:** HIGH

Already noted as C9-06 and documented with a comment. The alias means gsmarena directly references the same dict object as pixelpitch. If gsmarena were ever to modify this dict (even accidentally), it would corrupt the central table. The comment warns against this, but a safer approach would be to use a function that returns the value, or import the module and access `TYPE_SIZE` via attribute access.

This is already deferred. Re-confirming the finding is still valid.

---

## Summary
- NEW findings: 1 (1 MEDIUM)
- A11-01: create_camera_key couples identity to year (optional data) — MEDIUM
- No architectural regressions from previous cycles
