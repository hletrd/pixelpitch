# Architect Review (Cycle 30) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-29 fixes

## Previous Findings Status

ARCH29-01 and ARCH29-02 both fixed in C29. All previous architectural concerns remain deferred.

## New Findings

### ARCH30-01: GSMArena fetch() lacks per-phone error handling — inconsistent with other sources

**File:** `sources/gsmarena.py`, lines 246-252
**Severity:** MEDIUM | **Confidence:** HIGH

After the C29-02 fix, IR and Apotelyt have per-camera try/except. CineD already had it. But GSMArena was missed, creating an inconsistency in the error-resilience pattern across the four browser/HTTP fetch loops.

**Fix:** Add per-phone try/except to `gsmarena.fetch()`.

---

### ARCH30-02: deduplicate_specs() manual Spec reconstruction violates DRY

**File:** `pixelpitch.py`, lines 655-665 and 669-675
**Severity:** LOW | **Confidence:** HIGH

The C29-04 fix simplified `digicamdb.py` to a true alias, but `deduplicate_specs()` in `pixelpitch.py` still manually reconstructs Spec objects field-by-field. This is the same DRY violation. If Spec gains a field, this code silently drops it.

**Fix:** Use `dataclasses.replace()`.

---

## Summary

- ARCH30-01 (MEDIUM): GSMArena fetch() lacks per-phone error handling
- ARCH30-02 (LOW): deduplicate_specs() manual Spec reconstruction violates DRY
