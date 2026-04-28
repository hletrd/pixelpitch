# Architect Review (Cycle 33) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

ARCH31-01 (Spec/SpecDerived pitch duplication) fixed in C31. ARCH31-02 (BOM duplication) fixed in C31. All previous architectural concerns remain deferred.

## New Findings

### ARCH33-01: Truthy-vs-None pattern is a systemic design inconsistency

**Files:** pixelpitch.py (derive_spec, sorted_by, prettyprint, write_csv), templates/pixelpitch.html
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

The C32-01 fix addressed the serialization layer (write_csv) by replacing truthy checks with explicit `is not None` checks. However, the same pattern persists in the computation layer (derive_spec), sorting layer (sorted_by), display layer (prettyprint), and presentation layer (Jinja2 templates).

This is a design inconsistency: the data model allows 0.0 as a valid float value distinct from None, but only the serialization layer was updated to respect this. The other layers still treat 0.0 as equivalent to None.

The root cause is that the truthy-vs-None distinction was fixed locally (write_csv only) rather than holistically (all code paths that handle Optional[float] fields). A consistent approach would be to enforce `is not None` checks across all code paths that read Optional[float] fields.

**Fix:** Apply the same C32-01 pattern to derive_spec, sorted_by, prettyprint, and Jinja2 templates. Consider a project-wide search for `if <optional_float>` patterns.

---

## Summary

- ARCH33-01 (LOW-MEDIUM): Truthy-vs-None pattern is systemic — C32-01 fix was incomplete, needs holistic application
