# Architect Review (Cycle 39) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## Previous Findings Status

ARCH38-01 fixed. Template renders "unknown" for 0.0 pitch/mpix.

## New Findings

### ARCH39-01: Template validation is piecemeal — `!= 0.0` should be `> 0` for physical-quantity fields

**File:** `templates/pixelpitch.html`, lines 76-80, 84-88
**Severity:** MEDIUM | **Confidence:** HIGH

The C38-01 fix added `!= 0.0` checks to the template guards for pitch and mpix. This is a piecemeal approach that addresses only the specific 0.0 sentinel case. The correct architectural pattern for physical quantities that must be positive is `> 0`, which handles the entire class of invalid values (zero, negative, NaN) in a single condition.

This is a recurring pattern: each cycle finds a new "impossible" value that leaks through, and a new `!= X` check is added. The `> 0` guard eliminates this pattern entirely.

**Fix:** Replace `!= 0.0` with `> 0` in both pitch and mpix template guards.

---

## Summary

- ARCH39-01 (MEDIUM): Template validation should use `> 0` (positivity) instead of `!= 0.0` (exclusion) for physical quantities
