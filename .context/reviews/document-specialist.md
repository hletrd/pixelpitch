# Document Specialist Review (Cycle 36) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

DOC35-01 (BOM comment vs code mismatch) fixed. DOC35-02 (pixel_pitch ValueError doc) moot — ValueError no longer raised after C35-01 fix added the `area <= 0` guard.

## New Findings

### DOC36-01: `pixel_pitch` docstring incomplete — does not mention NaN/inf handling

**File:** `pixelpitch.py`, lines 178-183
**Severity:** LOW | **Confidence:** HIGH

The `pixel_pitch` docstring says:

> Returns 0.0 when mpix <= 0 or area <= 0 (physically meaningless inputs) instead of raising ValueError from sqrt.

After the C36 fix, it should also mention NaN and inf:

> Returns 0.0 when mpix <= 0, area <= 0, or either argument is NaN/inf (physically meaningless or non-finite inputs) instead of raising ValueError from sqrt.

**Fix:** Update docstring to match the new guard behavior.

---

## Summary

- DOC36-01 (LOW): `pixel_pitch` docstring should mention NaN/inf handling
