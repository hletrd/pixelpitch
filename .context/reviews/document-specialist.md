# Document Specialist Review (Cycle 31) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## Previous Findings Status

DOC29-01 (digicamdb docstring) fixed in C29. DOC30: no new findings.

## New Findings

### DOC31-01: derive_spec docstring missing — behavior not documented

**File:** `pixelpitch.py`, lines 680-704
**Severity:** LOW | **Confidence:** HIGH

The `derive_spec()` function has no docstring. It has non-obvious priority logic: `derived.pitch` is set to `spec.pitch` when available, otherwise computed from area+mpix. This priority is critical for understanding the merge consistency issue (CR31-01) but is undocumented.

**Fix:** Add a docstring explaining the pitch priority: direct measurement from `spec.pitch` takes precedence over computation from area+mpix.

---

## Summary

- DOC31-01 (LOW): derive_spec() missing docstring — pitch priority logic undocumented
