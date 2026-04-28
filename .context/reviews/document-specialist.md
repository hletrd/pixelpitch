# Document Specialist Review (Cycle 44) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28

## Previous Findings Status

All C43 findings resolved. No regressions.

## New Findings

### DOC44-01: CineD docstring claims FORMAT_TO_MM is 'kept for regex coverage test' but no such test exists

**File:** `sources/cined.py, module docstring`
**Severity:** LOW | **Confidence:** HIGH

The module docstring states 'The FORMAT_TO_MM table is kept for the regex coverage test only.' However, no test in test_parsers_offline.py references FORMAT_TO_MM. The docstring claim is inaccurate — there is no regex coverage test that uses FORMAT_TO_MM.

**Fix:** Either add a regex coverage test for FORMAT_TO_MM, or remove the dict and update the docstring.

---


## Summary

- DOC44-01 (LOW): CineD docstring claims FORMAT_TO_MM is 'kept for regex coverage test' but no such test exists
