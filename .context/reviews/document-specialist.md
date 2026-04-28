# Document Specialist Review (Cycle 28) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes

## Previous Findings Status

DOC27-01 (PITCH_UM_RE comment) fixed in C27 when "um" was added to the regex.

## New Findings

### DOC28-01: sources/__init__.py module docstring does not document the "um" addition to PITCH_UM_RE

**File:** `sources/__init__.py`
**Severity:** LOW | **Confidence:** MEDIUM

The module docstring at lines 1-23 documents the source modules but does not document the shared regex patterns or their supported formats. The comment in `pixelpitch.py` line 44 says:

```
# PITCH_UM_RE matches µm, μm (Greek mu), "microns", "um", and HTML entities.
```

This is now accurate after the C27-01 fix. However, `sources/__init__.py` itself has no inline comment documenting the PITCH_UM_RE pattern's supported formats. The pattern on line 66 is the canonical definition, and a developer reading only that file would not know what formats are supported without reading the regex.

**Fix:** Add a comment above the PITCH_UM_RE definition in `sources/__init__.py` listing the supported formats.

---

## Summary

- DOC28-01 (LOW): sources/__init__.py PITCH_UM_RE lacks inline documentation comment
