# Document Specialist Review (Cycle 27) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes

## Previous Findings Status

DOC26-01 (MPIX_RE docstring) — addressed in C26-01 when MPIX_RE was centralized. The `sources/__init__.py` docstring and comments already document the shared pattern's format support.

## New Findings

### DOC27-01: sources/__init__.py PITCH_UM_RE comment does not document "um" support status

**File:** `sources/__init__.py`, line 66 and module docstring
**Severity:** LOW | **Confidence:** HIGH

The module-level comment (lines 43-47) documents the shared patterns:

```python
# PITCH_UM_RE matches µm, μm (Greek mu), "microns", "um", and HTML entities.
```

Wait — actually the comment on line 45 says:

```
# PITCH_UM_RE matches µm, μm (Greek mu), "microns", "um", and HTML entities.
```

But the actual regex on line 66 does NOT match "um":

```python
PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|microns?|μm|&micro;m|&#0?956;m)", re.IGNORECASE)
```

This is a doc/code mismatch — the comment claims "um" is supported, but the regex does not include it.

**Verified:** `PITCH_UM_RE.search("5.12 um")` returns None, contradicting the comment.

**Fix:** Either add "um" to the regex (preferred, as the comment claims it's supported) or remove "um" from the comment.

---

## Summary

- DOC27-01 (LOW): PITCH_UM_RE comment claims "um" support but regex does not match it
