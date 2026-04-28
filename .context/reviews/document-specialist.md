# Document Specialist Review (Cycle 33) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

DOC31-01 (derive_spec docstring) fixed in C31. All previous doc/code mismatches resolved.

## New Findings

### DOC33-01: derive_spec docstring claims "always takes precedence" but 0.0 pitch is overridden

**File:** `pixelpitch.py`, lines 692-734
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

The `derive_spec` docstring states:

> Pixel pitch: ``spec.pitch`` (direct measurement) always takes precedence.

But the code at line 722 uses a truthy check:

```python
if spec.pitch:
    pitch = spec.pitch
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)
```

If `spec.pitch=0.0`, the truthy check is False, and pitch is computed from area+mpix instead. The "always takes precedence" claim is violated for 0.0.

**Fix:** Either fix the code to match the docstring (use `if spec.pitch is not None:`) or update the docstring to document the 0.0 exception.

---

## Summary

- DOC33-01 (LOW-MEDIUM): derive_spec docstring claims "always takes precedence" but 0.0 pitch is overridden
