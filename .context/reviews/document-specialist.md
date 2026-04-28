# Document Specialist Review (Cycle 35) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

DOC33-01 (derive_spec docstring "always takes precedence" vs 0.0 pitch) fixed in C33.

## New Findings

### DOC35-01: `_BOM` comment claims escape sequence is used, but literal is used

**File:** `sources/__init__.py`, lines 87-90
**Severity:** MEDIUM | **Confidence:** HIGH

The comment on lines 87-89 explicitly states:

> Using the escape sequence rather than the literal character guards against editors or CI pipelines that silently strip or normalise the invisible BOM glyph when re-encoding source files.

But line 90 uses the literal BOM character (`\xef\xbb\xbf` in raw bytes), not the escape sequence (`﻿`). The comment directly contradicts the implementation.

**Fix:** Either:
1. Replace the literal with the actual escape sequence `﻿` (preferred — makes the code match the documented intent), or
2. Update the comment to reflect that the literal is used (not recommended — defeats the documented purpose)

---

### DOC35-02: `pixel_pitch` docstring does not document ValueError for negative area

**File:** `pixelpitch.py`, lines 178-181
**Severity:** LOW | **Confidence:** HIGH

The `pixel_pitch` function docstring does not mention that it raises `ValueError` when `area < 0`. The function signature accepts `float` for area, and the docstring only states the return type is `float`. Callers have no indication that negative area is unsupported.

**Fix:** Add a "Raises" section to the docstring:
```
Raises:
    ValueError: If area < 0 (sqrt of negative number)
```

Or better, add a guard that returns 0.0 for negative area (as done for mpix <= 0).

---

## Summary

- DOC35-01 (MEDIUM): `_BOM` comment claims escape sequence, code uses literal — direct contradiction
- DOC35-02 (LOW): `pixel_pitch` docstring does not document ValueError for negative area
