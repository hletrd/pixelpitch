# Document Specialist Review (Cycle 43) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28

## Previous Findings Status

DOC42-01 fixed. `merge_camera_data` docstring now documents the derived field consistency requirement.

## New Findings

### DOC43-01: GSMArena `_phone_to_spec` docstring doesn't explain that `size` is set from TYPE_SIZE lookup, not measured

**File:** `sources/gsmarena.py`, lines 107-167
**Severity:** LOW | **Confidence:** HIGH

The `_phone_to_spec` function sets `size = PHONE_TYPE_SIZE.get(sensor_type)` at line 146 and passes it as `spec.size`. The docstring doesn't document that this is a TYPE_SIZE lookup value, not a measured sensor dimension. This matters because `merge_camera_data` treats `spec.size is not None` as "this source has measured data" and won't preserve more accurate measured values from other sources.

**Fix:** Add a note to the `_phone_to_spec` docstring (and the `gsmarena.py` module docstring) explaining that `size` is derived from the fractional-inch type designation via the TYPE_SIZE lookup table, not from a direct measurement. This is important context for understanding merge behavior.

---

### DOC43-02: CineD `_parse_camera_page` sets spec.size from FORMAT_TO_MM without documenting provenance

**File:** `sources/cined.py`, lines 94-102
**Severity:** LOW | **Confidence:** HIGH

Same as DOC43-01 but for CineD. The function sets `size = FORMAT_TO_MM.get(fmt.lower())` and passes it as `spec.size`. The module docstring says "If only the format class is given, we leave size None and let pixelpitch.derive_spec compute area from TYPE_SIZE / format" but the code does the opposite — it sets `spec.size` from FORMAT_TO_MM.

This is a doc/code mismatch. The docstring says size will be None for format-only entries, but the code sets it from FORMAT_TO_MM.

**Fix:** Either update the docstring to document that FORMAT_TO_MM sizes are set as spec.size (and note the provenance implications), or fix the code to match the docstring (leave spec.size = None for format-derived sizes).

---

## Summary

- DOC43-01 (LOW): GSMArena _phone_to_spec doesn't document that size is from TYPE_SIZE lookup, not measured
- DOC43-02 (LOW): CineD docstring says "we leave size None" for format-class entries but code sets it from FORMAT_TO_MM — doc/code mismatch
