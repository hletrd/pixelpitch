# Document Specialist Review (Cycle 25) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes

## Previous Findings Status

DOC24-01 (TYPE_FRACTIONAL_RE comment) addressed in C24-04.

## New Findings

### DOC25-01: parse_sensor_field docstring does not mention Unicode × or space handling limitations

**File:** `pixelpitch.py`, lines 530-539
**Severity:** LOW | **Confidence:** MEDIUM

The `parse_sensor_field()` docstring shows examples like `"36.0x24.0mm"` and `"6.94µm Pixelgröße"` which only use ASCII `x` and micro sign `µ`. The docstring does not mention that Unicode `×`, Greek mu `μ`, or spaces around `x` are not supported. If the regex is later upgraded to match `SIZE_MM_RE`/`PITCH_UM_RE` behavior, the docstring should be updated to reflect the expanded format support.

**Fix:** If/when the regex patterns are upgraded, update the docstring to show additional supported formats.

---

## Summary

- DOC25-01 (LOW): parse_sensor_field docstring does not mention format limitations
