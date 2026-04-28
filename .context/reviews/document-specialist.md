# Document Specialist Review (Cycle 26) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes

## Previous Findings Status

DOC25-01 (parse_sensor_field docstring format limitations) addressed as part of C25-01 fix.

## New Findings

### DOC26-01: parse_sensor_field docstring does not show "MP" or "Mega pixels" examples for mpix

**File:** `pixelpitch.py`, lines 527-537
**Severity:** LOW | **Confidence:** LOW

The `parse_sensor_field()` docstring was updated in C25-01 to show Unicode × and Greek mu examples, but it does not mention that mpix extraction happens in a separate function (`extract_specs()`) using `MPIX_RE`. If MPIX_RE is centralized to handle "MP" and "Mega pixels", the docstring for the module-level `MPIX_RE` pattern (or the `extract_specs()` function) should be updated to reflect the expanded format support.

**Status:** Deferred — will be addressed as part of C26-01 fix if MPIX_RE is centralized.

---

## Summary

- DOC26-01 (LOW): parse_sensor_field / MPIX_RE docstring does not mention "MP" or "Mega pixels" formats
