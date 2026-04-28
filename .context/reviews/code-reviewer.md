# Code Review (Cycle 44) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-43 fixes, focusing on NEW issues

## Previous Findings Status

C43-01 (GSMArena/CineD spec.size provenance) — COMPLETED. GSMArena now sets `spec.size = None` with `spec.type` only. CineD also leaves `spec.size = None` for format-class entries. All gate tests pass.

C43-02 (redundant derived.pitch write) — COMPLETED. The `derived.pitch` write was initially removed but then restored with improved comments explaining why it's needed (for the case where spec.pitch is None and pitch was computed from old area).

## New Findings

### CR44-01: FORMAT_TO_MM dict defined but never used in _parse_camera_page after C43-01 fix — dead code

**File:** `sources/cined.py`
**Severity:** LOW | **Confidence:** HIGH

After the C43-01 fix removed `FORMAT_TO_MM.get(fmt.lower())` from _parse_camera_page, the FORMAT_TO_MM dict is defined at module level but never used in any function. The module docstring says it's 'kept for the regex coverage test only', but there is no such test that actually references FORMAT_TO_MM. This is dead code that could confuse future maintainers into thinking it's used.

**Fix:** Remove the FORMAT_TO_MM dict entirely, or add a test that references it (e.g., verifying all format names in FORMAT_TO_MM have corresponding regex patterns in the format extraction regex).

---


## Summary

- CR44-01 (LOW): FORMAT_TO_MM dict defined but never used in _parse_camera_page after C43-01 fix — dead code
