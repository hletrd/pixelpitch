# Document Specialist Review (Cycle 20) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28

## Findings

No NEW doc/code mismatches found. All docstrings accurately describe the current code behavior. The openMVG docstring correctly describes the DSLR regex coverage after C17-04 fix. The `merge_camera_data` docstring correctly describes the year preservation behavior.

The `pixel_pitch` function has no docstring mentioning its behavior for non-positive mpix values, but this is because the function currently crashes on such input. Once C20-01 is fixed, a docstring note about the guard would be appropriate.

---

## Summary

No new actionable findings.
