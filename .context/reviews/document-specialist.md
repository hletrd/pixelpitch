# Document Specialist Review (Cycle 21) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28

## Findings

No NEW doc/code mismatches found. The `merge_camera_data` docstring says it preserves `year` when new data has `year=None`, which is accurate. After the C20-03 fix, it also preserves `type`, `size`, and `pitch`, but the docstring should be updated to reflect this. However, the docstring will need another update after the C21-01 fix (SpecDerived field preservation), so it's best to update it as part of that fix.

---

## Summary

No new actionable findings. Docstring update deferred to C21-01 fix.
