# Architect Review (Cycle 20) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## A20-01: Merge function field preservation is ad-hoc, not schema-driven
**Severity:** LOW | **Confidence:** HIGH

The `merge_camera_data` function has special-case logic for `year` preservation but not for `type`, `size`, or `pitch`. This ad-hoc approach means each field's merge behavior must be individually coded and tested. A schema-driven merge (e.g., a merge strategy per field: "prefer new", "prefer existing if new is None", "prefer non-None") would be more maintainable and less error-prone.

However, this is an architectural improvement, not a correctness fix. The current behavior works correctly for the primary use case (Geizhals data is always more complete than source data).

**Recommendation:** Document the merge behavior for each field. Consider a field-level merge strategy if the source diversity grows.

---

## Summary

No new actionable findings beyond what code-reviewer and critic already identified.
