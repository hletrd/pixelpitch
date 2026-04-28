# Architect Review (Cycle 22) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## A22-01: `merge_camera_data` field preservation is ad-hoc and fragile

**Severity:** LOW | **Confidence:** HIGH

The C22-01 bug (`elif` misattachment) is a symptom of a deeper design issue. The merge function now has 8 separate `if` statements for field preservation:

1. `spec.type` preservation
2. `spec.size` preservation
3. `spec.pitch` preservation
4. `spec.mpix` preservation
5. `spec.year` preservation (with elif for year-change logging)
6. `derived.size` preservation
7. `derived.area` preservation
8. `derived.pitch` preservation

This list will grow as more fields are added to the data model. The ad-hoc `if` chain is error-prone: inserting code in the middle (as C21-01 did) can break the conditional structure.

**Recommendation:** Extract a generic field-preservation helper:
```python
def _preserve_none_fields(new_obj, existing_obj, field_names):
    for field in field_names:
        if getattr(new_obj, field) is None and getattr(existing_obj, field) is not None:
            setattr(new_obj, field, getattr(existing_obj, field))
```

This would make the preservation logic declarative and less prone to insertion errors. However, the year-change logging is a special case that would still need separate handling.

This is an architectural improvement, not a correctness fix. It can be deferred.

---

## Summary

- A22-01 (LOW): Field preservation logic is ad-hoc and fragile — consider generic helper
