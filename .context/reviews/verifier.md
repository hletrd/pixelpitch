# Verifier Review (Cycle 22) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28

## V22-01: Year-change log unreachable — verified by code inspection

**File:** `pixelpitch.py`, lines 428-437
**Severity:** MEDIUM | **Confidence:** HIGH (static analysis confirmed)

Verified the `elif` attachment issue by inspecting the code structure:

```python
# Line 428
if new_spec.pitch is None and existing_spec.pitch is not None:
    new_spec.pitch = existing_spec.pitch
# Line 429-437
elif (
    new_spec.spec.year is not None
    and existing_spec.spec.year is not None
    and new_spec.spec.year != existing_spec.spec.year
):
    print(...)
```

The `elif` on line 429 is syntactically attached to the `if` on line 428 (SpecDerived pitch preservation), not to the `if` on line 417 (Spec year preservation). This means:

- When `new_spec.pitch is None and existing_spec.pitch is not None` → the `if` fires (pitch preserved), `elif` skipped
- When `new_spec.pitch is not None` → the `elif` is evaluated (year change logged if applicable)
- When `new_spec.pitch is None and existing_spec.pitch is None` → the `elif` is evaluated

The year value itself is NOT affected — new data's year always takes precedence. Only the diagnostic message is affected.

**Correctness impact:** None for data. Medium for observability.

---

## Summary

- V22-01 (MEDIUM): Year-change log unreachable due to `elif` attachment — verified, no data impact
