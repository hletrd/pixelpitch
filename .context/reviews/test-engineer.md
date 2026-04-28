# Test Engineer Review (Cycle 43)

**Reviewer:** test-engineer
**Date:** 2026-04-28

## Previous Findings Status

TE42-01 implemented — `test_merge_size_consistency` added and passing. TE42-02 (CLI --limit test) deferred as LOW.

## New Findings

### TE43-01: No test for GSMArena spec.size preventing merge preservation of measured values

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The GSMArena `_phone_to_spec` function sets `spec.size` from the TYPE_SIZE lookup table. This means when a Geizhals entry with measured `spec.size` is merged with a GSMArena entry, the merge sees `new_spec.spec.size is NOT None` and does NOT preserve the measured Geizhals value. There is no test for this scenario.

**Fix:** Add a test that creates the scenario: existing data with measured size, new data from GSMArena with type-derived size. Verify that the measured size is preserved (or that the merge correctly handles the conflict).

```python
def test_merge_gsmarena_size_not_overriding_measured():
    """Verify merge_camera_data preserves measured Geizhals spec.size
    even when new data has spec.size from TYPE_SIZE lookup (not None)."""
    import pixelpitch as pp
    from models import Spec

    # Existing: measured size from Geizhals (slightly different from TYPE_SIZE)
    existing_spec = Spec(name='Phone X', category='smartphone', type='1/1.3',
                         size=(9.76, 7.30), pitch=None, mpix=200.0, year=2025)
    existing = pp.derive_spec(existing_spec)
    existing.id = 0

    # New: size from TYPE_SIZE lookup (GSMArena sets spec.size from PHONE_TYPE_SIZE)
    new_spec = Spec(name='Phone X', category='smartphone', type='1/1.3',
                    size=(9.84, 7.40), pitch=None, mpix=200.0, year=2025)
    new = pp.derive_spec(new_spec)

    merged = pp.merge_camera_data([new], [existing])
    m = merged[0]
    # Currently: spec.size = (9.84, 7.40) from new data (TYPE_SIZE lookup)
    # Expected: spec.size = (9.76, 7.30) from existing (measured)
    # This test will FAIL until the fix is implemented
```

---

### TE43-02: No test for merge where new data has BOTH spec.size and spec.type set

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The existing `test_merge_size_consistency` tests the case where `new_spec.spec.size is None` and `spec.type is set`. There is no test where `new_spec.spec.size` is set (e.g., from a source like GSMArena) AND `spec.type` is also set, and existing data has a measured `spec.size` that differs. This is the CR43-02 scenario.

**Fix:** This would be covered by TE43-01.

---

## Summary

- TE43-01 (MEDIUM): No test for GSMArena spec.size preventing merge preservation of measured Geizhals values
- TE43-02 (LOW): No test for merge where new data has both spec.size and spec.type set (covered by TE43-01)
