# Designer Review (Cycle 43) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28

## Previous Findings Status

DES42-01 fixed. The merge size inconsistency (C42-01) means derived.size now matches spec.size after merge.

## New Findings

### DES43-01: GSMArena/CineD lookup-table sizes shown as measured data in UI — users see approximations instead of accurate values

**File:** `sources/gsmarena.py`, line 146; `sources/cined.py`, lines 94-102; `templates/pixelpitch.html`
**Severity:** MEDIUM | **Confidence:** HIGH

When GSMArena sets `spec.size` from the TYPE_SIZE lookup table and the merge does not preserve a more accurate Geizhals measured value (because `spec.size is not None`), the template displays the TYPE_SIZE approximation as if it were a measured value. For users:

1. **Wrong sensor dimensions displayed.** A phone with measured 9.76x7.30mm shows as 9.84x7.40mm (from the TYPE_SIZE lookup table).
2. **Wrong scatter plot positions.** The D3 scatter plot uses `data-sensor-width` for positioning and tooltips.
3. **No visual indication of approximation.** Unlike template-derived sizes (where `spec.size is None` and the template shows the type-derived value), these lookup values are stored as `spec.size` and indistinguishable from measured data.

From a UX perspective, this is worse than the C42-01 issue because there's no way for users to know the displayed value is an approximation. The C42-01 fix at least corrected the inconsistency between spec and derived fields, but this issue means the "ground truth" itself is wrong.

**Fix:** Fix the provenance (CR43-02) so GSMArena/CineD don't set `spec.size` from lookup tables.

---

## Summary

- DES43-01 (MEDIUM): GSMArena/CineD lookup-table sizes shown as measured data in UI — users see approximations without any visual indication of uncertainty
