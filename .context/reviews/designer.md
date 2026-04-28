# Designer Review (Cycle 21) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28

## D21-01: SpecDerived stale fields cause "unknown" display for preserved data

**Severity:** MEDIUM | **Confidence:** HIGH

When the merge function preserves `spec.size` and `spec.pitch` at the Spec level but not the SpecDerived level, cameras that should display sensor size and pixel pitch show "unknown" instead. This is a user-facing data quality issue that makes the site appear less comprehensive than it actually is.

**Impact:** 30.5% of cameras (532) have no size in the current CSV. Many of these could display preserved data if the SpecDerived fields were properly updated. Users relying on the site for sensor comparisons would see gaps in the data that don't actually exist.

---

## Summary

- D21-01 (MEDIUM): "unknown" display for cameras with preserved data — user-facing data gap
