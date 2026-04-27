# Designer Review (Cycle 16) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository UI/UX re-review after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
- All `target="_blank"` links have `rel="noopener noreferrer"`
- SRI hashes on all CDN resources
- Dark mode support with persistent theme toggle

## New Findings

### UX16-01: Duplicate camera entries visible on All Cameras page — user confusion
**File:** `pixelpitch.py`, merge_camera_data (C16-02)
**Severity:** MEDIUM | **Confidence:** HIGH

When the same camera appears in multiple sources with the same category, the All Cameras page shows duplicate rows. This is confusing for users who expect each camera to appear once. The tablesorter sorts by pitch, so the duplicates may not be adjacent, making them harder to spot.

**Fix:** Fix C16-02 (merge dedup) to prevent duplicates at the data level.

---

### UX16-02: Pentax cameras appear under wrong category (Mirrorless instead of DSLR)
**File:** `sources/openmvg.py`, line 47 (C16-03)
**Severity:** LOW | **Confidence:** HIGH

Pentax K3, K5, K7 and other DSLR models appear on the Mirrorless page instead of the DSLR page. This is misleading for users looking for Pentax DSLRs.

**Fix:** Fix C16-03 (Pentax regex) to correctly classify these cameras.

---

### UX16-03: Scatter plot does not show category differentiation
**File:** `templates/pixelpitch.html`, lines 329-426
**Severity:** NEGLIGIBLE | **Confidence:** LOW

The scatter plot shows all cameras as the same color regardless of category. Adding color coding by category would improve data visualization, but this is a feature enhancement, not a bug.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 1 LOW, 1 NEGLIGIBLE)
- UX16-01: Duplicate camera entries — MEDIUM
- UX16-02: Pentax misclassification — LOW
- UX16-03: Scatter plot color coding — NEGLIGIBLE (enhancement)
