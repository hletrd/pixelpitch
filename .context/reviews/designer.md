# Designer Review (Cycle 11) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full UI/UX review of Jinja2 templates, Bootstrap, D3.js, jQuery, tablesorter

## Previously Fixed (Cycles 1-10) — Confirmed Resolved
- About page title now shows "About Pixel Pitch" — FIXED (cycle 9)
- About page LD+JSON now uses @type: AboutPage — FIXED (cycle 9)
- C10-03: CSV download `./` prefix — FIXED (cycle 10)
- C10-05: selectattr('pitch') 0.0 edge case — FIXED (cycle 10)
- C10-06: Scatter plot error boundary — FIXED (cycle 10)

## Deferred Items Still Valid
- F35: Box plot hardcoded dimensions — DEFERRED
- F36: No skip-to-content link — DEFERRED
- F37: Filter dropdown doesn't show current state — DEFERRED
- F38: No loading indicator or pagination — DEFERRED
- F39: Navbar 9 items on mobile — DEFERRED

## New Findings

### UX11-01: Scatter plot year axis label overlap with many years
**File:** `templates/pixelpitch.html`, lines 376-394
**Severity:** LOW | **Confidence:** MEDIUM

The scatter plot uses `d3.scaleBand()` for the x-axis (years). With many years of data (20+ years of camera history), the band width shrinks and the year labels on the x-axis overlap, becoming unreadable. The axis text is not rotated.

**Fix:** Rotate x-axis labels 45 degrees when there are more than ~15 years, or use a time-based scale instead of a band scale.

---

### UX11-02: "Hide possibly invalid data" checkbox label is unclear for non-expert users
**File:** `templates/pixelpitch.html`, lines 156-160
**Severity:** LOW | **Confidence:** LOW

The label "Hide possibly invalid data" is checked by default, which means some cameras are hidden on first load. A non-expert user might not understand what "possibly invalid" means or why cameras are hidden. The feature is useful but the labeling could be clearer.

**Fix:** Consider a tooltip explaining what makes data "possibly invalid", or rename to something more descriptive like "Hide likely inaccurate specs".

---

## Summary
- NEW findings: 2 (both LOW)
- UX11-01: Scatter plot year axis label overlap — LOW
- UX11-02: "Hide possibly invalid data" label unclear — LOW
