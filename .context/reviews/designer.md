# Designer Review (Cycle 14) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full UI/UX review of Jinja2 templates, Bootstrap, D3.js, jQuery, tablesorter

## Previously Fixed (Cycles 1-13) — Confirmed Resolved
All previous UI/UX fixes remain intact. C13 findings were code-only, no UI impact.

## Deferred Items Still Valid
- F35: Box plot hardcoded dimensions — DEFERRED
- F36: No skip-to-content link — DEFERRED
- F37: Filter dropdown doesn't show current state — DEFERRED
- F38: No loading indicator or pagination — DEFERRED
- F39: Navbar 9 items on mobile — DEFERRED
- UX11-01: Scatter plot year axis label overlap — DEFERRED
- UX11-02: "Hide possibly invalid data" label unclear — DEFERRED

## New Findings

### UX14-01: openMVG DSLR misclassification causes visible duplicate entries on All Cameras page
**File:** `sources/openmvg.py`, lines 63-69 (data); `templates/pixelpitch.html` (display)
**Severity:** MEDIUM | **Confidence:** HIGH

The openMVG category misclassification (C14-01) has a direct UX impact: the "All Cameras" page shows duplicate entries for DSLR cameras. For example, "Canon EOS 5D" appears twice — once with "Mirrorless" category and once with "DSLR" category. This is confusing for users and undermines the site's credibility as a reference database.

The category column on the All Cameras page makes the duplicate especially visible because the same camera name appears with different category labels.

**Fix:** Fix the openMVG category heuristic (same as C14-01 code fix).

---

## Summary
- NEW findings: 1 (1 MEDIUM)
- UX14-01: Duplicate entries visible on All Cameras page — MEDIUM
- No other UI/UX regressions
