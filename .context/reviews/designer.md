# Designer Review (Cycle 15) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full UI/UX review of Jinja2 templates, Bootstrap, D3.js, jQuery, tablesorter

## Previously Fixed (Cycles 1-13) — Confirmed Resolved
All previous UI/UX fixes remain intact. C14 findings were data-focused, no new UI changes.

## Deferred Items Still Valid
- F35: Box plot hardcoded dimensions — DEFERRED
- F36: No skip-to-content link — DEFERRED
- F37: Filter dropdown doesn't show current state — DEFERRED
- F38: No loading indicator or pagination — DEFERRED
- F39: Navbar 9 items on mobile — DEFERRED
- UX11-01: Scatter plot year axis label overlap — DEFERRED
- UX11-02: "Hide possibly invalid data" label unclear — DEFERRED

## New Findings

### UX15-01: 43 triple-duplicate cameras visible on All Cameras page — severe data-quality UX issue
**File:** `pixelpitch.py`, lines 740-747, 339; `templates/pixelpitch.html` (display)
**Severity:** MEDIUM | **Confidence:** HIGH

The combination of Geizhals rangefinder misclassification and openMVG DSLR regex bugs causes 43 cameras to appear 3 times on the All Cameras page, each with a different category label. The Category column on the All Cameras page makes these duplicates especially visible: the same camera name appears with "DSLR", "Mirrorless", and "Rangefinder" labels.

For a reference database site, this is a significant credibility issue. Users expect each camera to appear once with the correct category.

**Fix:** Fix the underlying data issues (DSL regex bugs + rangefinder normalization) to eliminate duplicates.

---

### UX15-02: Samsung NX cameras appear on wrong page (DSLR instead of Mirrorless) — C14-01 regression
**File:** `sources/openmvg.py`, line 44; `templates/pixelpitch.html` (display)
**Severity:** LOW | **Confidence:** HIGH

The Samsung NX pattern in `_DSLR_NAME_RE` causes Samsung NX300 (and similar) to appear on the DSLR page instead of the Mirrorless page. While there are only a few Samsung NX cameras in the dataset, placing them on the wrong page is confusing for users who know Samsung NX cameras are mirrorless.

**Fix:** Remove Samsung NX from the DSLR regex.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- UX15-01: 43 triple-duplicate entries on All Cameras page — MEDIUM
- UX15-02: Samsung NX on wrong page — LOW
