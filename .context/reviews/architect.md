# Architect Review (Cycle 44) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## Previous Findings Status

ARCH43-01 (spec.size provenance ambiguity) — COMPLETED. GSMArena and CineD now leave spec.size=None for type-derived/format-derived dimensions.
ARCH42-02 (circular import gsmarena<->pixelpitch) — deferred.

## New Findings

### ARCH44-01: FORMAT_TO_MM dict in cined.py creates false coupling to removed behavior — dead code that suggests it's still used

**File:** `sources/cined.py, lines 37-51`
**Severity:** LOW | **Confidence:** HIGH

The FORMAT_TO_MM dict was the mechanism by which CineD set spec.size from format class names. After C43-01, this mechanism is no longer used. But the dict still exists, creating the appearance that format-derived sizes are still set. This is architectural dead weight that misleads about the data flow. A new developer reading the code would assume FORMAT_TO_MM is used somewhere because it's defined at module level.

**Fix:** Remove FORMAT_TO_MM. If format class detection is needed in the future, it can be re-added with proper provenance tracking.

---

### ARCH44-02: CineD format extraction regex runs but result is unused — wasted computation and misleading data flow

**File:** `sources/cined.py, _parse_camera_page, lines 92-119`
**Severity:** LOW | **Confidence:** HIGH

The format extraction regex (fmt_m) and fmt variable are computed in _parse_camera_page but never used after C43-01 removed the FORMAT_TO_MM.get call. This is wasted computation in a browser-dependent scraper (where every millisecond of page processing counts). More importantly, it suggests a data flow that doesn't exist.

**Fix:** Remove the fmt_m regex, fmt variable, and the `if size is None and fmt:` block.

---


## Summary

- ARCH44-01 (LOW): FORMAT_TO_MM dict in cined.py creates false coupling to removed behavior — dead code that suggests it's still used
- ARCH44-02 (LOW): CineD format extraction regex runs but result is unused — wasted computation and misleading data flow
