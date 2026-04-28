# Designer Review (Cycle 19) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository UI/UX re-review after cycles 1-18 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-18) — Confirmed Resolved

- UX18-01 (scatter plot hidden data): Fixed — visibility filter now respected.
- UX18-02 (sensor-size numeric sort): Implemented custom parser, but...

## New Findings

### UX19-01: Sensor Size numeric sort broken on 8 of 9 pages — regression from C18-08
**File:** `templates/pixelpitch.html`, lines 228-258
**Severity:** MEDIUM | **Confidence:** HIGH

The custom `sensor-width` parser is applied to column index 2 on all pages, but on non-"all" pages, Sensor Size is at column index 1 (because the Category column is absent). The result:

- **"All Cameras" page:** Sensor Size sorts numerically. Correct.
- **All other pages (DSLR, mirrorless, etc.):** Sensor Size sorts alphabetically (text parser applied instead). Resolution sorts by sensor width instead of megapixels.

**Concrete failure scenario:**
1. User navigates to the DSLR page
2. Clicks "Sensor Size" column header to sort by size
3. Table sorts alphabetically: "9.84 x 7.40 mm" appears after "35.9 x 23.9 mm"
4. User expected "9.84 x 7.40 mm" to appear first (smaller sensor)
5. User clicks "Resolution" to try that instead
6. Table sorts by sensor width instead of megapixels

**Fix:** Use conditional Jinja2 template blocks to assign correct column indices based on `page == "all"`.

---

## Summary
- NEW findings: 1 (MEDIUM)
- UX19-01: Sensor Size numeric sort broken on non-"all" pages — MEDIUM
