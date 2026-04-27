# Critic Review (Cycle 14) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-13 fixes

## Previously Fixed (Cycles 1-13) — Confirmed Resolved
All previous findings addressed or properly deferred. No regressions. C13-01 and C13-02 both fixed and tested.

## New Findings

### CR14-01: openMVG "mirrorless" classification is a systematic data-quality bug, not just a heuristic limitation
**File:** `sources/openmvg.py`, lines 63-69
**Severity:** MEDIUM | **Confidence:** HIGH

The openMVG dataset covers many classic DSLRs (Canon EOS 5D series, Nikon D800/850, etc.) that are classified as "mirrorless" because the heuristic only checks sensor width. This isn't a rare edge case — it's a systematic misclassification affecting potentially hundreds of cameras. The duplicate entries are visible on the All Cameras page, which is the primary landing page. The user experience impact is real: seeing "Canon EOS 5D" listed under both "mirrorless" and "dslr" categories is confusing and undermines the site's credibility as a reference database.

The previous cycles have focused on code-correctness issues; this is a data-correctness issue that has a direct user-visible impact.

**Fix:** Add a DSLR name-based heuristic to openMVG (e.g., if name contains "EOS" + digit pattern, or "D" + 1-3 digits at word boundary, classify as "dslr"). This is pragmatic but covers the vast majority of DSLRs in the openMVG dataset.

---

### CR14-02: BOM in CSV is a realistic failure mode — Excel saves UTF-8 with BOM by default
**File:** `pixelpitch.py`, lines 250-330; lines 238-247
**Severity:** MEDIUM | **Confidence:** HIGH

The BOM issue (C14-02) is not just theoretical. On macOS and Windows, Excel's default "Save as CSV" option produces "CSV UTF-8" which includes a BOM. A developer who opens `camera-data.csv` in Excel to check data and saves it would introduce a BOM, silently breaking the entire build pipeline. The error manifests as an empty site with no camera data — a catastrophic failure mode for a data-focused website.

The fix is trivial (strip BOM at parse entry point) and has no downside.

---

### CR14-03: CineD FORMAT_TO_MM has three unreachable entries — dead code that suggests missing regex coverage
**File:** `sources/cined.py`, lines 33-47; lines 88-89
**Severity:** LOW | **Confidence:** HIGH

The `super35`, `1 inch`, and `2/3-inch` entries in `FORMAT_TO_MM` are unreachable because the regex in `_parse_camera_page` doesn't capture those string patterns. This is dead code, but it also suggests the regex may be missing legitimate variants that CineD pages might use (e.g., "Super35" without a space). The `super35` entry in particular seems intentional — someone added it because they expected CineD to use that format string — but the regex doesn't support it.

**Fix:** Either extend the regex to capture the missing variants, or remove the unreachable entries.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- CR14-01: openMVG DSLR misclassification — systematic data-quality bug — MEDIUM
- CR14-02: BOM in CSV is a realistic failure mode — MEDIUM
- CR14-03: CineD unreachable FORMAT_TO_MM entries — LOW
