# Debugger Review (Cycle 15) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-14 fixes

## Previously Fixed (Cycles 1-14) — Confirmed Resolved
All previous fixes verified. No regressions detected in previously-fixed items.

## New Findings

### D15-01: Canon EOS xxxD DSLRs classified as mirrorless — regex bug from C14-01
**File:** `sources/openmvg.py`, lines 36-48
**Severity:** MEDIUM | **Confidence:** HIGH

Failure mode:
1. openMVG fetches Canon EOS 250D (APS-C sensor) → sensor width >= 20mm → interchangeable-lens branch
2. `_DSLR_NAME_RE.search("Canon EOS 250D")` → pattern `EOS[-\s]+\dD` requires single digit before D → NO MATCH
3. `category = "mirrorless"` (fallback default)
4. Same camera also fetched from Geizhals with `category="dslr"` → different merge keys → duplicate
5. User sees Canon EOS 250D twice on the All Cameras page

**Failure scenario:** A user browsing the All Cameras page sees Canon EOS 250D listed under both "Mirrorless" and "DSLR" categories.

The root cause is the regex pattern `Canon\s+EOS[-\s]+\dD` which only matches single-digit xD models (5D, 6D, 7D, 1D) and misses the xxxD/xxxD/xxxxD Rebel series.

**Fix:** Change `\dD` to `\d+D` in the Canon EOS pattern.

---

### D15-02: Samsung NX cameras classified as DSLR — wrong pattern from C14-01
**File:** `sources/openmvg.py`, line 44
**Severity:** MEDIUM | **Confidence:** HIGH

Failure mode:
1. openMVG fetches Samsung NX300 → sensor width >= 20mm → interchangeable-lens branch
2. `_DSLR_NAME_RE.search("Samsung NX300")` → `Samsung\s+NX\d{3}` matches → `category="dslr"`
3. Samsung NX300 is actually mirrorless — appears on DSLR page (wrong)
4. If the same camera also appears in Geizhals mirrorless data → different merge keys → duplicate

**Failure scenario:** A user sees Samsung NX300 on the DSLR page, which is incorrect — Samsung NX cameras are all mirrorless.

**Fix:** Remove `Samsung\s+NX\d{3}` from `_DSLR_NAME_RE`.

---

### D15-03: Geizhals rangefinder data creates 43 triple-duplicate entries
**File:** `pixelpitch.py`, lines 740-747, 339
**Severity:** MEDIUM | **Confidence:** HIGH

Failure mode:
1. Geizhals DSLR URL → Canon EOS 5D Mark IV → `category="dslr"`
2. Geizhals Rangefinder URL → Canon EOS 5D Mark IV → `category="rangefinder"` (misclassified by Geizhals)
3. openMVG → Canon EOS 5D Mark IV → `category="mirrorless"` (misclassified by regex)
4. Three different merge keys → three entries on All Cameras page
5. User sees the same camera 3 times with 3 different category labels

For mirrorless cameras (Canon EOS R5), the failure mode produces 2 entries (mirrorless + rangefinder).
For DSLRs (Canon EOS 5D), the failure mode produces 3 entries (dslr + mirrorless + rangefinder).

**Failure scenario:** 43 cameras appear 3 times on the All Cameras page with different category labels, undermining trust in the database.

**Fix:** Add category normalization during merge to discard Geizhals rangefinder entries that duplicate existing dslr/mirrorless entries.

---

### D15-04: openMVG CSV DictReader vulnerable to BOM — 0-record failure
**File:** `sources/openmvg.py`, lines 52-56
**Severity:** LOW | **Confidence:** HIGH

Failure mode:
1. openMVG GitHub CSV gets BOM (e.g., maintainer edits with Excel)
2. `http_get` returns BOM-prefixed string
3. `csv.DictReader` → first fieldname = `"﻿CameraMaker"` (BOM prefix)
4. `row.get("CameraMaker")` returns None → `if not maker or not model: continue`
5. Every row skipped → 0 records returned
6. Site generated without openMVG cameras → silent data loss

**Failure scenario:** After a BOM introduction, the next CI build produces a site missing all openMVG cameras. The data loss is silent.

**Fix:** Strip BOM from the CSV body before passing to `DictReader`.

---

## Summary
- NEW findings: 4 (3 MEDIUM, 1 LOW)
- D15-01: Canon EOS xxxD regex false negative — MEDIUM
- D15-02: Samsung NX regex false positive — MEDIUM
- D15-03: 43 triple-duplicate entries from Geizhals rangefinder — MEDIUM
- D15-04: openMVG CSV BOM vulnerability — LOW
