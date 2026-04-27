# Code Review (Cycle 15) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-14 fixes, focusing on NEW issues missed or introduced by previous fixes

## Previously Fixed (Cycles 1-14) — Confirmed Resolved

All previous fixes remain intact. C14-01 (openMVG DSLR heuristic), C14-02 (BOM defense), C14-03 (CineD regex), C14-04 (PHONE_TYPE_SIZE copy), C14-05 (openMVG docstring) all verified as correctly applied and tested.

## New Findings

### C15-01: openMVG DSLR regex misses Canon EOS xxxD Rebel-series DSLRs — introduced by incomplete C14-01 fix
**File:** `sources/openmvg.py`, lines 36-48 (`_DSLR_NAME_RE`)
**Severity:** MEDIUM | **Confidence:** HIGH

The C14-01 fix added `_DSLR_NAME_RE` to classify DSLRs by name pattern, but the Canon pattern only matches `EOS + single_digit + D` (e.g., EOS 5D, 6D, 7D). It misses the entire Canon Rebel/xxxD series which uses 2-4 digits before the final D: EOS 250D, 800D, 850D, 1200D, 2000D, 4000D, etc. All are DSLRs.

Verified against current dist data: 5 Canon EOS xxxD cameras are misclassified as "mirrorless" in the openMVG source data, causing duplicates on the All Cameras page.

**Concrete failure:** Canon EOS 250D (a DSLR) appears on the site classified as "mirrorless" from openMVG, alongside its correct "dslr" classification from Geizhals.

**Fix:** Change `Canon\s+EOS[-\s]+\dD` to `Canon\s+EOS[-\s]+\d+D` (allow multiple digits before the final D). This covers xxxD Rebels, xxD mid-range, and xD pro lines.

---

### C15-02: openMVG Samsung NX pattern classifies mirrorless cameras as DSLRs — introduced by C14-01 fix
**File:** `sources/openmvg.py`, line 44 (`Samsung\s+NX\d{3}`)
**Severity:** MEDIUM | **Confidence:** HIGH

The `_DSLR_NAME_RE` includes `Samsung\s+NX\d{3}` which matches Samsung NX100, NX200, NX300, NX500. But **all Samsung NX cameras are mirrorless** — Samsung never made a DSLR under the NX brand. The NX series uses the Samsung NX-mount which is exclusively mirrorless.

The original C14-01 fix comment says "Samsung NX300 (some were DSLR-style)" but "DSLR-style" refers to body shape, not the actual camera type. The NX300 is a mirrorless camera with an electronic viewfinder.

Verified: at least 1 Samsung NX camera is misclassified as "dslr" in the current dist data.

**Concrete failure:** Samsung NX300 (mirrorless) appears on the DSLR page instead of the Mirrorless page.

**Fix:** Remove `Samsung\s+NX\d{3}` from `_DSLR_NAME_RE`. Samsung has no DSLRs in the NX lineup.

---

### C15-03: openMVG Sigma SD regex `\d?` misses 2-digit models (SD10, SD14, SD15) — introduced by incomplete C14-01 fix
**File:** `sources/openmvg.py`, line 43 (`Sigma\s+SD\d?`)
**Severity:** LOW | **Confidence:** HIGH

The pattern `Sigma\s+SD\d?` only matches Sigma SD0 through SD9 and SD (no digit). But Sigma also made SD10, SD14, and SD15 — all Foveon-sensor DSLRs. The `\d?` quantifier (0 or 1 digit) misses these.

**Concrete failure:** Sigma SD10, SD14, SD15 (all DSLRs) would be classified as "mirrorless" by the openMVG heuristic if they appear in the dataset with sensor width >= 20mm.

**Fix:** Change `Sigma\s+SD\d?` to `Sigma\s+SD\d+` to match 1+ digits.

---

### C15-04: Geizhals "rangefinder" category (Messsucher) misclassifies 43 non-rangefinder cameras, causing triple-category duplicates
**File:** `pixelpitch.py`, lines 740-747 (CATEGORIES); `pixelpitch.py`, line 339 (`create_camera_key`)
**Severity:** MEDIUM | **Confidence:** HIGH

Geizhals's German "Messsucher" filter (`xf=1480_Messsucher`) returns not only actual rangefinder cameras (Leica M series) but also many interchangeable-lens cameras that have optical viewfinders — including 43 Canon EOS DSLRs and Fujifilm mirrorless cameras. Since `create_camera_key` uses `name + category` for deduplication, the same camera appearing under "dslr" (from Geizhals DSLR filter), "mirrorless" (from openMVG), and "rangefinder" (from Geizhals Messsucher filter) creates 3 separate entries.

Current dist data has exactly 43 cameras with triple-category duplicates (dslr + mirrorless + rangefinder). None of these 43 cameras are actual rangefinders.

**Concrete failure:** Canon EOS 5D Mark IV appears 3 times on the All Cameras page: once as "DSLR", once as "Mirrorless", once as "Rangefinder".

**Fix options:**
1. Normalize Geizhals rangefinder entries: if a camera name already exists in dslr/mirrorless data from Geizhals, skip the rangefinder duplicate during merge
2. Post-filter the Geizhals rangefinder category to only include cameras from known rangefinder manufacturers (Leica, Voigtlander, Zeiss)
3. Add a name-based rangefinder validation (actual rangefinders: Leica M*, Voigtlander Bessa*, Zeiss ZM*)

---

### C15-05: openMVG fetch has no BOM defense for the remote CSV — potential 0-record parse if upstream adds BOM
**File:** `sources/openmvg.py`, lines 52-56; `sources/__init__.py`, lines 48-61
**Severity:** LOW | **Confidence:** HIGH

The openMVG fetcher reads the CSV from GitHub and passes it directly to `csv.DictReader`. If the remote CSV ever contains a UTF-8 BOM (e.g., if the repository maintainer edits it with Excel), `DictReader`'s first fieldname becomes `"﻿CameraMaker"` instead of `"CameraMaker"`, causing `KeyError` on every row and returning 0 records. Unlike `parse_existing_csv` (which now strips BOM), the openMVG fetcher has no BOM defense.

Verified by test: feeding BOM-prefixed CSV to `csv.DictReader` produces `"﻿CameraMaker"` as the first header.

**Concrete failure:** If the openMVG CSV repository adds a BOM, the next CI build would fetch 0 records from openMVG, silently losing all openMVG cameras from the site.

**Fix:** Strip BOM from the CSV body before passing to `DictReader` in `openmvg.fetch()`, or use `encoding="utf-8-sig"` in `http_get`.

---

## Summary
- NEW findings: 5 (3 MEDIUM, 2 LOW)
- C15-01: Canon EOS xxxD DSLR regex incomplete — MEDIUM (regression from C14-01)
- C15-02: Samsung NX pattern misclassifies mirrorless as DSLR — MEDIUM (regression from C14-01)
- C15-03: Sigma SD regex misses 2-digit models — LOW (regression from C14-01)
- C15-04: Geizhals rangefinder misclassification causes 43 triple-duplicates — MEDIUM
- C15-05: openMVG CSV fetch has no BOM defense — LOW
