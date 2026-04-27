# Critic Review (Cycle 15) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-14 fixes

## Previously Fixed (Cycles 1-14) — Confirmed Resolved
All previous findings addressed or properly deferred. No regressions. C14-01 through C14-05 all fixed and tested.

## New Findings

### CR15-01: The C14-01 DSLR regex fix introduced two regressions — Samsung NX false positives and Canon xxxD false negatives
**File:** `sources/openmvg.py`, lines 36-48 (`_DSLR_NAME_RE`)
**Severity:** MEDIUM | **Confidence:** HIGH

The C14-01 fix was well-intentioned but poorly implemented. The regex has two bugs:

1. **Samsung NX false positives**: `Samsung\s+NX\d{3}` matches Samsung NX100/NX200/NX300/NX500, all of which are mirrorless cameras. Samsung never made a DSLR under the NX brand. The original comment "some were DSLR-style" confuses body styling with camera type.

2. **Canon EOS xxxD false negatives**: `Canon\s+EOS[-\s]+\dD` only matches single-digit xD models (5D, 6D, 7D, 1D). The entire Canon Rebel series (xxxD: 250D, 800D, 850D; xxD: 70D, 80D, 90D; and xxxxD: 1200D, 2000D, 4000D) is missed.

The net effect of these two bugs partially cancels out: some cameras are wrongly promoted to DSLR (Samsung NX), others are wrongly demoted to mirrorless (Canon xxxD). But the real-world impact is clear — verified against dist data: 6 cameras are misclassified (5 Canon xxxD as mirrorless, 1 Samsung NX as dslr).

**Fix:** Change Canon pattern to `Canon\s+EOS[-\s]+\d+D` and remove Samsung NX pattern entirely.

---

### CR15-02: Geizhals rangefinder (Messsucher) category is fundamentally misaligned with the site's category system — 43 triple-duplicates
**File:** `pixelpitch.py`, lines 740-747 (CATEGORIES); line 339 (create_camera_key)
**Severity:** MEDIUM | **Confidence:** HIGH

Geizhals's "Messsucher" filter returns 53 cameras, but only ~10 are actual rangefinders (Leica M series). The other 43 are interchangeable-lens cameras with optical viewfinders that Geizhals also lists under "DSLR" or "Mirrorless". Because the merge key includes category, the same camera generates 3 separate entries (dslr + mirrorless + rangefinder).

This is the most user-visible data-quality issue on the site: 43 cameras appear 3 times each on the All Cameras page.

The previous cycles focused on code-correctness issues; this is a data-correctness + architecture issue that directly impacts the user experience.

**Fix:** Add a deduplication step that normalizes Geizhals rangefinder entries against the DSLR/mirrorless data before merge. If a camera already exists in the Geizhals DSLR or mirrorless category, the rangefinder entry should be discarded (since the camera is not a true rangefinder).

---

### CR15-03: openMVG Sigma SD regex too restrictive — SD10/SD14/SD15 missed
**File:** `sources/openmvg.py`, line 43
**Severity:** LOW | **Confidence:** HIGH

The pattern `Sigma\s+SD\d?` only matches 0-1 digits after "SD", missing Sigma SD10, SD14, and SD15 — all of which are DSLRs. This is a minor gap since these are older cameras that may not appear in the openMVG dataset, but the regex should be correct.

**Fix:** Change to `Sigma\s+SD\d+`.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- CR15-01: DSLR regex regressions from C14-01 — MEDIUM
- CR15-02: Geizhals rangefinder misclassification causes 43 triple-duplicates — MEDIUM
- CR15-03: Sigma SD regex too restrictive — LOW
