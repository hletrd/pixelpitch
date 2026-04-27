# Debugger Review (Cycle 14) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-13 fixes

## Previously Fixed (Cycles 1-13) — Confirmed Resolved
All previous fixes verified. No regressions detected. C13-01 (load_csv UnicodeDecodeError) and C13-02 (Sony fallback normalizations) both fixed and tested.

## New Findings

### D14-01: openMVG DSLR misclassification — systematic duplicate bug
**File:** `sources/openmvg.py`, lines 63-69
**Severity:** MEDIUM | **Confidence:** HIGH

Failure mode:
1. openMVG fetches Canon EOS 5D (36mm sensor width) → `category="mirrorless"`
2. Geizhals fetches Canon EOS 5D → `category="dslr"`
3. `create_camera_key` produces different keys → merge preserves both
4. "All Cameras" page shows Canon EOS 5D twice — once under "mirrorless", once under "dslr"
5. User sees a visible data-quality bug on the primary page

This is not an edge case — it affects every DSLR that openMVG covers. The openMVG dataset has thousands of cameras, many of which are DSLRs that also appear on Geizhals.

**Failure scenario:** A user browsing the "All Cameras" page sees "Canon EOS 5D" listed twice with different category labels, undermining trust in the database.

**Fix:** Add DSLR name-pattern detection to openMVG's category heuristic.

---

### D14-02: UTF-8 BOM in CSV causes 0-row parse — complete data loss
**File:** `pixelpitch.py`, lines 250-330
**Severity:** MEDIUM | **Confidence:** HIGH

Failure mode:
1. Developer opens `camera-data.csv` in Excel, saves as "CSV UTF-8"
2. Excel adds BOM (`\xef\xbb\xbf`)
3. `load_csv` reads file, BOM preserved as `﻿` in text
4. `parse_existing_csv` → `header[0] = "﻿id"` ≠ `"id"` → `has_id = False`
5. Column indices misaligned → `int("5.12")` ValueError on every row
6. 0 records parsed → `existing_specs = []`
7. `merge_camera_data(new_specs, [])` → only fresh data, no preserved cameras
8. Site regenerates with missing data — cameras that were only in existing data are gone

**Failure scenario:** After a BOM introduction, the next CI build produces a site missing all preserved cameras. The data loss is silent — no error is raised, the site just has fewer cameras.

**Fix:** Strip BOM at entry point of `parse_existing_csv`.

---

### D14-03: CineD "Super35" variant not captured by regex — camera data loss
**File:** `sources/cined.py`, lines 88-89
**Severity:** LOW | **Confidence:** MEDIUM

Failure mode:
1. CineD camera page contains "Super35" (no space) in body text
2. Regex `Super 35(?:\s*mm)?` requires space → no match
3. `FORMAT_TO_MM.get("super35")` exists but is never reached because `fmt_m` is None
4. If no explicit mm dimensions found → `size = None`
5. Camera may fail `if not (size or mpix): return None` check
6. Camera silently dropped from CineD data

**Failure scenario:** A cinema camera on CineD that only lists "Super35" format (without explicit mm dimensions) would be dropped from the site.

**Fix:** Extend regex to capture "Super35" (no space) and other missing variants.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- D14-01: openMVG DSLR duplicates — MEDIUM
- D14-02: BOM parse failure — MEDIUM
- D14-03: CineD "Super35" variant missed — LOW
