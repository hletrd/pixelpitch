# Architect Review (Cycle 15) — Architectural/Design Risks, Coupling, Layering

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository architecture review after cycles 1-14 fixes

## Previously Noted (Deferred, Still Valid)
- F32: `pixelpitch.py` is a ~1067-line monolith — DEFERRED
- F31: No source Protocol/base class — DEFERRED
- A5-02: Template description blocks DRY violation — DEFERRED

## Previously Fixed (Cycles 1-14)
- A14-01: openMVG category heuristic misalignment — PARTIALLY FIXED (C14-01 added DSLR regex, but regex has bugs)
- A14-02: No BOM defense in CSV parsing layer — FIXED

## New Findings

### A15-01: The DSLR regex approach in openMVG is fragile — C14-01 fix introduced new bugs
**File:** `sources/openmvg.py`, lines 36-48
**Severity:** MEDIUM | **Confidence:** HIGH

The C14-01 fix added a regex-based DSLR classification to openMVG. This approach is inherently fragile because:
1. Camera naming conventions vary by manufacturer and change over time
2. The regex must be manually maintained for every camera brand
3. The Samsung NX pattern was added with incorrect information ("some were DSLR-style")

The fix introduced two regressions (Samsung NX false positives, Canon xxxD false negatives) because the regex was not thoroughly tested against the full range of camera names in the dataset.

A more robust architectural approach would be:
1. Make openMVG's category a "suggestion" that the merge layer can override based on existing Geizhals data
2. Or use a known-model whitelist instead of a regex pattern heuristic
3. Or simply set openMVG interchangeable-lens cameras to `"unknown"` and let the merge layer classify them based on Geizhals data

However, these architectural changes are significant. For now, fixing the regex bugs is the pragmatic approach.

**Fix:** Fix the immediate regex bugs (Canon `\d+D`, remove Samsung NX) and update the docstring to warn about fragility.

---

### A15-02: Geizhals rangefinder category (Messsucher) creates architectural coupling with incorrect data
**File:** `pixelpitch.py`, lines 740-747 (CATEGORIES)
**Severity:** MEDIUM | **Confidence:** HIGH

The `CATEGORIES` list includes `RANGEFINDER_URL` with `category="rangefinder"`. But Geizhals's "Messsucher" filter returns many cameras that are not rangefinders — they simply have an optical viewfinder. This creates a data-quality issue at the source layer that propagates through the entire pipeline.

The architectural concern is that the code trusts Geizhals's category assignment without validation. When Geizhals says a camera is a "rangefinder" but it's actually a DSLR or mirrorless camera, the merge layer has no mechanism to correct this.

**Fix options:**
1. Post-filter the rangefinder category: only keep cameras from known rangefinder manufacturers
2. Normalize during merge: if a camera already exists in dslr/mirrorless categories from Geizhals, discard the rangefinder duplicate
3. Remove the rangefinder URL from CATEGORIES entirely (simplest but loses actual rangefinders)

Recommended: Option 2 (normalize during merge).

---

## Summary
- NEW findings: 2 (both MEDIUM)
- A15-01: DSLR regex fragility — MEDIUM
- A15-02: Geizhals rangefinder data coupling — MEDIUM
