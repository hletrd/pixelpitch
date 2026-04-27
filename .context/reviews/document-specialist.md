# Document Specialist Review (Cycle 15) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository doc/code consistency review after cycles 1-14 fixes

## Previously Fixed (Cycles 1-14) — Confirmed Resolved
- DS11-01: sensors.json schema documented — FIXED
- DS11-02: openMVG listed as data source — FIXED
- DS14-01: openMVG docstring documents category heuristic limitation — FIXED

## New Findings

### DS15-01: openMVG docstring claims DSLR name-pattern heuristic works but regex has bugs
**File:** `sources/openmvg.py`, lines 10-17 (module docstring)
**Severity:** LOW | **Confidence:** HIGH

The docstring (added in C14-05) says: "a name-based check distinguishes DSLRs from mirrorless cameras." While technically true, the name-based check has known bugs (Samsung NX false positives, Canon xxxD false negatives). The docstring should warn about these limitations so developers understand the heuristic is not comprehensive.

Additionally, the DSLR_NAME_RE comment on line 35 says "Samsung NX300 (some were DSLR-style)" which is misleading — all Samsung NX cameras are mirrorless.

**Fix:** Update the docstring to document the known limitations of the DSLR heuristic. Remove or correct the Samsung NX comment.

---

## Summary
- NEW findings: 1 (1 LOW)
- DS15-01: Docstring doesn't warn about DSLR regex bugs — LOW
