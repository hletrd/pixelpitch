# Document Specialist Review (Cycle 14) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository doc/code consistency review after cycles 1-13 fixes

## Previously Fixed (Cycles 1-12) — Confirmed Resolved
- DS11-01: sensors.json schema documented — FIXED
- DS11-02: openMVG listed as data source — FIXED

## New Findings

### DS14-01: openMVG module docstring says "category" but heuristic is incomplete
**File:** `sources/openmvg.py`, module docstring (lines 1-10)
**Severity:** LOW | **Confidence:** HIGH

The module docstring says "Coverage skews toward 2010s consumer compacts" but doesn't mention that the category heuristic (`size[0] >= 20 → mirrorless`) misclassifies DSLRs. The docstring should document this known limitation so developers understand why duplicate entries may appear for DSLRs.

**Fix:** Add a note to the docstring about the category heuristic limitation.

---

## Summary
- NEW findings: 1 (1 LOW)
- DS14-01: openMVG docstring doesn't document category heuristic limitation — LOW
