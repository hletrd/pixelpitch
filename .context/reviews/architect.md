# Architect Review (Cycle 14) — Architectural/Design Risks, Coupling, Layering

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository architecture review after cycles 1-13 fixes

## Previously Noted (Deferred, Still Valid)
- F32: `pixelpitch.py` is a ~1061-line monolith — DEFERRED
- F31: No source Protocol/base class — DEFERRED
- A5-02: Template description blocks DRY violation — DEFERRED

## Previously Fixed (Cycles 1-13)
- A11-01: create_camera_key year coupling — FIXED
- A12-01: Inconsistent field stripping in CSV parser — FIXED
- A12-02: Sony URL parsing assumes specific URL structure — FIXED
- A13-01: load_csv and _load_per_source_csvs inconsistent error handling — FIXED

## New Findings

### A14-01: openMVG category heuristic is architecturally misaligned with the merge/dedup system
**File:** `sources/openmvg.py`, lines 63-69; `pixelpitch.py`, line 340 (`create_camera_key`)
**Severity:** MEDIUM | **Confidence:** HIGH

The architecture relies on `create_camera_key(name, category)` for deduplication across sources. But openMVG has no body-type information and uses a sensor-width heuristic (`size[0] >= 20 → mirrorless`) that systematically misclassifies DSLRs. This creates a category mismatch with Geizhals data for the same camera, producing different merge keys and visible duplicates.

This is a layering concern: the openMVG source module makes a category decision without the context of other sources, and the merge layer trusts the source's category decision. The architectural fix would be either:
1. Make openMVG's category a "suggestion" that the merge layer can override based on existing data, OR
2. Have openMVG not set category at all (use a sentinel like `"unknown"`) and let the merge layer classify based on existing entries, OR
3. Add a name-based DSLR heuristic to openMVG (pragmatic but fragile)

**Concrete failure:** Canon EOS 5D appears on the All Cameras page twice: once as "mirrorless" (openMVG) and once as "dslr" (Geizhals).

---

### A14-02: CSV parsing layer has no BOM defense — BOM bypasses schema detection
**File:** `pixelpitch.py`, lines 250-330 (`parse_existing_csv`); `load_csv` (lines 238-247)
**Severity:** MEDIUM | **Confidence:** HIGH

The CSV parsing layer assumes the first header field is exactly `"id"` to detect schema version. A UTF-8 BOM character prepended to the CSV content makes `header[0]` equal to `"﻿id"`, breaking schema detection and causing complete parse failure. This is a layering issue: the I/O layer (`load_csv`, `_load_per_source_csvs`) reads raw bytes without BOM handling, and the parsing layer (`parse_existing_csv`) has no BOM stripping.

The fix should be at the parsing layer entry point (`parse_existing_csv`) since it's the single point where all CSV content flows through, regardless of source.

---

## Summary
- NEW findings: 2 (both MEDIUM)
- A14-01: openMVG category heuristic misaligned with merge system — MEDIUM
- A14-02: No BOM defense in CSV parsing layer — MEDIUM
