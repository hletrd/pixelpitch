# Critic Review (Cycle 16) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
All previous fixes remain intact. No regressions detected.

## New Findings

### CR16-01: `sensor_size_from_type` is a crash waiting to happen — defensive programming gap
**File:** `pixelpitch.py`, lines 152-165
**Severity:** MEDIUM | **Confidence:** HIGH

Same as C16-01/S16-01. The function performs arithmetic on user-derived data (sensor type strings from HTML parsing) without any error handling. This violates the principle of failing gracefully. Any source HTML containing an unusual sensor type format would crash the entire pipeline. The fix is simple: wrap the computation in try/except and return None on failure.

---

### CR16-02: `merge_camera_data` lacks self-dedup for new_specs — a fundamental correctness issue
**File:** `pixelpitch.py`, lines 349-407
**Severity:** MEDIUM | **Confidence:** HIGH

Same as C16-02. The merge function deduplicates against existing data but not among its own inputs. This is a fundamental design oversight: when the same camera appears in multiple sources with the same category, the user sees duplicate rows on the All Cameras page. The function should dedup among new_specs before or during the merge loop.

---

### CR16-03: Pentax DSLR regex is incomplete — multiple model families missed
**File:** `sources/openmvg.py`, line 47
**Severity:** LOW | **Confidence:** HIGH

Same as C16-03. The `Pentax\s+K[-\s]\d` pattern is overly restrictive. It misses at least 10 Pentax DSLR models that lack the hyphen or have letter suffixes. The fix is to broaden the regex to `Pentax\s+K[-\s]?\d+\w?` or similar.

---

### CR16-04: digicamdb alias creates silent duplication risk
**File:** `sources/digicamdb.py`; `pixelpitch.py`, line 985
**Severity:** LOW | **Confidence:** HIGH

Same as C16-04. The digicamdb source is a pure alias for openMVG. Having both in SOURCE_REGISTRY means a manual `python pixelpitch.py source digicamdb` creates an identical CSV, compounding the merge dedup issue.

---

## Summary
- NEW findings: 4 (2 MEDIUM, 2 LOW)
- CR16-01: sensor_size_from_type crash — MEDIUM
- CR16-02: merge_camera_data self-dedup missing — MEDIUM
- CR16-03: Pentax regex incomplete — LOW
- CR16-04: digicamdb alias duplication — LOW
