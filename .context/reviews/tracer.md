# Tracer Review (Cycle 27) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes

## Previous Findings Status

TR26-01 (MPIX_RE gap) and TR26-02 (ValueError in source modules) both fixed in C26.

## New Findings

### TR27-01: PITCH_UM_RE "um" gap traced — potential silent data loss if Geizhals format changes

**File:** `sources/__init__.py` line 66, consumed by `pixelpitch.py` `parse_sensor_field()` line 559
**Severity:** LOW | **Confidence:** HIGH

**Causal trace:**
1. Geizhals HTML contains a pixel pitch field with text like `"5.12 um"` (ASCII-only, no micro sign)
2. `PITCH_UM_RE.search("5.12 um")` returns None (shared pattern does not match "um")
3. `result["pitch"]` stays None
4. In `extract_specs()`, pitch remains None
5. `derive_spec()` computes pitch from area+mpix if both are available — so the camera may still get a computed pitch
6. If size or mpix is also missing, camera shows "unknown" pixel pitch

**Competing hypothesis:** Geizhals uses German/European conventions (µm/μm), not ASCII "um". The GSMArena source is the only one that encounters "um" and has its own local pattern. This trace is a DRY consistency concern, not a current runtime bug.

**Root cause:** The C25-01 centralization of PITCH_UM_RE did not include "um" which GSMArena's local PITCH_RE handles.

---

### TR27-02: parse_existing_csv year=0 traced — displays "0" on website

**File:** `pixelpitch.py` line 336, rendered by `templates/pixelpitch.html` line 103
**Severity:** LOW | **Confidence:** HIGH

**Causal trace:**
1. A CSV file contains `year=0` in a data row
2. `parse_existing_csv()` converts: `year = int(year_str) if year_str else None` → year=0
3. The year is stored in `spec.year` as int(0)
4. Template renders: `{% if spec.spec.year %}` → 0 is truthy in Jinja2 (only None, False, empty strings/collections are falsy)
5. Website displays "0" as the year

**Competing hypothesis:** No current source produces year=0. The `parse_year()` function only matches 19xx/20xx. This requires a corrupted CSV to trigger.

**Root cause:** Missing range validation on the year column in the CSV parser.

---

## Summary

- TR27-01 (LOW): PITCH_UM_RE "um" gap — DRY inconsistency, not a runtime bug
- TR27-02 (LOW): parse_existing_csv year=0 — displays "0" on website
