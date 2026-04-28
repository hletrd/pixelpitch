# Architect Review (Cycle 35) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

ARCH34-01 (match_sensors percentage comparisons) fixed in C34. All previous architectural concerns remain deferred.

## New Findings

### ARCH35-01: No input validation layer — negative/malformed values propagate silently

**Files:** `pixelpitch.py` (parse_existing_csv, derive_spec, pixel_pitch), `sources/openmvg.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The data pipeline has no systematic input validation. Numeric fields from CSV parsing are accepted as-is, including negative values for sensor dimensions, area, pitch, and mpix. While year has a validation guard (1900-2100), all other numeric fields are unvalidated.

This creates a defense-in-depth gap:
1. `parse_existing_csv` accepts negative values without warning
2. `derive_spec` crashes with `ValueError` when negative area reaches `pixel_pitch`
3. `openmvg.fetch` produces positive mpix from negative pixel dimensions
4. `write_csv` writes negative values without warning
5. Template renders negative values as `-2.0 µm` and `-10.0 MP`

**Architectural recommendation:** Add a validation function that rejects negative values for sensor dimensions, area, pitch, and mpix. Apply it in `parse_existing_csv` and `derive_spec`.

---

### ARCH35-02: `_BOM` literal character undermines the documented defense-in-depth measure

**File:** `sources/__init__.py`, line 90
**Severity:** MEDIUM | **Confidence:** HIGH

The comment explicitly documents that the escape sequence is used instead of the literal to guard against editor stripping. But the implementation uses the literal, defeating the purpose of the documented defense. This is a case where the documented architecture was not implemented.

**Fix:** Replace the literal BOM with the escape sequence.

---

## Summary

- ARCH35-01 (MEDIUM): No input validation layer — negative values propagate and crash
- ARCH35-02 (MEDIUM): `_BOM` literal undermines documented defense-in-depth
