# Architect Review (Cycle 43) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## Previous Findings Status

ARCH42-01 (Spec/SpecDerived field duplication) addressed pragmatically with the C42-01 fix. ARCH42-02 (circular import gsmarena<->pixelpitch) deferred.

## New Findings

### ARCH43-01: spec.size provenance ambiguity — GSMArena/CineD set spec.size from lookup tables, breaking merge's "None means unknown" contract

**File:** `sources/gsmarena.py`, line 146; `sources/cined.py`, lines 94-102; `pixelpitch.py`, line 456
**Severity:** MEDIUM | **Confidence:** HIGH

The `merge_camera_data` function uses a simple contract: `None` means "unknown, preserve from existing if available". This works when sources correctly mark unknown fields as `None`. However, GSMArena and CineD set `spec.size` from lookup tables (TYPE_SIZE and FORMAT_TO_MM), treating approximate values as known.

This violates the "None means unknown" contract. The merge sees `spec.size = (9.84, 7.40)` and treats it as a measured value, never preserving the Geizhals measured value from existing data.

This is an architectural issue with the Spec data model: there is no distinction between "measured" and "approximated" values. The fix options are:

1. **Simple: Sources should not set spec.size from lookup tables.** Use `spec.type` for type-derived sizes and `spec.size` only for measured dimensions. This preserves the "None means unknown" contract.
2. **Moderate: Add a provenance field to Spec.** A `size_provenance: Optional[str]` field ("measured", "type_lookup", "format_lookup") that merge can use to make informed decisions.
3. **Large: Refactor Spec to use a "Measurement" type with uncertainty.** Overkill for this codebase.

Option 1 is the most pragmatic. It requires GSMArena to stop setting `spec.size` from `PHONE_TYPE_SIZE` and CineD to stop setting it from `FORMAT_TO_MM`. Both should set `spec.type` instead and let `derive_spec` compute `derived.size`.

**Impact on existing data:**
- The per-source CSVs (`camera-data-gsmarena.csv`, `camera-data-cined.csv`) currently store `spec.size` in the sensor_width_mm/sensor_height_mm columns. After the fix, GSMArena cameras will have empty width/height columns and a type column (e.g., "1/1.3"). The `derive_spec` pipeline will compute `derived.size` from the type, so the displayed values will be the same for GSMArena-only cameras.
- For cameras that also exist in Geizhals data, the merge will now correctly preserve the measured Geizhals size instead of overriding it with the TYPE_SIZE approximation.

---

## Summary

- ARCH43-01 (MEDIUM): spec.size provenance ambiguity — GSMArena/CineD set spec.size from lookup tables, breaking merge's "None means unknown" contract
