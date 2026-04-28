# Tracer Review (Cycle 43) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## Previous Findings Status

TR42-01 fixed. `merge_camera_data` now checks derived.size consistency after preserving spec.size from existing.

## New Findings

### TR43-01: GSMArena spec.size provenance trace — lookup-table values masquerade as measured data

**Files:** `sources/gsmarena.py`, `pixelpitch.py`
**Severity:** MEDIUM | **Confidence:** HIGH

**Full causal trace of the GSMArena spec.size provenance issue:**

1. **GSMArena fetch:** `_phone_to_spec` at line 146: `size = PHONE_TYPE_SIZE.get(sensor_type)` — e.g., `size = (9.84, 7.40)` for "1/1.3"
2. **Spec creation:** `Spec(name="Samsung S25 Ultra", category="smartphone", type="1/1.3", size=(9.84, 7.40), ...)` — `spec.size` is set from the lookup table
3. **derive_spec:** `spec.size is not None` → `size = spec.size = (9.84, 7.40)` (skips TYPE_SIZE lookup since spec.size already set)
4. **write_csv:** Writes `9.84,7.40` to `camera-data-gsmarena.csv`
5. **render_html:** `_load_per_source_csvs` reads the per-source CSV, parses it back into SpecDerived
6. **merge_camera_data:** `new_spec.spec.size = (9.84, 7.40)` → NOT None → merge condition `new_spec.spec.size is None` is False → Geizhals measured value (9.76, 7.30) is NOT preserved
7. **Result:** The TYPE_SIZE approximation (9.84, 7.40) permanently replaces the measured Geizhals value (9.76, 7.30)
8. **write_csv (final):** Writes the wrong values to the master CSV
9. **Next merge cycle:** The wrong values are now "existing data" — the correct Geizhals value is permanently lost

**Root cause:** GSMArena treats TYPE_SIZE lookup values as if they were measured sensor dimensions. The `spec.size` field has no provenance tracking, so the merge cannot distinguish "measured" from "approximated" values.

**Fix:** GSMArena should set `spec.size = None` and `spec.type = "1/1.3"`. Let `derive_spec` compute `derived.size` from the type lookup. Then `merge_camera_data` will see `spec.size is None` and correctly preserve the measured Geizhals value.

---

## Summary

- TR43-01 (MEDIUM): GSMArena spec.size provenance trace — TYPE_SIZE lookup values masquerade as measured data, preventing merge from preserving accurate Geizhals measurements
