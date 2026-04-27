# Document Specialist Review (Cycle 11) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository doc/code consistency review after cycles 1-10 fixes

## Previously Fixed (Cycles 1-10) — Confirmed Resolved
All previous doc fixes remain intact. About page CSV link `./` prefix fixed in cycle 10.

## New Findings

### DS11-01: `sensors.json` is loaded but never documented — unknown schema
**File:** `pixelpitch.py`, lines 172-178; `sensors.json`
**Severity:** LOW | **Confidence:** HIGH

`load_sensors_database()` loads `sensors.json` which is used by `match_sensors`. The JSON schema (what fields are expected) is not documented anywhere. From code inspection, the expected structure is:
```json
{
  "sensor_name": {
    "sensor_width_mm": float,
    "sensor_height_mm": float,
    "megapixels": [float, ...]
  }
}
```
But this is implicit. If someone adds a sensor with different field names, `match_sensors` would silently skip it.

**Fix:** Add a docstring or comment documenting the expected schema of sensors.json.

---

### DS11-02: `about.html` page doesn't mention openMVG as a data source
**File:** `templates/about.html`; `templates/pixelpitch.html`
**Severity:** LOW | **Confidence:** MEDIUM

The about page lists geizhals.eu, Imaging Resource, Apotelyt, GSMArena, and CineD as sources. It does NOT mention openMVG/CameraSensorSizeDatabase, which is a primary data source (bulk CSV with thousands of cameras). The `pixelpitch.html` alert text also omits openMVG.

**Fix:** Add openMVG to the source list on the about page and in the pixelpitch.html alert text.

---

## Summary
- NEW findings: 2 (both LOW)
- DS11-01: sensors.json schema undocumented — LOW
- DS11-02: openMVG not listed as a data source on about page — LOW
