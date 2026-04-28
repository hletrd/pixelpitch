# Tracer Review (Cycle 38) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes

## Previous Findings Status

TR37-01 (derive_spec area=nan round-trip) fixed by isfinite guard.

## New Findings

### TR38-01: Zero-pitch data flow: `pixel_pitch` returns 0.0 → template renders "0.0 µm" → JS hides row — three layers disagree

**Files:** `pixelpitch.py` (pixel_pitch), `templates/pixelpitch.html` (template + JS)
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. `pixel_pitch(0.0, 0.0)` returns `0.0` (guard: `mpix <= 0 or area <= 0` → returns 0.0)
2. `derive_spec` with `spec.pitch=None, mpix=0.0, area=0.0` → `pitch = pixel_pitch(0.0, 0.0) = 0.0`
3. `write_csv` writes `0.00` for pitch
4. `_safe_float("0.00")` returns `0.0` — round-trip preserves the value
5. Template: `spec.pitch is not none` → True (0.0 is not None) → renders "0.0 µm"
6. JS: `isInvalidData` → `pitch === 0` → returns true → row hidden

Three layers disagree:
- **Python logic**: `pixel_pitch` returns `0.0` as a sentinel for "invalid"
- **Template**: treats `0.0` as a valid number, renders it
- **JS**: treats `0.0` as invalid, hides the row

The correct fix is to align all three layers: either `pixel_pitch` should return `None` (not `0.0`) for invalid inputs, or the template should treat `0.0` as "unknown" (matching JS), or both.

Since changing `pixel_pitch` to return `None` would be a larger refactor with cascading effects on `sorted_by`, `write_csv`, and `merge_camera_data`, the simpler fix is to update the template to render "unknown" for `0.0` pitch/mpix values, aligning with the JS behavior.

**Fix:** Add `spec.pitch != 0.0` check to the template's pitch condition. Add `spec.spec.mpix != 0.0` check to the mpix condition.

---

## Summary

- TR38-01 (MEDIUM): Three-layer disagreement on zero-pitch semantics — Python returns 0.0, template renders it, JS hides it
