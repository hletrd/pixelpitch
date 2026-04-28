# Tracer Review (Cycle 35) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

TR34-01 (match_sensors ZeroDivisionError) fixed in C34. Verified.

## New Findings

### TR35-01: `derive_spec` crashes with ValueError via `pixel_pitch` on negative area — causal trace

**File:** `pixelpitch.py`, line 725
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Source parser (e.g., `openmvg.fetch`) produces `Spec(size=(-5.0, 3.7), pitch=None, mpix=10.0)` — negative width from malformed data
2. `derive_spec` computes `area = -5.0 * 3.7 = -18.5` — negative area
3. Since `spec.pitch is None`, calls `pixel_pitch(-18.5, 10.0)`
4. `sqrt(-18.5 / 10_000_000)` raises `ValueError: expected a nonnegative input`
5. Exception is NOT caught — crashes the render pipeline

**Competing hypothesis:** Could negative dimensions actually reach `derive_spec`?
- `openmvg.fetch` guards `sw > 0 and sh > 0` for mm dimensions, but NOT for pixel dimensions
- `parse_existing_csv` accepts any float for width/height without validation
- `merge_camera_data` preserves existing size values which could be negative from corrupted CSV

**Fix:** Add `area <= 0` guard in `pixel_pitch` or `derive_spec` before the sqrt call.

---

### TR35-02: `_BOM` literal character — silent failure trace

**File:** `sources/__init__.py`, line 90
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. An editor or CI pipeline normalizes UTF-8 source files
2. The invisible BOM literal character on line 90 is silently stripped
3. `_BOM` becomes `''` (empty string)
4. `strip_bom` checks `text[0] == _BOM` which is `text[0] == ''` — always False
5. BOM-prefixed CSVs are not stripped
6. `openmvg.fetch`: DictReader sees mangled header `"﻿CameraMaker"` instead of `"CameraMaker"`
7. `KeyError` on every row — 0 records returned
8. The build silently produces empty data with no error message

**Fix:** Replace the literal BOM with the escape sequence `﻿` so it survives editor normalization.

---

## Summary

- TR35-01 (MEDIUM): `derive_spec` crashes with ValueError on negative area — crash path traced
- TR35-02 (MEDIUM): `_BOM` literal character — silent failure traced through CSV parsing
