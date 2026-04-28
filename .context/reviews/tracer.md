# Tracer Review (Cycle 32) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-31 fixes, focusing on NEW issues

## Previous Findings Status

TR31-01 (merge pitch inconsistency) fixed in C31.

## New Findings

### TR32-01: write_csv 0.0 data loss — causal trace

**File:** `pixelpitch.py`, lines 824-827
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Source parser produces `Spec(mpix=0.0)` or `derive_spec()` computes `derived.pitch=0.0` via `pixel_pitch()` when `mpix <= 0`
2. `write_csv()` at line 825: `mpix_str = f"{spec.mpix:.1f}" if spec.mpix else ""` — `bool(0.0) is False` → `mpix_str = ""`
3. CSV written with empty mpix column
4. Next build: `parse_existing_csv()` reads empty string → `mpix = None`
5. Data permanently changed from `0.0` to `None`

**Competing hypothesis:** Is 0.0 a valid value for mpix/area/pitch? In the physical domain, no camera has 0 MP or 0 pitch. But in the data model, `0.0` is a valid float that should be preserved by the serialization layer regardless of physical meaning. The serialization should be lossless.

**Fix:** Use `if x is not None` instead of `if x` for all numeric fields in `write_csv`.

---

## Summary

- TR32-01 (LOW-MEDIUM): write_csv 0.0 data loss via falsy checks — causal trace confirmed
