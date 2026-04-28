# Tracer Review (Cycle 33) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

TR32-01 (write_csv 0.0 data loss) fixed in C32. TR31-01 (merge pitch inconsistency) fixed in C31.

## New Findings

### TR33-01: derive_spec 0.0 pitch override — causal trace

**File:** `pixelpitch.py`, line 722
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Source parser (e.g., GSMArena) produces `Spec(pitch=0.0, mpix=33.0, size=(35.9, 23.9))`
2. `derive_spec()` line 722: `if spec.pitch:` → `bool(0.0) is False` → skips to elif
3. elif: `spec.mpix is not None and area is not None` → True
4. `pitch = pixel_pitch(858.61, 33.0)` → 5.12
5. Result: `derived.pitch=5.12` — the original 0.0 measurement is lost
6. `write_csv` (now fixed in C32) would correctly write 5.12 (not 0.0, because derived.pitch was overridden)
7. The 0.0 value is permanently lost in the derivation step, BEFORE write_csv even runs

**Competing hypothesis:** Could spec.pitch=0.0 ever be a valid direct measurement? In the physical domain, no camera has 0 µm pixel pitch. But the data model allows it, and a source parser could produce it (e.g., from a parsing bug or malformed data). The derive_spec docstring says spec.pitch "always takes precedence" — this contract should hold regardless of physical plausibility.

**Fix:** Replace `if spec.pitch:` with `if spec.pitch is not None:` in derive_spec.

---

## Summary

- TR33-01 (LOW-MEDIUM): derive_spec 0.0 pitch override — causal trace confirms data loss before write_csv
