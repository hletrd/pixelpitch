# Critic Review (Cycle 40) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## Previous Findings Status

CRIT39-01 fixed. Template uses `> 0` guard. But the fix was narrow — it only addressed the template rendering, not the upstream data flow.

## New Findings

### CRIT40-01: C39 fix addressed rendering but not the upstream data flow — `derive_spec` still produces 0.0 sentinel pitch

**File:** `pixelpitch.py`, lines 757-762; `templates/pixelpitch.html`, line 183
**Severity:** MEDIUM | **Confidence:** HIGH

The C39 fix changed the template guard from `!= 0.0` to `> 0`. This correctly renders "unknown" for 0.0 pitch/mpix values. However, the root cause was never addressed: `derive_spec` still produces `derived.pitch = 0.0` when `pixel_pitch()` returns its 0.0 sentinel. This 0.0 value then:

1. Passes through `selectattr('pitch', 'ne', None)` — the camera appears in the "with pitch" table instead of "without pitch"
2. Flows through `write_csv` — the CSV contains "0.00" for pitch
3. Gets picked up by `parse_existing_csv` — which correctly rejects 0.0 pitch (positivity check)
4. This creates a data loss on CSV round-trip: pitch=0.0 becomes pitch=None

The correct fix is to stop the 0.0 sentinel at its source. `derive_spec` should convert `pixel_pitch()`'s 0.0 return to None. The `pixel_pitch` function's 0.0 sentinel is an internal implementation detail that should not leak past `derive_spec`.

**Fix:** In `derive_spec`, after computing pitch from `pixel_pitch()`, convert 0.0 to None:
```python
pitch = pixel_pitch(area, spec.mpix)
if pitch == 0.0:
    pitch = None
```

This is the correct architectural fix — it eliminates the 0.0 sentinel at the boundary between the computation layer and the data model, rather than patching each downstream consumer.

---

## Summary

- CRIT40-01 (MEDIUM): C39 fix was narrow — `derive_spec` still produces 0.0 sentinel pitch that leaks through selectattr and write_csv
