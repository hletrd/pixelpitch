# Architect Review (Cycle 40) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## Previous Findings Status

ARCH39-01 fixed. Template uses `> 0` positivity guard for physical quantities.

## New Findings

### ARCH40-01: `pixel_pitch` 0.0 sentinel leaks through `derive_spec` — contract mismatch

**File:** `pixelpitch.py`, lines 178-187 (pixel_pitch), 757-762 (derive_spec)
**Severity:** MEDIUM | **Confidence:** HIGH

`pixel_pitch()` returns 0.0 as a sentinel for "invalid input" rather than raising an exception or returning None. This is a deliberate design choice (documented in the docstring). However, `derive_spec` does not account for this sentinel — it treats the 0.0 return as a valid computed value and stores it in `derived.pitch`.

This violates the boundary contract: `pixel_pitch`'s output domain includes 0.0 as a special value, but `derive_spec`'s consumer contract expects that `derived.pitch` is either None (unknown) or a positive finite value (valid measurement). The 0.0 value is neither.

The downstream consequences cascade:
1. `selectattr('pitch', 'ne', None)` includes 0.0 — wrong table section
2. `write_csv` writes "0.00" — round-trip data loss (parse rejects 0.0)
3. JS `isInvalidData` hides the row (pitch===0)

**Fix:** `derive_spec` should convert 0.0 returns from `pixel_pitch()` to None, establishing the correct contract that `derived.pitch` is either a positive finite value or None.

---

## Summary

- ARCH40-01 (MEDIUM): `pixel_pitch` 0.0 sentinel leaks through `derive_spec` — contract mismatch between computation and data model
