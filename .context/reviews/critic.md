# Critic Review (Cycle 32) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-31 fixes, focusing on NEW issues

## Previous Findings Status

C31-01 and C31-02 both implemented. All previous fixes stable.

## New Findings

### CRIT32-01: write_csv falsy checks lose 0.0 values on CSV round-trip

**File:** `pixelpitch.py`, lines 824-827
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

This is a correctness issue that subsumes CR32-01. The `write_csv` function uses Python truthiness (`if x`) for float fields (area, mpix, pitch) and year. For float values, `0.0` is falsy and would be written as empty string, then read back as `None` by `parse_existing_csv`.

The deeper issue is that this is an inconsistency: the data model (Spec/SpecDerived) allows `0.0` as a valid value, but the serialization layer (write_csv/parse_existing_csv) treats `0.0` as equivalent to `None`. This violates the principle that serialization should be lossless.

**Fix:** Use `if x is not None` instead of `if x` for all float and integer fields in `write_csv`.

---

## Summary

- CRIT32-01 (LOW-MEDIUM): write_csv falsy checks create asymmetry between data model and CSV serialization for 0.0 values
