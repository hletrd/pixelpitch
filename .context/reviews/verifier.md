# Verifier Review (Cycle 40) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28

## V40-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C39 fixes verified working. No regressions.

## V40-02: `derive_spec` produces pitch=0.0 for computed path — selectattr misclassification

**File:** `pixelpitch.py`, lines 757-762
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence (verified by execution):**

| Input | spec.pitch | spec.mpix | derived.pitch | selectattr includes? | Correct section |
|-------|-----------|-----------|---------------|---------------------|-----------------|
| size=(5.0,3.7), mpix=0.0, pitch=None | None | 0.0 | 0.0 | Yes (0.0!=None) | Without pitch |
| size=(5.0,3.7), mpix=-1.0, pitch=None | None | -1.0 | 0.0 | Yes (0.0!=None) | Without pitch |
| size=(35.9,23.9), mpix=33.0, pitch=0.0 | 0.0 | 33.0 | 0.0 | Yes (0.0!=None) | Without pitch |
| size=(35.9,23.9), mpix=33.0, pitch=None | None | 33.0 | ~5.12 | Yes | With pitch |
| size=None, mpix=None, pitch=None | None | None | None | No | Without pitch |

The `pixel_pitch` function returns 0.0 as a sentinel for invalid inputs. `derive_spec` propagates this 0.0 without converting to None, causing `selectattr('pitch', 'ne', None)` to misclassify cameras with invalid computed pitch into the "with pitch" table.

**Fix:** In `derive_spec`, convert `pixel_pitch()` result of 0.0 to None.

## V40-03: `write_csv` outputs inf/nan strings for non-finite float values

**File:** `pixelpitch.py`, lines 839-872
**Severity:** LOW | **Confidence:** HIGH

**Evidence:**
- `write_csv` with `spec.mpix=float('inf')` produces CSV row containing `inf`
- `write_csv` with `spec.mpix=float('nan')` produces CSV row containing `nan`
- `write_csv` with `derived.area=float('inf')` produces CSV row containing `inf`
- `parse_existing_csv` correctly rejects these on re-read via `_safe_float`

While the round-trip is safe (parse rejects them), other CSV consumers may not handle inf/nan.

**Fix:** Add `isfinite` checks before formatting float fields in `write_csv`.

---

## Summary

- V40-01: Gate tests pass
- V40-02 (MEDIUM): `derive_spec` propagates 0.0 sentinel — selectattr misclassification
- V40-03 (LOW): `write_csv` outputs inf/nan without validation
