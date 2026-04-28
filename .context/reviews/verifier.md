# Verifier Review (Cycle 41) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28

## V41-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C40 fixes verified working. No regressions.

## V41-02: `derive_spec` preserves invalid direct `spec.pitch` values — verified by execution

**File:** `pixelpitch.py`, lines 759-760
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence (verified by execution):**

| Input | spec.pitch | derived.pitch | selectattr includes? | Correct? |
|-------|-----------|---------------|---------------------|----------|
| pitch=0.0 (direct) | 0.0 | 0.0 | Yes (0.0!=None) | NO — should be None |
| pitch=-1.0 (direct) | -1.0 | -1.0 | Yes (-1.0!=None) | NO — should be None |
| pitch=nan (direct) | nan | nan | No (nan comparison) | Partially — still bad data |
| pitch=None, mpix=0.0 (computed) | None | None | No | YES (C40 fix working) |
| pitch=5.12 (direct) | 5.12 | 5.12 | Yes | YES |

The C40 fix only addressed the computed path. The direct path (when `spec.pitch is not None` but invalid) is unguarded.

**Fix:** Validate `spec.pitch` in `derive_spec`: reject non-finite or non-positive values by converting to None.

---

## V41-03: `write_csv` writes 0.0 and negative mpix/pitch — verified by execution

**File:** `pixelpitch.py`, lines 866-868
**Severity:** LOW | **Confidence:** HIGH

**Evidence (verified by execution):**

| Input | CSV output | Re-parsed by parse_existing_csv | Data loss? |
|-------|-----------|-------------------------------|-------------|
| mpix=0.0 | "0.0" | None (rejected by <=0) | YES |
| mpix=-5.0 | "-5.0" | None (rejected by <=0) | YES |
| pitch=0.0 | "0.00" | None (rejected by <=0) | YES |
| pitch=-1.0 | "-1.00" | None (rejected by <=0) | YES |
| mpix=inf | "" (isfinite blocks) | None | NO — correct |
| mpix=33.0 | "33.0" | 33.0 | NO |

`isfinite(0.0)` returns True and `isfinite(-1.0)` returns True, so the isfinite-only guard is insufficient for physically-meaningful fields.

**Fix:** Replace `isfinite` with positivity checks (`> 0`) in `write_csv` for mpix, pitch, and area fields.

---

## V41-04: `merge_camera_data` preserves spec.pitch=0.0 from existing — verified by execution

**File:** `pixelpitch.py`, lines 449, 471-473
**Severity:** LOW | **Confidence:** MEDIUM

**Evidence:** Created a new spec with `pitch=None` and an existing spec with `pitch=0.0`. After merge, `merged[0].spec.pitch = 0.0` and `merged[0].pitch = 0.0`. The 0.0 sentinel is re-introduced through the existing data preservation path.

---

## Summary

- V41-01: Gate tests pass
- V41-02 (MEDIUM): `derive_spec` preserves invalid direct spec.pitch values (0.0, negative, NaN)
- V41-03 (LOW): `write_csv` writes 0.0/negative mpix/pitch — isfinite guard insufficient
- V41-04 (LOW): `merge_camera_data` preserves spec.pitch=0.0 from existing data
