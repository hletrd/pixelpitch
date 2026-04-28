# Debugger Review (Cycle 22) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## D22-01: `elif` year-change branch unreachable — C21-01 regression

**File:** `pixelpitch.py`, lines 428-437
**Severity:** MEDIUM | **Confidence:** HIGH (static analysis + traced execution)

The C21-01 fix inserted SpecDerived field preservation (lines 423-428) between the Spec field preservation block and the year-change `elif`. The `elif` was originally attached to the `if new_spec.spec.year is None` guard (line 418). After the insertion, it is now attached to `if new_spec.pitch is None and existing_spec.pitch is not None` (line 428).

**Execution trace:**
1. `new_spec.spec.year is None and existing_spec.spec.year is not None` → `new_spec.spec.year = existing_spec.spec.year` (line 418)
2. `new_spec.size is None and existing_spec.size is not None` → `new_spec.size = existing_spec.size` (line 424)
3. `new_spec.area is None and existing_spec.area is not None` → `new_spec.area = existing_spec.area` (line 426)
4. `new_spec.pitch is None and existing_spec.pitch is not None` → `new_spec.pitch = existing_spec.pitch` (line 428)
5. The `elif` on line 429 is evaluated ONLY when step 4's condition is False
6. But the year-change message should fire when BOTH years are non-None and differ — regardless of pitch

**Failure mode:** When a camera's pitch is preserved from existing data (step 4 condition True), the `elif` is never evaluated. The year change IS correctly applied (new data takes precedence on line 418 when year is not None), but the diagnostic log is suppressed. This means the "Year changed for ..." message is partially silenced.

**Impact:** Not a data correctness bug — years are still correct. But it's a diagnostic regression that makes it harder to track year changes in CI logs.

---

## Summary

- D22-01 (MEDIUM): `elif` year-change branch unreachable — diagnostic regression from C21-01
