# Verifier Review — Cycle 46

**Date:** 2026-04-28
**Reviewer:** verifier

## Previous Findings Status

V45-01 (GSMArena decimal MP regex) — COMPLETED. Fix verified working.

## Verification Results

### V46-VERIFY-01: matched_sensors data loss in merge_camera_data

Tested with direct Python execution:

```python
from models import Spec, SpecDerived
from pixelpitch import merge_camera_data, derive_spec

# Existing: Canon R5 with matched_sensors from CSV
spec1 = Spec(name='Canon EOS R5', category='dslr', type=None, size=(36.0, 24.0), pitch=4.39, mpix=45.0, year=2020)
existing = SpecDerived(spec=spec1, size=(36.0, 24.0), area=864.0, pitch=4.39, matched_sensors=['IMX309', 'IMX366', 'IMX609'], id=0)

# New: same camera but derive_spec with empty sensors_db
new = derive_spec(spec1, {})
# new.matched_sensors = []

merged = merge_camera_data([new], [existing])
# merged[0].matched_sensors = []  <-- DATA LOSS
```

Result: `matched_sensors=['IMX309', 'IMX366', 'IMX609']` is overwritten by `[]`. **BUG CONFIRMED**.

Also verified: when `derive_spec` is called WITH a populated `sensors_db`, the new data has correct `matched_sensors` and no data loss occurs. The bug only manifests when `sensors_db` is empty or not provided.

### V46-VERIFY-02: LENS_RE dead code in gsmarena.py

Searched entire codebase: `LENS_RE` is defined at `sources/gsmarena.py:45` but never referenced in any file. **DEAD CODE CONFIRMED**.

## New Findings

### V46-01: matched_sensors not preserved in merge_camera_data — verified data loss

**File:** `pixelpitch.py`, merge_camera_data
**Severity:** MEDIUM | **Confidence:** HIGH (verified by live execution)

Verified that `merge_camera_data` loses `matched_sensors` from existing data when new data has `matched_sensors=[]` (from `derive_spec` with empty `sensors_db`). The merge code preserves fields when `new.X is None`, but `[]` is not `None`, so preservation is bypassed.

**Fix:** Return `matched_sensors=None` from `derive_spec` when `sensors_db` is not provided, and add preservation logic for `matched_sensors` in `merge_camera_data`.

---

## Summary

- V46-01 (MEDIUM): matched_sensors not preserved in merge_camera_data — verified data loss
