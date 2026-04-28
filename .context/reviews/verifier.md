# Verifier Review (Cycle 43) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28

## V43-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C42 fixes verified working. No regressions.

## V43-02: GSMArena sets spec.size from TYPE_SIZE — verified that merge cannot preserve measured Geizhals values

**File:** `sources/gsmarena.py`, line 146; `pixelpitch.py`, lines 456-457
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence (verified by code trace):**

1. GSMArena `_phone_to_spec` at line 146: `size = PHONE_TYPE_SIZE.get(sensor_type) if sensor_type else None`
2. Returns `Spec(..., size=size, ...)` — so `spec.size = (9.84, 7.40)` for a "1/1.3" type phone
3. In `merge_camera_data` at line 456: `if new_spec.spec.size is None and existing_spec.spec.size is not None:` → condition is False because `new_spec.spec.size = (9.84, 7.40)` from GSMArena
4. Geizhals measured value (e.g., (9.76, 7.30)) is never preserved

**Concrete scenario (simulated):**
```python
import pixelpitch as pp
from models import Spec

# Geizhals: measured size
existing = pp.derive_spec(Spec(name="Samsung S25 Ultra", category="smartphone",
                                type="1/1.3", size=(9.76, 7.30), pitch=None, mpix=200.0, year=2025))
existing.id = 0

# GSMArena: size from TYPE_SIZE lookup (NOT None)
new = pp.derive_spec(Spec(name="Samsung S25 Ultra", category="smartphone",
                           type="1/1.3", size=(9.84, 7.40), pitch=None, mpix=200.0, year=2025))

merged = pp.merge_camera_data([new], [existing])
m = merged[0]
# m.spec.size = (9.84, 7.40)  ← TYPE_SIZE lookup (WRONG — should be (9.76, 7.30) from Geizhals)
# m.size = (9.84, 7.40)       ← TYPE_SIZE lookup (WRONG)
# Measured Geizhals value (9.76, 7.30) is LOST
```

The C42-01 fix only applies when `spec.size is None`. Here `spec.size` is not None, so the fix doesn't trigger. This is a variant of the size inconsistency bug that the C42-01 fix did not cover.

**Fix:** GSMArena should not set `spec.size` from the lookup table. It should set `spec.type` and leave `spec.size = None`.

---

## V43-03: C42-01 fix writes derived.pitch redundantly — verified by code trace

**File:** `pixelpitch.py`, line 467
**Severity:** LOW | **Confidence:** HIGH

**Evidence:** In `merge_camera_data`, when the C42-01 consistency fix fires:

1. Line 467: `new_spec.pitch = existing_spec.pitch` — writes derived.pitch from existing
2. Line 468-473: `new_spec.spec.pitch = existing_spec.spec.pitch` — preserves spec.pitch from existing
3. Lines 498-501: if `spec.pitch != derived.pitch`, override `derived.pitch = spec.pitch`

If `existing_spec.pitch != existing_spec.spec.pitch` (e.g., existing derived.pitch was computed from area+mpix while spec.pitch was a direct measurement), step 3 will overwrite the value from step 1. The write at step 1 is redundant.

This does NOT cause a correctness issue — the final value is correct (spec.pitch is authoritative). But it's misleading code.

---

## Summary

- V43-01: Gate tests pass
- V43-02 (MEDIUM): GSMArena sets spec.size from TYPE_SIZE → merge can't preserve measured Geizhals values — verified by code trace showing condition is False
- V43-03 (LOW): C42-01 fix writes derived.pitch redundantly — verified by code trace showing step 3 overwrites step 1
