# Debugger Review (Cycle 43) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## Previous Findings Status

DBG42-01 and DBG42-02 fixed. `merge_camera_data` now has derived.size consistency check. CLI `--limit` has try/except.

## New Findings

### DBG43-01: GSMArena spec.size from TYPE_SIZE lookup silently overwrites measured Geizhals values in merge

**File:** `sources/gsmarena.py`, line 146; `pixelpitch.py`, line 456
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode (GSMArena phone with type lookup, existing Geizhals measured size):**

1. GSMArena provides `Spec(name="Phone X", type="1/1.3", size=(9.84, 7.40), mpix=200.0)` — size from TYPE_SIZE
2. Geizhals provides measured `spec.size=(9.76, 7.30)` (the actual measured value from the product page)
3. Merge: `new_spec.spec.size is NOT None` (it's (9.84, 7.40) from TYPE_SIZE) → condition `new_spec.spec.size is None` is False → Geizhals measured value NOT preserved
4. Result: spec.size=(9.84, 7.40) from TYPE_SIZE lookup, derived.size=(9.84, 7.40) — the measured Geizhals value (9.76, 7.30) is permanently lost
5. CSV stores wrong values, template shows wrong dimensions
6. Next merge: wrong values are "existing data" — correct Geizhals value permanently gone

**Reproduction:**
```python
import pixelpitch as pp
from models import Spec

existing = pp.derive_spec(Spec(name="Samsung S25 Ultra", category="smartphone",
                                type="1/1.3", size=(9.76, 7.30), pitch=None, mpix=200.0, year=2025))
existing.id = 0

new = pp.derive_spec(Spec(name="Samsung S25 Ultra", category="smartphone",
                           type="1/1.3", size=(9.84, 7.40), pitch=None, mpix=200.0, year=2025))

merged = pp.merge_camera_data([new], [existing])
# merged.spec.size = (9.84, 7.40)  ← WRONG (TYPE_SIZE lookup)
# merged.size = (9.84, 7.40)       ← WRONG
# Measured Geizhals value (9.76, 7.30) is LOST
```

**Fix:** GSMArena should not set `spec.size` from TYPE_SIZE. Set `spec.type` and leave `spec.size = None`.

---

### DBG43-02: Same issue for CineD FORMAT_TO_MM lookup sizes

**File:** `sources/cined.py`, lines 94-102
**Severity:** MEDIUM | **Confidence:** MEDIUM

Same failure mode as DBG43-01 but for CineD's FORMAT_TO_MM lookup. CineD sets `spec.size` from format class (e.g., "Super 35" → (24.89, 18.66)). If Geizhals has a measured size for a cinema camera that's slightly different, the merge will not preserve it.

---

### DBG43-03: C42-01 fix writes derived.pitch redundantly — wasted write

**File:** `pixelpitch.py`, line 467
**Severity:** LOW | **Confidence:** HIGH

The C42-01 fix includes `new_spec.pitch = existing_spec.pitch` at line 467. But lines 498-501 already handle derived.pitch consistency by checking `spec.pitch != derived.pitch`. Since spec.pitch is also preserved from existing (lines 468-473), the consistency check will overwrite derived.pitch if there's a mismatch. The write at line 467 is redundant.

Not a correctness bug — the final value is correct. But it's a code clarity issue that could confuse future maintainers.

---

## Summary

- DBG43-01 (MEDIUM): GSMArena spec.size from TYPE_SIZE silently overwrites measured Geizhals values in merge
- DBG43-02 (MEDIUM): Same for CineD FORMAT_TO_MM lookup sizes
- DBG43-03 (LOW): C42-01 fix writes derived.pitch redundantly
