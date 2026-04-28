# Critic Review (Cycle 43) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## Previous Findings Status

CRIT42-01 (merge size inconsistency) fixed. The C42-01 fix adds a derived.size/area/pitch override when spec.size is preserved from existing and derived.size disagrees.

## New Findings

### CRIT43-01: GSMArena/CineD lookup-table sizes in spec.size prevent merge from preserving measured Geizhals values — the real root cause of the "size inconsistency" class of bugs

**File:** `sources/gsmarena.py`, line 146; `sources/cined.py`, lines 94-102; `pixelpitch.py`, lines 456-457
**Severity:** MEDIUM | **Confidence:** HIGH

The C42-01 fix addressed a symptom (derived.size disagrees with spec.size after merge) but missed the deeper root cause: GSMArena and CineD set `spec.size` from lookup tables, which means `spec.size` is not None when it should be. The merge only preserves existing `spec.size` when `new_spec.spec.size is None`. Because GSMArena sets `spec.size` from TYPE_SIZE, the merge sees it as "new data has a size" and never preserves the measured Geizhals value.

This is worse than C42-01 because:
1. **C42-01 scenario:** spec.size=None → merge preserves existing spec.size → derived.size was wrong but fixed by C42-01 patch
2. **CR43-01 scenario:** spec.size=TYPE_SIZE lookup → merge does NOT preserve existing spec.size → both spec.size and derived.size are wrong (TYPE_SIZE approximation), and the measured Geizhals value is permanently lost

The C42-01 fix only helps when `spec.size is None`. For GSMArena/CineD data, `spec.size` is never None (it's set from the lookup table), so the fix doesn't apply.

**Fix:** GSMArena and CineD should set `spec.type` instead of `spec.size` for lookup-table-derived dimensions. This way:
- `spec.size = None` (honest about provenance)
- `derive_spec` computes `derived.size` from `spec.type` using TYPE_SIZE/FORMAT_TO_MM
- `merge_camera_data` sees `spec.size is None` → preserves measured Geizhals `spec.size`
- C42-01 fix handles the derived.size consistency

---

### CRIT43-02: C42-01 fix writes `derived.pitch` from existing unnecessarily — redundant write

**File:** `pixelpitch.py`, line 467
**Severity:** LOW | **Confidence:** HIGH

The C42-01 fix includes `new_spec.pitch = existing_spec.pitch` at line 467. But lines 490-501 already handle derived.pitch consistency: if spec.pitch is preserved from existing (which happens at lines 468-473), the consistency check at lines 498-501 ensures `derived.pitch = spec.pitch`. Writing `derived.pitch` at line 467 is redundant because the consistency check will overwrite it anyway if there's a mismatch.

The fix should only override `derived.size` and `derived.area` (which don't have a later consistency check). Including `derived.pitch` creates a subtle ordering dependency and makes the code harder to reason about.

---

## Summary

- CRIT43-01 (MEDIUM): GSMArena/CineD set spec.size from lookup tables → merge can't preserve measured Geizhals values → deeper root cause of the size inconsistency class of bugs
- CRIT43-02 (LOW): C42-01 fix writes derived.pitch redundantly — consistency check already handles it
