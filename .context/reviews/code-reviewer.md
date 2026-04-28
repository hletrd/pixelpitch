# Code Review (Cycle 20) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-19 fixes, focusing on NEW issues

## C20-01: `pixel_pitch()` crashes with ZeroDivisionError when mpix=0.0
**File:** `pixelpitch.py`, line 182
**Severity:** MEDIUM | **Confidence:** HIGH

The `pixel_pitch()` function divides by `mpix * 10**6` without checking for zero. When `mpix=0.0`, this crashes with `ZeroDivisionError`. While 0 MP cameras don't exist in practice, the function should handle the edge case gracefully rather than crashing the entire pipeline.

**Concrete failure scenario:** If any source returns a spec with `mpix=0.0` and `pitch=None`, the `derive_spec()` function will call `pixel_pitch(area, 0.0)` and crash, halting the entire render.

**Fix:** Add a guard in `pixel_pitch()`:
```python
def pixel_pitch(area: float, mpix: float) -> float:
    if mpix <= 0:
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

---

## C20-02: `pixel_pitch()` crashes with ValueError when mpix < 0
**File:** `pixelpitch.py`, line 182
**Severity:** LOW | **Confidence:** HIGH

If `mpix` is negative, `sqrt(area / (mpix * 10**6))` computes sqrt of a negative number, raising `ValueError`. Same for negative area. This is a defensive programming issue — while negative values shouldn't occur in practice, the math function should be robust.

**Fix:** Same guard as C20-01 — `if mpix <= 0` covers both cases.

---

## C20-03: Sony FX series cameras misnamed by `_parse_camera_name`
**File:** `sources/imaging_resource.py`, lines 156-164
**Severity:** MEDIUM | **Confidence:** HIGH

The `_parse_camera_name()` function applies `.title()` to URL slugs, which converts "fx3" to "Fx3". The function has a special-case replacement for "Sony Zv " -> "Sony ZV-" but no equivalent for Sony FX series. This causes Sony cinema cameras like FX3, FX6, FX30 to appear as "Sony Fx3", "Sony Fx6", "Sony Fx30" — incorrect capitalization.

**Concrete failure scenario:** A user searches for "Sony FX3" on the All Cameras page and doesn't find it because it's listed as "Sony Fx3".

**Fix:** Add FX-series normalization similar to ZV. Use a regex for robustness:
```python
cleaned = re.sub(r'\bFx(\d)', r'FX\1', cleaned)
```

---

## C20-04: `merge_camera_data` does not preserve type/size/pitch from existing data when new data has None
**File:** `pixelpitch.py`, lines 397-412
**Severity:** LOW | **Confidence:** HIGH

The merge function has explicit logic to preserve `year` from existing data when the new data has `year=None`. However, it does NOT preserve `type`, `size`, or `pitch` in the same way. When a new spec has `type=None` but the existing spec had `type='1/2.3'`, the type is lost.

**Concrete failure scenario:** Camera "Test Cam" has type='1/2.3' from a previous Geizhals fetch. In the next run, Geizhals doesn't provide the type (field missing), but openMVG has the same camera without a type. The merge uses the new spec, losing the type.

**Fix:** Add field-level merge logic similar to year:
```python
if new_spec.spec.type is None and existing_spec.spec.type is not None:
    new_spec.spec.type = existing_spec.spec.type
if new_spec.spec.size is None and existing_spec.spec.size is not None:
    new_spec.spec.size = existing_spec.spec.size
if new_spec.spec.pitch is None and existing_spec.spec.pitch is not None:
    new_spec.spec.pitch = existing_spec.spec.pitch
```

---

## C20-05: 259 duplicate (name, category) pairs in dist/camera-data.csv from stale data
**File:** `dist/camera-data.csv`
**Severity:** LOW | **Confidence:** HIGH (stale artifact, not a code bug)

The current CSV has 259 duplicate (name, category) pairs. These are artifacts from previous CI runs before the merge dedup logic was fixed. The current `merge_camera_data` code correctly deduplicates — running `merge_camera_data([], existing)` reduces 1742 records to 1472 with 0 duplicates. The next CI run will clean this up.

No code fix needed — this is a data artifact that will be resolved on the next deployment.

---

## Summary

- C20-01 (MEDIUM): `pixel_pitch()` ZeroDivisionError when mpix=0
- C20-02 (LOW): `pixel_pitch()` ValueError when mpix < 0 (same fix as C20-01)
- C20-03 (MEDIUM): Sony FX cameras misnamed as "Fx3"/"Fx6"/"Fx30"
- C20-04 (LOW): Merge doesn't preserve type/size/pitch from existing data
- C20-05 (LOW): Stale CSV duplicates (will resolve on next CI run)
