# Code Review (Cycle 43) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-42 fixes, focusing on NEW issues

## Previous Findings Status

C42-01 through C42-05 all implemented and verified. `merge_camera_data` now has the derived.size consistency check. CLI `--limit` has try/except. Docstring updated. All gate tests pass.

## New Findings

### CR43-01: `merge_camera_data` overrides `derived.pitch` from existing even when `spec.pitch` was already preserved — double-write inconsistency

**File:** `pixelpitch.py`, lines 456-467 and 490-501
**Severity:** LOW | **Confidence:** HIGH

When `spec.size` is None and existing has a measured size, the C42-01 fix adds:
```python
if new_spec.size is not None and new_spec.size != new_spec.spec.size:
    new_spec.size = existing_spec.size
    new_spec.area = existing_spec.area
    new_spec.pitch = existing_spec.pitch
```

This overrides `new_spec.pitch` with `existing_spec.pitch`. But then later, lines 490-501:
```python
if new_spec.pitch is None and existing_spec.pitch is not None:
    new_spec.pitch = existing_spec.pitch
# Consistency: derived.pitch must always track spec.pitch when the latter is set.
if (new_spec.spec.pitch is not None
        and isfinite(new_spec.spec.pitch) and new_spec.spec.pitch > 0
        and new_spec.pitch != new_spec.spec.pitch):
    new_spec.pitch = new_spec.spec.pitch
```

If `existing_spec.pitch` was set from an existing computed value (not from `spec.pitch`), but the new `spec.pitch` was also preserved from existing (line 468-473), then the consistency check at lines 498-501 would overwrite the derived.pitch from line 467 with `spec.pitch`. This is actually correct behavior — `spec.pitch` is authoritative over `derived.pitch`. But the double-write creates a subtle ordering dependency:

1. Line 467: `new_spec.pitch = existing_spec.pitch` (derived.pitch from existing)
2. Line 468-473: `new_spec.spec.pitch = existing_spec.spec.pitch` (spec.pitch from existing)
3. Line 498-501: if `spec.pitch != derived.pitch`, override `derived.pitch = spec.pitch`

If `existing_spec.pitch` and `existing_spec.spec.pitch` are different (which can happen if the existing derived.pitch was computed from area+mpix while spec.pitch was a direct measurement), the C42-01 fix writes `existing_spec.pitch` at step 1, but then step 3 overwrites it with `spec.pitch`. The final result is correct (spec.pitch wins), but the intermediate write at step 1 is wasted and misleading to readers.

This is a code clarity issue, not a correctness bug. The fix at step 1 should NOT write `new_spec.pitch = existing_spec.pitch` because the consistency check at step 3 already handles it. Writing `derived.pitch` at step 1 and then overwriting it at step 3 is confusing.

**Fix:** Remove `new_spec.pitch = existing_spec.pitch` from the C42-01 fix block (line 467). The existing pitch consistency logic at lines 490-501 already ensures derived.pitch tracks spec.pitch. The C42-01 fix should only override `derived.size` and `derived.area` (which don't have a later consistency check).

---

### CR43-02: `gsmarena._phone_to_spec` returns Spec with `size` set from `PHONE_TYPE_SIZE` lookup — this becomes `spec.size` in the merged data, masquerading as measured data

**File:** `sources/gsmarena.py`, lines 144-167
**Severity:** MEDIUM | **Confidence:** HIGH

When GSMArena provides a sensor type like "1/1.3", `_phone_to_spec` does:
```python
size = PHONE_TYPE_SIZE.get(sensor_type) if sensor_type else None
```

And returns `Spec(..., size=size, ...)`. This means `spec.size` is set to the TYPE_SIZE lookup value, not None. When this data flows through `derive_spec`:
```python
if spec.size is None:
    size = sensor_size_from_type(spec.type)
else:
    size = spec.size
```

Since `spec.size` is not None, the lookup is skipped and `derived.size = spec.size` directly. This is correct.

But the problem is in `merge_camera_data`: when a Geizhals entry exists with a measured `spec.size` that differs from the TYPE_SIZE value, the merge sees `new_spec.spec.size is NOT None` (because GSMArena set it from the lookup table), so it does NOT preserve the measured Geizhals value:

```python
if new_spec.spec.size is None and existing_spec.spec.size is not None:
    new_spec.spec.size = existing_spec.spec.size
```

This condition is False because `new_spec.spec.size` was set by GSMArena. The measured Geizhals value is silently overwritten by the approximate TYPE_SIZE lookup value. This is a WORSE variant of the C42-01 bug — not only are derived fields inconsistent, the spec.size itself is wrong because the merge never preserves it.

**Concrete scenario:**
```python
# Geizhals: measured size from product specs
existing = Spec(name="Samsung S25 Ultra", category="smartphone", type="1/1.3",
                size=(9.76, 7.30), pitch=None, mpix=200.0, year=2025)

# GSMArena: size from TYPE_SIZE lookup
new = Spec(name="Samsung S25 Ultra", category="smartphone", type="1/1.3",
           size=(9.84, 7.40), pitch=None, mpix=200.0, year=2025)  # from PHONE_TYPE_SIZE

# Merge: new_spec.spec.size = (9.84, 7.40) → NOT None → Geizhals measured value (9.76, 7.30) is LOST
```

**Fix:** GSMArena should set `spec.size = None` and only use `spec.type` to indicate the sensor format. Let `derive_spec` compute `derived.size` from the type lookup. This way, `merge_camera_data` will correctly preserve the measured Geizhals `spec.size` because `new_spec.spec.size is None` will be True.

Alternatively, if GSMArena should provide the size, it needs a provenance flag to distinguish "measured" from "type-lookup" sizes. But that's a larger refactor. The simpler fix is to not set `spec.size` in GSMArena.

---

### CR43-02b: Same issue exists for `cined.py` — `_parse_camera_page` sets `spec.size` from `FORMAT_TO_MM` lookup

**File:** `sources/cined.py`, lines 94-102
**Severity:** MEDIUM | **Confidence:** MEDIUM

```python
if size is None and fmt:
    size = FORMAT_TO_MM.get(fmt.lower())
```

This sets `spec.size` from a format lookup table, same as GSMArena. If a cinema camera also appears in Geizhals data with a measured size, the merge will not preserve the Geizhals value because `new_spec.spec.size` is not None.

The same fix applies: don't set `spec.size` from format lookups; use `spec.type` instead and let `derive_spec` handle it.

---

## Summary

- CR43-01 (LOW): C42-01 fix writes `derived.pitch` from existing but it gets overwritten by the pitch consistency check — redundant write is misleading, should be removed
- CR43-02 (MEDIUM): GSMArena sets `spec.size` from TYPE_SIZE lookup, preventing merge from preserving measured Geizhals values — silent data loss for phones with measured dimensions
- CR43-02b (MEDIUM): CineD sets `spec.size` from FORMAT_TO_MM lookup, same issue as GSMArena
