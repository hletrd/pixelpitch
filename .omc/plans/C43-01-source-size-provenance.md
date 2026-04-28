# Plan: Cycle 43 Findings — Source Size Provenance, Redundant Pitch Write

**Created:** 2026-04-28
**Status:** IN PROGRESS
**Source Reviews:** CR43-02, CR43-02b, SR43-01, CRIT43-01, V43-02, TR43-01, ARCH43-01, DBG43-01, DBG43-02, DES43-01, DOC43-01, DOC43-02, TE43-01, TE43-02, CR43-01, CRIT43-02, V43-03, DBG43-03

---

## Task 1: Fix GSMArena spec.size provenance — stop setting spec.size from TYPE_SIZE lookup — C43-01 (core)

**Finding:** C43-01 (14-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `sources/gsmarena.py`, lines 144-167

### Problem

GSMArena's `_phone_to_spec` sets `spec.size` from the `PHONE_TYPE_SIZE` lookup table (e.g., `size = (9.84, 7.40)` for type "1/1.3"). This means `spec.size` is not None, which prevents `merge_camera_data` from preserving measured Geizhals values. The merge condition `if new_spec.spec.size is None and existing_spec.spec.size is not None` is False because GSMArena set `spec.size` from the lookup table.

### Implementation

In `sources/gsmarena.py`, change `_phone_to_spec` to NOT set `spec.size` from the lookup table:

**Before:**
```python
fmt_match = TYPE_FRACTIONAL_RE.search(main)
sensor_type = fmt_match.group(1) if fmt_match else None
size = PHONE_TYPE_SIZE.get(sensor_type) if sensor_type else None
```

**After:**
```python
fmt_match = TYPE_FRACTIONAL_RE.search(main)
sensor_type = fmt_match.group(1) if fmt_match else None
# Don't set spec.size from TYPE_SIZE lookup — the lookup table provides
# approximate dimensions based on the fractional-inch designation. Setting
# spec.size from the lookup prevents merge_camera_data from preserving
# more accurate measured values from Geizhals (because the merge only
# preserves existing spec.size when new spec.size is None). Instead, set
# only spec.type and let derive_spec compute derived.size from the type.
size = None
```

This means `Spec.size = None` and `Spec.type = "1/1.3"`. When `derive_spec` processes this:
1. `spec.size is None` → `size = sensor_size_from_type(spec.type)` → `size = (9.84, 7.40)` from TYPE_SIZE
2. `derived.size = (9.84, 7.40)` — same value as before for GSMArena-only cameras
3. But now `spec.size is None` → merge will preserve Geizhals measured values

Also remove the `PHONE_TYPE_SIZE` dict and the import of `TYPE_SIZE` from pixelpitch since it's no longer needed (the central `TYPE_SIZE` in pixelpitch.py is used by `sensor_size_from_type` via `derive_spec`).

**Before:**
```python
from pixelpitch import TYPE_SIZE as SENSOR_TYPE_SIZE
...
PHONE_TYPE_SIZE: dict[str, tuple[float, float]] = dict(SENSOR_TYPE_SIZE)
```

**After:** Remove both lines. The `sensor_size_from_type` function in pixelpitch.py already has the full TYPE_SIZE table.

Also update the test in `tests/test_parsers_offline.py` for the GSMArena fixture. The test currently expects `spec.size = (9.84, 7.40)` from the lookup table. After the fix, `spec.size` will be None and the dimensions come from `derived.size`.

---

## Task 2: Fix CineD spec.size provenance — stop setting spec.size from FORMAT_TO_MM lookup — C43-01

**Finding:** C43-01 (14-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `sources/cined.py`, lines 94-102

### Problem

Same issue as Task 1 but for CineD. `_parse_camera_page` sets `spec.size` from `FORMAT_TO_MM` when no explicit mm dimensions are found.

### Implementation

In `sources/cined.py`, change `_parse_camera_page` to NOT set `spec.size` from FORMAT_TO_MM:

**Before:**
```python
if size is None and fmt:
    size = FORMAT_TO_MM.get(fmt.lower())
```

**After:**
```python
# Don't set spec.size from FORMAT_TO_MM lookup — the lookup provides
# approximate dimensions from the format class name. Setting spec.size
# from the lookup prevents merge_camera_data from preserving more accurate
# measured values from Geizhals. Leave spec.size = None; the template
# will show "unknown" for sensor size, which is more honest than showing
# an approximation as if it were measured data.
# Note: we also don't set spec.type because format class names like
# "Super 35" or "APS-C" are not fractional-inch types that TYPE_SIZE
# understands. The dimensions will be available only when Geizhals
# provides measured values for the same camera.
```

CineD's format names (e.g., "Super 35", "Full Frame", "APS-C") don't correspond to TYPE_SIZE fractional-inch entries. So we can't set `spec.type` instead. We simply leave `spec.size = None` and `spec.type = None` for format-only entries. The CineD module already has a `size` from explicit mm dimensions when available (line 95-100), which will still be set.

Also update the module docstring to match: currently says "If only the format class is given, we leave size None and let pixelpitch.derive_spec compute area from TYPE_SIZE / format" — but after this fix, we actually do leave size None (matching the docstring that was previously wrong!).

---

## Task 3: Remove redundant `derived.pitch` write from C42-01 fix — C43-02

**Finding:** C43-02 (4-agent agreement)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py`, line 467

### Implementation

In `pixelpitch.py`, `merge_camera_data`, remove `new_spec.pitch = existing_spec.pitch` from the C42-01 fix block:

**Before (lines 464-467):**
```python
if new_spec.size is not None and new_spec.size != new_spec.spec.size:
    new_spec.size = existing_spec.size
    new_spec.area = existing_spec.area
    new_spec.pitch = existing_spec.pitch
```

**After:**
```python
if new_spec.size is not None and new_spec.size != new_spec.spec.size:
    new_spec.size = existing_spec.size
    new_spec.area = existing_spec.area
    # Note: derived.pitch is NOT overridden here because the
    # pitch consistency check at lines 498-501 already ensures
    # derived.pitch tracks spec.pitch when the latter is preserved.
```

---

## Task 4: Update test for GSMArena spec.size=None — C43-01 test

**Finding:** TE43-01
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`

### Implementation

Update `test_gsmarena` to expect `spec.size = None` instead of the TYPE_SIZE lookup value:

**Before:**
```python
expect("size width mm",  spec.size[0] if spec.size else None, 9.84, tol=0.1)
expect("size height mm", spec.size[1] if spec.size else None, 7.40, tol=0.1)
```

**After:**
```python
# spec.size is None because GSMArena only provides the fractional-inch type,
# not measured dimensions. The type-derived dimensions are in derived.size.
expect("spec.size is None (type-derived, not measured)", spec.size, None)
```

Also add a test that `derived.size` is computed from the type:
```python
# Test derive_spec produces the correct derived.size from spec.type
import pixelpitch as pp
derived = pp.derive_spec(spec)
expect("derived.size from type 1/1.3",
       derived.size[0] if derived.size else None, 9.84, tol=0.1)
expect("derived.size from type 1/1.3 height",
       derived.size[1] if derived.size else None, 7.40, tol=0.1)
```

Also add a new test for the merge scenario where GSMArena data with `spec.size=None` preserves Geizhals measured values:

```python
def test_merge_gsmarena_measured_preserved():
    """Verify merge_camera_data preserves measured Geizhals spec.size
    when GSMArena provides only spec.type (spec.size=None)."""
    section("merge GSMArena measured preservation")
    import pixelpitch as pp
    from models import Spec

    # Existing: measured size from Geizhals (slightly different from TYPE_SIZE)
    existing_spec = Spec(name='Phone X', category='smartphone', type='1/1.3',
                         size=(9.76, 7.30), pitch=None, mpix=200.0, year=2025)
    existing = pp.derive_spec(existing_spec)
    existing.id = 0

    # New: spec.size=None, spec.type='1/1.3' (GSMArena after fix)
    new_spec = Spec(name='Phone X', category='smartphone', type='1/1.3',
                    size=None, pitch=None, mpix=200.0, year=2025)
    new = pp.derive_spec(new_spec)

    merged = pp.merge_camera_data([new], [existing])
    m = merged[0]

    # spec.size should be preserved from existing (measured Geizhals value)
    expect("merge GSMArena: spec.size preserved from measured",
           m.spec.size, (9.76, 7.30), tol=0.01)
    # derived.size must match spec.size (not TYPE_SIZE)
    expect("merge GSMArena: derived.size matches spec.size",
           m.size, m.spec.size, tol=0.01)
    # derived.area must be consistent with derived.size
    expect("merge GSMArena: area consistent",
           abs(m.area - m.size[0] * m.size[1]) < 0.01, True)
```

---

## Task 5: Update GSMArena module docstring and remove unused import/variable — C43-01 cleanup

**Finding:** DOC43-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/gsmarena.py`

### Implementation

1. Remove the `from pixelpitch import TYPE_SIZE as SENSOR_TYPE_SIZE` import (line 27)
2. Remove the `PHONE_TYPE_SIZE` dict (line 54)
3. Update the module docstring to note that spec.size is not set from the lookup table

---

## Task 6: Update CineD module docstring — C43-01 cleanup

**Finding:** DOC43-02
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/cined.py`

### Implementation

The module docstring already says "If only the format class is given, we leave size None and let pixelpitch.derive_spec compute area from TYPE_SIZE / format." After Task 2, the code matches the docstring — no change needed. But update the note to clarify that FORMAT_TO_MM is still defined (used by the regex coverage test) but no longer used to set spec.size.

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- GSMArena `_phone_to_spec` returns `spec.size=None` with `spec.type='1/1.3'`
- CineD `_parse_camera_page` returns `spec.size=None` for format-class-only entries
- `merge_camera_data` with GSMArena `spec.size=None` preserves measured Geizhals values
- `derive_spec` computes `derived.size` from `spec.type` correctly
- C42-01 fix still works (derived.size consistency check)
- No redundant `derived.pitch` write in C42-01 fix block
- All existing tests pass (with updates for new spec.size=None behavior)

---

## Deferred Findings

No new deferred findings. All findings from cycle 43 reviews are either scheduled for implementation in this plan or were already deferred in previous cycles.
