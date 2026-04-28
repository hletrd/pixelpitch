# Plan: Cycle 46 Findings — matched_sensors Merge Preservation & LENS_RE Dead Code

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR46-01, CRIT46-01, V46-01, TR46-01, ARCH46-01, DBG46-01, TE46-01, CR46-02

---

## Task 1: Fix derive_spec to return matched_sensors=None when sensors_db unavailable — C46-01 (core) [COMPLETED]

**Finding:** C46-01 (7-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py`, derive_spec function (lines 808-816)

### Problem

`derive_spec` always initializes `matched_sensors=[]` regardless of whether `sensors_db` was provided or consulted. This creates a semantic ambiguity: `[]` can mean either "checked and found nothing" or "didn't check." The merge code preserves fields only when new value is `None`, so `[]` bypasses preservation.

### Implementation

In `pixelpitch.py`, `derive_spec` function:

**Before:**
```python
matched_sensors = []
if sensors_db and size:
    matched_sensors = match_sensors(size[0], size[1], spec.mpix, sensors_db)
```

**After:**
```python
if sensors_db and size:
    matched_sensors = match_sensors(size[0], size[1], spec.mpix, sensors_db)
else:
    matched_sensors = None
```

This makes the semantics clear:
- `None` = sensors_db was not available or size was unknown (not checked)
- `[]` = sensors_db was consulted but found no matches (checked, empty)
- `['IMX455']` = sensors_db was consulted and found matches

---

## Task 2: Add matched_sensors preservation in merge_camera_data — C46-01 (merge) [COMPLETED]

**Finding:** C46-01
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py`, merge_camera_data function (lines 489-508)

### Problem

The merge code preserves type, size, pitch, mpix, year, area from existing data when new has None. But matched_sensors is not in the preservation list.

### Implementation

In `pixelpitch.py`, `merge_camera_data` function, after the existing field preservation blocks (after line 508), add:

```python
# Preserve matched_sensors from existing data if new data has None
# (meaning sensors_db was not consulted). When new has [] (checked,
# found nothing), that is authoritative and should not be overridden.
if new_spec.matched_sensors is None and existing_spec.matched_sensors is not None:
    new_spec.matched_sensors = existing_spec.matched_sensors
```

This follows the same pattern as the other field preservation checks.

---

## Task 3: Handle matched_sensors=None in write_csv — C46-01 (output) [COMPLETED - NO CHANGE NEEDED]

**Finding:** C46-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py`, write_csv function (lines 910-912)

### Problem

`write_csv` currently uses `derived.matched_sensors` in a join, which would fail if it's `None` after the Task 1 change.

### Implementation

In `pixelpitch.py`, `write_csv` function, the `sensors_str` assignment (line 911):

**Before:**
```python
sensors_str = (
    ";".join(derived.matched_sensors) if derived.matched_sensors else ""
)
```

**After:**
```python
sensors_str = (
    ";".join(derived.matched_sensors) if derived.matched_sensors else ""
)
```

No change needed — `derived.matched_sensors` being `None` would cause `if None` to be falsy, so the `else ""` branch is taken. The `;".join()` is never called on `None`. This is safe as-is.

Wait — let me double-check. If `matched_sensors` is `None`, then `if derived.matched_sensors` evaluates to `False` (None is falsy), so `sensors_str = ""`. This is correct. No change needed.

---

## Task 4: Add matched_sensors preservation test — TE46-01 [COMPLETED]

**Finding:** TE46-01
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`

### Implementation

Add a test case in `test_merge_field_preservation` that verifies `matched_sensors` is preserved from existing data when new has `None`:

```python
# matched_sensors preservation: new has None, existing has ['IMX455']
existing_ms = SpecDerived(
    spec=Spec(name='Cam MS', category='dslr', type=None,
              size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
    size=(36.0, 24.0), area=864.0, pitch=4.39,
    matched_sensors=['IMX455'], id=0
)
new_ms = derive_spec(Spec(name='Cam MS', category='dslr', type=None,
                          size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
                     sensors_db={})  # empty db -> matched_sensors=None
merged_ms = pp.merge_camera_data([new_ms], [existing_ms])
expect("merge: preserves matched_sensors from existing",
       merged_ms[0].matched_sensors, ['IMX455'])
```

Also test that `matched_sensors=[]` (checked, found nothing) does NOT preserve existing data:

```python
# matched_sensors=[] is authoritative (checked, found nothing) — should NOT be overridden
existing_ms2 = SpecDerived(
    spec=Spec(name='Cam MS2', category='dslr', type=None,
              size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
    size=(36.0, 24.0), area=864.0, pitch=4.39,
    matched_sensors=['IMX455'], id=0
)
new_ms2 = derive_spec(Spec(name='Cam MS2', category='dslr', type=None,
                           size=(36.0, 24.0), pitch=None, mpix=45.0, year=2020),
                      sensors_db={'IMX999': {'sensor_width_mm': 99.0, 'sensor_height_mm': 99.0, 'megapixels': [99.0]}})  # db with no match -> []
merged_ms2 = pp.merge_camera_data([new_ms2], [existing_ms2])
expect("merge: [] from checked db does not preserve existing",
       merged_ms2[0].matched_sensors, [])
```

---

## Task 5: Remove LENS_RE dead code from gsmarena.py — C46-02 [COMPLETED]

**Finding:** CR46-02
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/gsmarena.py`, lines 45-50

### Problem

`LENS_RE` regex is defined but never used. Dead code similar to C44-01 FORMAT_TO_MM.

### Implementation

Remove lines 45-50 from `sources/gsmarena.py`:

```python
LENS_RE = re.compile(
    r"(?P<mp>[\d.]+)\s*MP[^,]*,"  # 50 MP,
    r"\s*f/(?P<f>[\d.]+)[^,]*,"   # f/1.7,
    r"[^,]*?(?P<role>wide|ultrawide|ultra ?wide|telephoto|tele|periscope|macro|depth)?",
    re.IGNORECASE,
)
```

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- `derive_spec` returns `matched_sensors=None` when `sensors_db` is None or empty
- `derive_spec` returns `matched_sensors=[]` when `sensors_db` was consulted but found no matches
- `merge_camera_data` preserves `matched_sensors` from existing when new has `None`
- `merge_camera_data` does NOT preserve `matched_sensors` from existing when new has `[]` (authoritative)
- `write_csv` handles `matched_sensors=None` correctly (writes empty string)
- `LENS_RE` is no longer defined in gsmarena.py
- New tests pass for matched_sensors preservation

---

## Deferred Findings

No new deferred findings. All findings from cycle 46 reviews are scheduled for implementation in this plan.
