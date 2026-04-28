# Plan 6: Negative Value Guards & Data Integrity

**Status:** pending
**Priority:** P0/P1 (crash prevention, data integrity)
**Findings addressed:** C35-01, C35-02, C35-03, C35-04, C35-05 (partial), C35-06

## Problem

The data pipeline has no validation for negative numeric values. This causes:
1. `derive_spec` crashes with `ValueError` when negative area reaches `pixel_pitch` (C35-01)
2. `openmvg.fetch` produces positive mpix from negative pixel dimensions (C35-04)
3. `parse_existing_csv` matched_sensors can contain empty strings from semicolons (C35-03)
4. `_BOM` uses literal character instead of documented escape sequence (C35-02)
5. Negative values render in template as "-2.0 µm" (C35-05)
6. `pixel_pitch` docstring missing ValueError documentation (C35-06)

## Implementation Steps

### Step 1: Fix `pixel_pitch` crash on negative area (C35-01)

**File:** `pixelpitch.py`, line 178-181

- [ ] Add `area <= 0` guard to `pixel_pitch`:
  ```python
  def pixel_pitch(area: float, mpix: float) -> float:
      if mpix <= 0 or area <= 0:
          return 0.0
      return 1000 * sqrt(area / (mpix * 10**6))
  ```
- [ ] Update docstring to note that negative area returns 0.0 (also fixes C35-06)

### Step 2: Fix `_BOM` literal vs escape sequence (C35-02)

**File:** `sources/__init__.py`, line 90

- [ ] Replace the literal BOM character with the escape sequence `﻿`
- [ ] This requires careful editing since the difference is invisible in most editors
- [ ] Verify by checking raw bytes: `_BOM = '﻿'` should show as `5c7566656666` (ASCII for `﻿`) not `efbbbf` (UTF-8 BOM bytes)

### Step 3: Fix matched_sensors empty strings from semicolons (C35-03)

**File:** `pixelpitch.py`, line 343

- [ ] Replace `sensors_str.split(";")` with `[s for s in sensors_str.split(";") if s]`
- [ ] This filters out empty strings from leading/trailing/doubled semicolons

### Step 4: Fix openmvg negative pixel dimensions (C35-04)

**File:** `sources/openmvg.py`, line 89

- [ ] Replace `if pw and ph` with `if pw > 0 and ph > 0`:
  ```python
  mpix = round(pw * ph / 1_000_000, 1) if pw > 0 and ph > 0 else None
  ```

### Step 5: Add negative value check to JS `isInvalidData` (C35-05)

**File:** `templates/pixelpitch.html`, line 263-293

- [ ] Add check for negative pitch in `isInvalidData`:
  ```javascript
  if (pitch < 0) {
    return true;
  }
  ```

### Step 6: Add test coverage (TE35-01 through TE35-04)

**File:** `tests/test_parsers_offline.py`

- [ ] TE35-01: Test `pixel_pitch` with negative area
- [ ] TE35-02: Test `derive_spec` with negative sensor dimensions
- [ ] TE35-03: Test `parse_existing_csv` with semicolons in matched_sensors
- [ ] TE35-04: Test `openmvg.fetch` with negative pixel dimensions

## Exit Criteria

- `pixel_pitch(-864.0, 33.0)` returns 0.0 instead of crashing
- `derive_spec` with negative size does not crash
- `_BOM` uses escape sequence (verified by raw bytes)
- `parse_existing_csv` with `;IMX455;` produces `['IMX455']` without empty strings
- `openmvg.fetch` with negative pixel dimensions produces `mpix=None`
- `isInvalidData` hides cameras with negative pitch
- All gate tests pass
