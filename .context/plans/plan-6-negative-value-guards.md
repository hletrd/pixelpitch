# Plan 6: Negative Value Guards & Data Integrity

**Status:** completed
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

- [x] Add `area <= 0` guard to `pixel_pitch`
- [x] Update docstring to note that negative area returns 0.0 (also fixes C35-06)

### Step 2: Fix `_BOM` literal vs escape sequence (C35-02)

**File:** `sources/__init__.py`, line 90

- [x] Replace the literal BOM character with the escape sequence `﻿`
- [x] Verified by checking raw bytes: contains `5c7566656666` (ASCII for `﻿`) not `efbbbf`

### Step 3: Fix matched_sensors empty strings from semicolons (C35-03)

**File:** `pixelpitch.py`, line 343

- [x] Replace `sensors_str.split(";")` with `[s for s in sensors_str.split(";") if s]`
- [x] Filters out empty strings from leading/trailing/doubled semicolons

### Step 4: Fix openmvg negative pixel dimensions (C35-04)

**File:** `sources/openmvg.py`, line 89

- [x] Replace `if pw and ph` with `if pw > 0 and ph > 0`

### Step 5: Add negative value check to JS `isInvalidData` (C35-05)

**File:** `templates/pixelpitch.html`, line 263-293

- [x] Add check for negative pitch in `isInvalidData`

### Step 6: Add test coverage (TE35-01 through TE35-04)

**File:** `tests/test_parsers_offline.py`

- [x] TE35-01: Test `pixel_pitch` with negative area
- [x] TE35-02: Test `derive_spec` with negative sensor dimensions
- [x] TE35-03: Test `parse_existing_csv` with semicolons in matched_sensors
- [x] TE35-04: Test `openmvg.fetch` with negative pixel dimensions

## Exit Criteria

- `pixel_pitch(-864.0, 33.0)` returns 0.0 instead of crashing
- `derive_spec` with negative size does not crash
- `_BOM` uses escape sequence (verified by raw bytes)
- `parse_existing_csv` with `;IMX455;` produces `['IMX455']` without empty strings
- `openmvg.fetch` with negative pixel dimensions produces `mpix=None`
- `isInvalidData` hides cameras with negative pitch
- All gate tests pass
