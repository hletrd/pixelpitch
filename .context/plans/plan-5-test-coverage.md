# Plan 5: Test Coverage Expansion

**Status:** completed
**Priority:** P1 (tests for buggy code)
**Findings addressed:** F7 (partial), F25, F33

## Problem

Critical functions (`parse_existing_csv`, `deduplicate_specs`, `merge_camera_data`, `sensor_size_from_type`, `pixel_pitch`, `match_sensors`) have zero or minimal test coverage. Test monkey-patching is fragile.

## Implementation Steps

### Step 1: Add tests for `parse_existing_csv` (F7)
- [ ] Test `has_id=True` with 10+ columns (happy path)
- [ ] Test `not has_id` with 9+ columns
- [ ] Test with special characters in names (commas, quotes)
- [ ] Test with empty CSV, header-only CSV
- [ ] Test with None values in fields
- [ ] Add round-trip test: `write_csv` → `parse_existing_csv` → verify equality

### Step 2: Add tests for `deduplicate_specs`
- [ ] Test specs with color variants that should be unified
- [ ] Test specs with different specs (type/size/pitch/mpix) that should NOT be unified
- [ ] Test specs with parenthetical suffixes
- [ ] Test specs without EXTRAS matches (exact duplicates should be removed)
- [ ] Test empty input

### Step 3: Add tests for `merge_camera_data`
- [ ] Test merge with overlapping cameras (update)
- [ ] Test merge with cameras only in existing (preserve)
- [ ] Test merge with cameras only in new (add)
- [ ] Test merge with empty existing list
- [ ] Test merge with None years

### Step 4: Add tests for `sensor_size_from_type`
- [ ] Test with type in lookup table and `use_table=True`
- [ ] Test with type in lookup table and `use_table=False`
- [ ] Test with type not in lookup table (1/x format)
- [ ] Test with None type
- [ ] Test with unknown type

### Step 5: Add tests for `pixel_pitch`
- [ ] Test with known sensor area and megapixels
- [ ] Test with edge cases (very small/large sensors)

### Step 6: Add tests for `match_sensors`
- [ ] Test with matching width/height/mpix
- [ ] Test with matching width/height but None mpix (F10 fix)
- [ ] Test with no match

### Step 7: Replace monkey-patching with `unittest.mock.patch` (F25)
- [ ] In `test_parsers_offline.py`, replace `openmvg.http_get = lambda...` with `unittest.mock.patch`

## Exit Criteria
- All new tests pass
- Test coverage includes all branches of `parse_existing_csv`
- No monkey-patching without `unittest.mock`
- Offline test gate passes
