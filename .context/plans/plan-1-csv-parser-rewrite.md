# Plan 1: CSV Parser Rewrite — Correctness & Data Integrity

**Status:** completed
**Priority:** P0 (data corruption risk)
**Findings addressed:** F1, F2, F7 (partial), F19

## Problem

`parse_existing_csv` (pixelpitch.py lines 181–291) has confirmed column-index bugs that cause IndexError or data corruption (sensors_str reads year column). The hand-rolled CSV parser doesn't handle RFC 4180 edge cases correctly. `write_csv` only escapes the name field.

## Implementation Steps

### Step 1: Rewrite `parse_existing_csv` using `csv` module
- [ ] Replace manual character-by-character parsing with `csv.reader(io.StringIO(csv_content))`
- [ ] Simplify column-index logic: detect `has_id` from header, then use consistent index mapping
- [ ] Fix the `sensors_str = values[8]` bug (should be `values[9]` when `len > 9`)
- [ ] Fix the `has_id and len >= 9` branch where `year_str = values[9]` causes IndexError
- [ ] Remove dead/unreachable branches

### Step 2: Fix `write_csv` to use `csv.writer`
- [ ] Replace manual string formatting with `csv.writer`
- [ ] Ensure all fields are properly escaped (not just name)
- [ ] Verify output format is unchanged for existing consumers

### Step 3: Add round-trip test
- [ ] Write test that creates SpecDerived objects, writes CSV, reads it back, and asserts equality
- [ ] Test with special characters in names (commas, quotes, newlines)
- [ ] Test with None values in all fields

## Exit Criteria
- All `parse_existing_csv` branches produce correct column mappings
- Round-trip test (write → read) preserves all data
- No manual CSV parsing logic remains
- Offline test gate passes
