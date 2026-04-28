# C52-01: parse_existing_csv year tolerance + record_id tolerance

**Cycle:** 52
**Status:** COMPLETED
**Findings addressed:** F52-01, F52-02, F52-04, F52-DS-01

## Implementation

- `73d6843` fix(csv): _safe_year + _safe_int_id helpers wired into
  parse_existing_csv
- `b32de64` test(csv): year + record_id parse-tolerance tests

Both gates pass at `b32de64`:
- `flake8 .` → 0 errors
- `python3 -m tests.test_parsers_offline` → All checks passed (including
  the two new test sections: 9 year-tolerance assertions + 6
  id-tolerance assertions)

## Background

`parse_existing_csv` in `pixelpitch.py` is the deserialization side of
the CSV round-trip. Cycles 50-51 hardened the `matched_sensors` column
against Excel hand-edits (`;`-injection, whitespace, duplicates). Two
remaining columns parse via `int(...)` and silently drop or skip data
when Excel coerces them to a `"X.0"` float string:

- `year` (line 366-372) → silent drop of the year value
- `record_id` (line 319) → broad-except logs and skips the entire row

`write_csv` always emits clean integer values, so the internal
round-trip works today. The fix is defense-in-depth against external
edits.

## Repo policy applied

Per CLAUDE.md and AGENTS.md (no project-specific override of standard
git policy):

- All commits GPG-signed.
- Conventional commit + gitmoji.
- One commit per fine-grained change.
- `git pull --rebase` before push.
- No `--no-verify`.
- Run gates (flake8, `tests.test_parsers_offline`) before push.

## Plan

### Step 1 — Refresh per-agent reviews and aggregate (F52-03)

DONE in this cycle's review pass. The 12 review files and `_aggregate.md`
have been refreshed for cycle 52.

### Step 2 — Implement F52-01 + F52-02 in pixelpitch.py

**File:** `pixelpitch.py`

- Add a small `_safe_year(year_str: str) -> Optional[int]` helper near
  `_safe_float` (line 268).
  - Returns None for empty/None inputs.
  - Tries `int(year_str)` first.
  - Falls back to `int(float(year_str))` on ValueError; this also
    handles `"2023.0"` and `" 2023.0 "`.
  - Re-applies the 1900-2100 range guard.
  - Catches `OverflowError` (for `inf`) and `ValueError` (for `nan`).

- Replace the inline year-parse block (lines 365-372) with
  `year = _safe_year(year_str)`.

- Add a small `_safe_int_id(s: str) -> Optional[int]` helper for
  `record_id`.
  - Returns None for empty inputs.
  - Tries `int(s)` first; falls back to `int(float(s))` on ValueError.
  - Catches `OverflowError`.
  - Returns None on any failure (instead of raising — this avoids the
    full-row skip in the broad except at line 390).

- Replace `record_id = int(values[0]) if values[0] else None` (line 319)
  with `record_id = _safe_int_id(values[0])`.

- Update the `parse_existing_csv` docstring to mention "tolerant to
  Excel hand-edits in numeric columns" (F52-DS-01).

### Step 3 — Add parse-tolerance test (F52-04)

**File:** `tests/test_parsers_offline.py`

Add a section near the existing matched_sensors parse-tolerance block:

```python
# == year parse tolerance (Excel hand-edit) ==
print("\n== year parse tolerance (Excel hand-edit) ==")
csv_text = (
    "id,name,category,type,sensor_width_mm,sensor_height_mm,"
    "sensor_area_mm2,megapixels,pixel_pitch_um,year,matched_sensors\n"
    "0,Cam Int,fixed,,,,,,,2023,\n"
    "1,Cam Float,fixed,,,,,,,2023.0,\n"
    "2,Cam Pad,fixed,,,,,,, 2023 ,\n"
    "3,Cam Bad,fixed,,,,,,,abc,\n"
    "4,Cam Empty,fixed,,,,,,,,\n"
)
parsed = parse_existing_csv(csv_text)
assert_equal(len(parsed), 5,
              "year-tolerance: 5 rows parsed")
assert_equal(parsed[0].spec.year, 2023,
              "year-tolerance: integer year parses")
assert_equal(parsed[1].spec.year, 2023,
              "year-tolerance: float year (2023.0) parses")
assert_equal(parsed[2].spec.year, 2023,
              "year-tolerance: padded year parses")
assert_equal(parsed[3].spec.year, None,
              "year-tolerance: garbage year → None")
assert_equal(parsed[4].spec.year, None,
              "year-tolerance: empty year → None")

# == record_id parse tolerance (Excel hand-edit) ==
print("\n== record_id parse tolerance (Excel hand-edit) ==")
csv_text = (
    "id,name,category,type,sensor_width_mm,sensor_height_mm,"
    "sensor_area_mm2,megapixels,pixel_pitch_um,year,matched_sensors\n"
    "5,Cam Int,fixed,,,,,,,,\n"
    "5.0,Cam Float,fixed,,,,,,,,\n"
)
parsed = parse_existing_csv(csv_text)
assert_equal(len(parsed), 2,
              "id-tolerance: 2 rows parsed (no row drop)")
assert_equal(parsed[0].id, 5,
              "id-tolerance: integer id parses")
assert_equal(parsed[1].id, 5,
              "id-tolerance: float id (5.0) parses")
```

### Step 4 — Run gates and commit

- `flake8 .` → must exit 0.
- `python3 -m tests.test_parsers_offline` → must exit 0 with all
  sections green.

Commits (one per fine-grained change, all GPG-signed):

1. `fix(csv): 🛡️ tolerate "2023.0"-style years in parse_existing_csv`
   - Adds `_safe_year` helper, replaces inline block.
2. `fix(csv): 🛡️ tolerate "5.0"-style record_ids in parse_existing_csv`
   - Adds `_safe_int_id` helper, replaces inline `int()` call.
3. `test(csv): ✅ add year and record_id parse-tolerance tests`
   - Adds the two test sections in `tests/test_parsers_offline.py`.
4. `docs(reviews): 📝 add cycle 52 reviews and plan for year/id parse hardening`
   - Adds 12 review files + `_aggregate.md` + this plan file.

### Exit criteria

- Both gates pass at HEAD.
- `parse_existing_csv` accepts `2023`, `2023.0`, ` 2023 `, and rejects
  `abc`, `nan`, `inf`, out-of-range integers.
- `parse_existing_csv` accepts `5`, `5.0`, ` 5 ` for id, and rejects
  `abc` (returns None, does not crash).
- Plan archived (status: COMPLETED) only after every commit lands.

## Deferred items recorded for cycle 52

No new deferrals this cycle. F52-03 (process — review hygiene) is
addressed in step 1; F52-DS-01 (docstring update) folds into step 2;
F52-01, F52-02, F52-04 are all implemented in steps 2-3.

The cycle 51 deferred queue (`.context/plans/deferred.md`) remains
unchanged. Re-validation of those entries is a process task — F51-03
(deferred queue audit) remains a future cycle's work.
