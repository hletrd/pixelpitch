# C54-01: refresh `matched_sensors` on per-source CSV load + add test

**Cycle:** 54
**Status:** COMPLETED
**Findings addressed:** F54-01, F54-T01, F54-DOC-01, F54-DOC-02

## Implementation

- `1660cc5` fix(merge): 🐛 refresh matched_sensors on per-source CSV load
- `4f3096b` test(merge): ✅ assert _load_per_source_csvs refreshes matched_sensors
- `f6a5078` docs(merge): 📝 document matched_sensors tri-valued sentinel contract

Both gates pass at `f6a5078`:
- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  (54 sections, including `_load_per_source_csvs refresh against
  sensors.json`).

## Problem

`_load_per_source_csvs` (pixelpitch.py:1028-1053) parses each
`dist/camera-data-{source}.csv` via `parse_existing_csv` and only
clears the per-row `id`. The `matched_sensors` column is parsed
verbatim from the file. Per-source CSVs were written by
`fetch_source` against whatever `sensors.json` was current at write
time. If `sensors.json` is later edited (sensor renamed, removed,
or megapixel list trimmed) but the per-source CSV is not regenerated,
the stale matched_sensors propagate through `merge_camera_data` into
`dist/camera-data.csv` and the rendered HTML.

`merge_camera_data` only re-matches **existing-only** cameras
(those in existing CSV but not in new). Cameras coming back via
per-source CSVs are in `new_specs`, so they keep stale matches.

The function's own docstring says these CSVs "serve as caches
between deployments", but the implementation treats them as
authoritative for everything except the id. Doc/code mismatch.

## Plan

### Step 1: refresh matched_sensors on load

In `_load_per_source_csvs`:
- Lazy-load `sensors_db` once before the loop (only if any per-source
  CSV exists).
- For each parsed `SpecDerived`, replace `matched_sensors` by
  recomputing via `match_sensors(d.size[0], d.size[1], d.spec.mpix,
  sensors_db)` when `d.size` is known and `sensors_db` is non-empty.
- When `d.size` is None or sensors_db is empty, set
  `matched_sensors=None` (sentinel for "not checked"). This matches
  the `derive_spec` contract.

Calling `derive_spec` directly is heavier than necessary because it
also recomputes size/area/pitch — those values from the CSV are
already correct. Recomputing only `matched_sensors` is cleaner and
matches the cache semantics.

### Step 2: update docstrings

- `_load_per_source_csvs`: document the cache semantics and
  matched_sensors refresh.
- `merge_camera_data`: add a sentence describing the matched_sensors
  preservation rules from C46.

### Step 3: add unit tests

In `tests/test_parsers_offline.py`, add a section
"`_load_per_source_csvs` refresh" that:

1. Writes a temp directory with a `camera-data-openmvg.csv` containing
   one row whose `matched_sensors` references a name NOT present in
   the actual `sensors.json`.
2. Calls `pixelpitch._load_per_source_csvs(tmp_dir)`.
3. Asserts the loaded `SpecDerived.matched_sensors` no longer contains
   the stale name (the refresh dropped it).
4. Asserts `id` is None.
5. Asserts the missing per-source CSV file does not raise.

### Step 4: gate

- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → green.

## Repo policy applied

Per `~/.claude/CLAUDE.md`:

- All commits GPG-signed (`git commit -S`).
- Conventional Commits + gitmoji.
- One commit per fine-grained change (refactor, test, docs).
- `git pull --rebase` before push.
- No `--no-verify`, no Co-Authored-By.
