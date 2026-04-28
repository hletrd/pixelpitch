# C55-01: preserve per-source matched_sensors cache when sensors.json unavailable

**Cycle:** 55 (orchestrator cycle 8)
**Status:** COMPLETED
**Findings addressed:** F55-01, F55-03, F55-DOC-01, F55-DOC-02

## Implementation

- `fix(merge)`: preserve matched_sensors cache in `_load_per_source_csvs`
  when `sensors.json` is unavailable.
- `test(csv)`: add `_load_per_source_csvs` cache-fallback test (F55-01)
  and `parse_existing_csv` BOM has_id detection regression test (F55-03).
- `docs(readme)`: enumerate generated HTML pages including
  smartphone.html and cinema.html (F55-DOC-02).

Both gates pass post-fix:
- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  (including the two new sections).

## Problem

After C54-01, `_load_per_source_csvs` (`pixelpitch.py:1074-1086`)
unconditionally overwrites `matched_sensors` for every parsed row.
When `load_sensors_database()` returns `{}` (file missing or invalid
JSON), the rows fall through to `d.matched_sensors = None`, dropping
the per-row cache that was just parsed from the CSV.

`merge_camera_data`'s existing-only branch (line 615-628) does NOT
overwrite when sensors_db is empty — it preserves the cached
matched_sensors. The two paths therefore disagree on cache fallback
semantics.

The desirable contract is: **the per-source CSV's matched_sensors
column is a cache; when sensors.json cannot be consulted, fall back
to the cache rather than dropping it.** This matches the merge
existing-only branch and the docstring's stated cache role.

## Plan

### Step 1: fix `_load_per_source_csvs` empty-db fallback

In the per-row loop, when sensors_db is empty/None, leave
`d.matched_sensors` untouched (the value parsed from the CSV).
When sensors_db is non-empty AND `d.size` is set, refresh.
When `d.size` is None, set to `None` (matches `derive_spec`
contract — size unknown means "not checked").

```python
for d in parsed:
    d.id = None
    if d.size is not None:
        if sensors_db is None:
            sensors_db = load_sensors_database()
        if sensors_db:
            d.matched_sensors = match_sensors(
                d.size[0], d.size[1], d.spec.mpix, sensors_db
            )
        # else: keep parsed cache as fallback
    else:
        d.matched_sensors = None
```

### Step 2: docstring

Update `_load_per_source_csvs` docstring to describe the new
contract: refresh when sensors.json is loadable, fall back to cache
otherwise, drop only when size is unknown.

### Step 3: tests

Add two tests in `tests/test_parsers_offline.py`:

1. **F55-01 fix:** monkeypatch `load_sensors_database` to return
   `{}`. Write a per-source CSV with a known `matched_sensors=["S1"]`
   row. Call `_load_per_source_csvs`. Assert the loaded row's
   matched_sensors still equals `["S1"]` (cache preserved).
2. **F55-03 (BOM regression):** prepend `﻿` to a `parse_existing_csv`
   input. Assert has_id detection still resolves correctly.

### Step 4: README mention smartphone/cinema (F55-DOC-02)

Add a one-line mention to `README.md` listing all generated pages.

### Step 5: gates

- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → green.

## Out of scope (deferred)

- F55-02 (boundary tolerance test for `match_sensors`): kept as
  deferred — boundary semantics are already implicitly correct via
  `<=`; no observed bug.
- F55-04 (latent input mutation in `merge_camera_data`): no current
  caller observes; defer per F55-04 disposition.
- F55-05 (hand-edited blank-leading-cell CSVs): out-of-scope per
  F55-05 disposition (CSVs are produced by write_csv).
- F55-06 (`_refresh_matched_sensors` helper extraction): bundled
  conceptually into F55-01 fix but defer the extraction itself —
  the two paths still differ in their existing-only vs per-source
  semantics enough that a unified helper would obscure them.

## Repo policy applied

Per `~/.claude/CLAUDE.md`:

- All commits GPG-signed (`git commit -S`).
- Conventional Commits + gitmoji.
- One commit per fine-grained change (fix, test, docs).
- `git pull --rebase` before push.
- No `--no-verify`, no Co-Authored-By.
