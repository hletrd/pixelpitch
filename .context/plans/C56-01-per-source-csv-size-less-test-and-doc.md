# C56-01: pin `_load_per_source_csvs` size-less branch with a test and tighten docstring

**Cycle:** 56 (orchestrator cycle 9)
**Status:** COMPLETED
**Findings addressed:** F56-01, F56-02, F56-03

## Implementation summary

- `docs(merge)`: tighten `_load_per_source_csvs` docstring to call out
  that the size-less branch overrides any cached value (F56-03).
- `test(csv)`: add `_load_per_source_csvs` size-less branch test
  asserting the cache is dropped when sensors_db is non-empty (F56-01).
- F56-02 (empty matched_sensors preserved as `[]`) is already pinned
  by the existing C55-01 cache-fallback section
  (`tests/test_parsers_offline.py:2344-2348`); no new test needed.

Both gates pass post-fix:
- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  including the new `_load_per_source_csvs size-less row drops cache
  (sensors_db non-empty)` section.

## Background

After C55-01 the `_load_per_source_csvs` per-row branch is:

```python
for d in parsed:
    d.id = None
    if d.size is not None:
        if sensors_db is None:
            sensors_db = load_sensors_database()
        if sensors_db:
            d.matched_sensors = match_sensors(...)
        # else: keep parsed cache as fallback (F55-01)
    else:
        # Size unknown — matched_sensors is meaningless;
        # honor derive_spec's "not checked" sentinel.
        d.matched_sensors = None
```

Two contracts are observable:
1. **size-less rows**: `matched_sensors` is forced to `None`,
   *overriding any cached value* (intentional; honors the
   `derive_spec` "not checked when size unknown" sentinel).
2. **size present + sensors_db empty**: cache preserved (F55-01).

Tests pin both the refresh path (C54-01) and the cache fallback
path (C55-01). The size-less branch is currently untested. The
docstring at `pixelpitch.py:1041-1057` is correct but does not
explicitly state that the size-less branch overrides any cached
value.

## Plan

### Step 1: tighten docstring on `_load_per_source_csvs` (F56-03)

In `pixelpitch.py:1041-1057`, change

> When the row has no sensor size, matched_sensors is set to
> `None` ("not checked"), matching the `derive_spec` contract.

to

> When the row has no sensor size, matched_sensors is set to
> `None` ("not checked", overriding any cached value), matching
> the `derive_spec` contract.

This pins the override semantics so a future maintainer who reads
the docstring will not "fix" the override branch by mistake.

### Step 2: add a size-less branch test (F56-01)

Add a new section to `tests/test_parsers_offline.py`:

```python
print("--- _load_per_source_csvs size-less row drops cache ---")

# A per-source CSV row with empty width/height cells should have
# matched_sensors = None after _load_per_source_csvs, even if the
# parsed cache had a value.

with tempfile.TemporaryDirectory() as td:
    tmpdir = Path(td)
    src = SOURCE_REGISTRY[0]
    csv_path = tmpdir / f"camera-data-{src}.csv"
    csv_path.write_text(
        "id,name,category,sensor_type,width,height,area,mpix,pitch,year,matched_sensors\n"
        "1,FooCam,smartphone,1/2.3\",,,,12.0,,2024,IMX355\n",
        encoding="utf-8",
    )

    real_load = pixelpitch.load_sensors_database
    pixelpitch.load_sensors_database = lambda: {"IMX355": {...}}
    try:
        loaded = pixelpitch._load_per_source_csvs(tmpdir)
    finally:
        pixelpitch.load_sensors_database = real_load

    assert len(loaded) == 1
    assert loaded[0].size is None
    assert loaded[0].matched_sensors is None, (
        f"size-less row must drop matched_sensors cache (got "
        f"{loaded[0].matched_sensors})"
    )

print("OK: _load_per_source_csvs size-less row drops cache")
```

### Step 3: extend the existing C55-01 cache-fallback section with empty-list assertion (F56-02)

In the existing `_load_per_source_csvs cache fallback when
sensors.json missing` section, also assert that a row with an
empty `matched_sensors` cache (`""` → `[]` after parse) is
preserved as `[]` when sensors_db is empty.

This pins the strip+dedup interaction with the cache-preservation
branch: the empty-string column already parses to `[]`, and the
preservation branch must not mutate it.

### Step 4: gates

- `python3 -m flake8 .` → must remain 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  including the new size-less branch and empty-cache assertions.

## Acceptance Criteria

- Docstring updated with override callout.
- New size-less branch test in test_parsers_offline.py asserting
  `matched_sensors is None` for a size-less row regardless of
  cache contents.
- Existing C55-01 cache-fallback section augmented with an
  empty-cache (empty list) assertion.
- Gates remain green.
- One commit per change (docstring, size-less test, empty-cache
  augmentation), each GPG-signed with conventional+gitmoji
  message.

## Deferred items reaffirmed (no action this cycle)

- F56-04 / F55-06 / F55-A-01: `_refresh_matched_sensors` helper
  extraction. Re-deferred per cleanup-risk rationale.
- F56-DOC-03: deferred.md sweep. Re-deferred; periodic.
- All carry-overs (F32, F35..F40, F49-04, C10-07, C10-08, F55-04,
  F55-05) remain deferred per their original rationale.
