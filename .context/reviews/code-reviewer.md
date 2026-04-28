# Code Reviewer — Cycle 54

**HEAD:** `93851b0`
**Scope:** Whole repository, with focus on `pixelpitch.py`,
`sources/`, and `tests/test_parsers_offline.py`.

## Inventory

- `pixelpitch.py` (1378 lines)
- `models.py` (27 lines)
- `sources/__init__.py` (110)
- `sources/apotelyt.py` (183)
- `sources/cined.py` (149)
- `sources/digicamdb.py` (32)
- `sources/gsmarena.py` (269)
- `sources/imaging_resource.py` (302)
- `sources/openmvg.py` (129)
- `tests/test_parsers_offline.py` (2283)
- `tests/test_sources.py` (111)

## Findings

### F54-01 — `_load_per_source_csvs` does not re-derive sensor matches against current `sensors.json` — LOW

- **File:** `pixelpitch.py:1028-1053`
- **Severity:** LOW | **Confidence:** MEDIUM
- **Description:** `_load_per_source_csvs` reads each
  `dist/camera-data-{source}.csv` via `parse_existing_csv` and only
  clears the per-row `id`. The `matched_sensors` column is parsed from
  the file as-is. Per-source CSVs were written by `fetch_source` (line
  1296-1305) via `derive_specs` against whatever `sensors.json` was
  current at that time. If `sensors.json` is later edited (sensor
  renamed, removed, or megapixel list trimmed) but the per-source CSV
  is not regenerated, the stale matched_sensors will be merged into
  the final dataset by `merge_camera_data` and will surface in
  `dist/camera-data.csv` and the rendered HTML.
- **Repro path:**
  1. Run `python pixelpitch.py source openmvg` to write
     `dist/camera-data-openmvg.csv` with `matched_sensors=IMX455`.
  2. Edit `sensors.json` to rename `IMX455` to `IMX455A`.
  3. Run `python pixelpitch.py html dist`. The freshly rendered
     output will still show the old `IMX455` token because
     `_load_per_source_csvs` did not re-match.
- **Fix:** After parsing, call `derive_spec(d.spec, sensors_db)`
  (lazy-loaded) to refresh `matched_sensors`, OR explicitly clear
  `matched_sensors=None` so `merge_camera_data` falls back to
  existing CSV matches. Document the chosen semantics.

### F54-02 — `merge_camera_data` overwrites a valid new id with a None existing id — LOW (theoretical)

- **File:** `pixelpitch.py:524`
- **Severity:** LOW | **Confidence:** LOW
- **Description:** `new_spec.id = existing_spec.id` blindly copies
  the id even when `existing_spec.id is None`. In current call sites
  this is harmless because line 623-624 reassigns sequential ids
  before `write_csv`. Defense-in-depth would be
  `new_spec.id = existing_spec.id if existing_spec.id is not None
  else new_spec.id`.
- **Severity:** LOW (mitigated by sequential reassignment).

## Final sweep — commonly missed issue types

- Logic bugs: F54-01 (stale matched_sensors).
- Edge cases: F54-02 (None id overwrite).
- Race conditions: N/A (single-threaded).
- Error handling: parse_existing_csv broad except is intentional and
  documented.
- Invariant violations: F54-01 violates the implicit "rendered output
  reflects current sensors.json" invariant.
- Test gaps: no test asserts that per-source CSV parse + merge
  refreshes matched_sensors.

## Confirmed clean

- All previously flagged issues C8-C53 still fixed at HEAD.
- Both gates pass at HEAD.
