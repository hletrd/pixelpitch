# Tracer — Cycle 54

**HEAD:** `93851b0`

## Causal trace — stale matched_sensors propagation

Hypothesis: matched_sensors emitted in `dist/camera-data.csv` may
not reflect the current state of `sensors.json` when per-source
CSVs are stale.

### Forward trace

1. User runs `python pixelpitch.py source openmvg` at time T0
   when `sensors.json` defines sensor `S0` matching some camera C.
2. `fetch_source` → `derive_specs` → `match_sensors(...)` →
   `matched_sensors=["S0"]` → `write_csv` → file
   `dist/camera-data-openmvg.csv` row for C contains `S0`.
3. At time T1 > T0, `sensors.json` is edited: `S0` removed.
4. User runs `python pixelpitch.py html dist`.
5. `render_html` → `_load_per_source_csvs` → `parse_existing_csv`
   loads C's row → matched_sensors=["S0"] (from disk).
6. `merge_camera_data` adds C as a new_spec. C's existing entry in
   `dist/camera-data.csv` may have matched_sensors=[] (if the
   previous render-after-edit cleared it). new_spec wins because
   the merge gives new precedence (line 547-585), so `S0` is kept
   even though it no longer exists in `sensors.json`.
7. `write_csv(merged)` → `dist/camera-data.csv` row for C contains
   `S0`. Output diverges from sensors.json.

### Competing hypothesis 1: derive_spec is called somewhere

Searched for `derive_spec` calls outside `derive_specs` and
`fetch_source`. Found only `tests/test_parsers_offline.py` mocks.
Not called on parsed-from-CSV data. Hypothesis 1 rejected.

### Competing hypothesis 2: merge_camera_data re-matches

Lines 602-617 re-match for **existing-only** cameras (those in
existing but not new). C is in new (came from per-source CSV), so
this path does not run for C. Hypothesis 2 rejected.

### Competing hypothesis 3: parse_existing_csv strips stale entries

Lines 437-445 only filter empty/whitespace-only tokens. Sensor
name `"S0"` is not empty, so it survives. Hypothesis 3 rejected.

## Conclusion

F54-01 confirmed. Severity LOW because:
- Sensor renames/removals are infrequent.
- The matched_sensors column is informational, not load-bearing.
- Self-healing on next per-source fetch.

## Final sweep

No other suspicious flows in current diff.
