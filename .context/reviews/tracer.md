# tracer Review (Cycle 52)

**Date:** 2026-04-29
**HEAD:** 331c6f5

## Causal trace: F52-01 — year column drop on Excel-roundtrip

1. CI runs `python pixelpitch.py render` →
   `render_html(output_dir)` (`pixelpitch.py:992`).
2. `load_csv(output_dir)` reads `dist/camera-data.csv`
   (`pixelpitch.py:256`).
3. `parse_existing_csv(previous_csv)` enters the row loop
   (`pixelpitch.py:310`).
4. `year_str = values[9]` (or `values[8]` for no-id rows).
5. `int(year_str)` (`pixelpitch.py:368`) — if Excel saved the file with
   `"2023.0"`, this raises ValueError.
6. The except branch is `pass`; year stays None
   (`pixelpitch.py:371`).
7. `merge_camera_data` then sees `existing_spec.spec.year = None` →
   the year-preservation branch
   (`pixelpitch.py:495`) does NOT fire (it triggers only when new is
   None and existing is non-None).
8. `write_csv` then emits an empty year column for all edited rows.

Single hypothesis confirmed; no competing flow.

## Causal trace: matched_sensors stability

After cycle 51, the chain
`derive_spec → merge_camera_data → write_csv → parse_existing_csv →
merge_camera_data` for matched_sensors is verified idempotent under:

- whitespace (cycle 51 fix)
- duplicates (cycle 51 fix)
- `;`-injection (cycle 50 fix)
- missing sensors_db (cycle 46 fix)

No new trace-level concerns this cycle.

## Verdict

F52-01 is real, single-cause, single-fix.
