# Debugger Review (Cycle 56)

**Reviewer:** debugger
**Date:** 2026-04-29
**HEAD:** `e8d5414`

## Latent bug surface

### F56-D-01 (resolved): C55-01 cache-fallback discrepancy — RESOLVED

- The cycle 55 finding (sensors_db-empty path drops matched_sensors
  cache) is fixed at `pixelpitch.py:1073-1087`. Verified with the
  new `_load_per_source_csvs cache fallback` test section.

### F56-D-02 (false alarm): `pitch != spec.pitch` float compare

- Same as F55-D-02. Intentional re-derive from canonical pitch.

### F56-D-03 (false alarm): `values[10]` bounds on padded row

- Same as F55-D-03. `if has_id` row is padded to 10; conditional
  indexing for column 10. Confirmed safe.

### F56-D-04 (informational): `_load_per_source_csvs` size-less branch is reachable when source CSV row has empty width/height cells

- **File:** `pixelpitch.py:1084-1087`
- **Detail:** A per-source CSV row with empty width/height parses
  to `size = None`. Branch fires `matched_sensors = None`. This
  drops a parsed cache value. Intentional per the docstring's
  "size unknown means not checked" contract, matching `derive_spec`.
- **Severity:** N/A (intentional). **Confidence:** HIGH.

## No new latent bugs this cycle.
