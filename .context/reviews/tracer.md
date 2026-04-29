# Tracer Review (Cycle 57)

**Reviewer:** tracer
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## Causal flow inventory

### Flow 1: CSV → SpecDerived (parse path)

1. `load_csv` reads `dist/camera-data.csv`.
2. `parse_existing_csv` constructs `SpecDerived` directly:
   - `width, height` from cols 4,5 (or 3,4 if no id)
   - `area` from col 6 (or 5)
   - `mpix` from col 7 (or 6)
   - `pitch` from col 8 (or 7)
   - `year` from col 9 (or 8)
   - `matched_sensors` from col 10 (or 9)
3. SpecDerived is returned to `merge_camera_data`.

### F57-T-01: `area` is set independently of `width*height` — LOW

- **Files:** `pixelpitch.py:413-426`
- **Detail:** Same as F57-CR-01 from the trace angle. The parse path
  encodes `(width, height) → size, area_col → area` as two
  independent reads, but `derive_spec` enforces
  `area = width * height`. The two paths agree on freshly-derived
  Specs but disagree on hand-edited CSVs.
- **Suggested:** Recompute `area = width * height` in
  `parse_existing_csv` when both are present.

### Flow 2: per-source CSV → SpecDerived (cache fallback)

1. `_load_per_source_csvs` iterates SOURCE_REGISTRY.
2. For each, parse_existing_csv → list[SpecDerived].
3. For each row:
   - if size present + sensors_db ok → re-match
   - if size present + sensors_db empty → preserve cache
   - if size None → matched_sensors = None
4. Returned to `merge_camera_data`.

All branches pinned by tests after C54/C55/C56.

### Flow 3: Spec → SpecDerived → CSV (derive path)

1. Source scrapers produce `Spec`.
2. `derive_spec` computes size, area, pitch, matched_sensors.
3. write_csv emits all 11 columns.

Verified: derive_spec sets area = w*h consistently. No bug here.

## Competing hypotheses for F57-T-01

- H1 (preferred): `parse_existing_csv` should recompute area from
  width/height to match `derive_spec`.
- H2: Trust the column for human override scenarios. **Rejected:**
  `derive_spec` is the canonical source of truth, and a human
  override of area without updating width/height is more likely a
  bug than intent.

## Confidence summary

- 1 LOW actionable (F57-T-01 = F57-CR-01).
- All other flows trace cleanly.
