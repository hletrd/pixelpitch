# Tracer — Cycle 53

**Date:** 2026-04-29
**HEAD:** `1c968dd`

## Hypothesis under test

H1: F53-01 (`_safe_int_id("1e308")` returns 309-digit int) propagates
into the committed CSV.

## Causal chain

1. Excel rewrites integer ids in scientific notation on save.
2. CI loads the CSV via `load_csv` (line 256-265).
3. `parse_existing_csv` calls `_safe_int_id(values[0])` (line 379).
4. `int("1.0E+308")` raises ValueError.
5. Falls through: `float("1.0E+308") ≈ 1e308`. `isfinite` → True.
6. `int(1e308)` → 309-digit big-int.
7. `record_id = <309-digit int>` → stored on `SpecDerived.id`.
8. `merge_camera_data` line 516: `new_spec.id = existing_spec.id`.
9. `main()` line 1293: `d.id = i` reassigns sequential ids before
   `write_csv` is called.

## Counter-hypothesis

The 309-digit value is overwritten before write_csv. So the bad id
never reaches `dist/camera-data.csv`. Impact narrows to "transient
in-memory garbage on the parsed object during merge", not "bad
data persisted to disk."

Still a finding because:
- The original id-to-row mapping for that row is permanently lost
  (parsed bad → reassigned new id, no continuity with prior CSV).
- Asymmetric with `_safe_year`'s range guard (process inconsistency).

## Verdict

H1 partially confirmed. LOW severity. Agreement with code-reviewer
F53-01.

No other suspicious flows surfaced.
