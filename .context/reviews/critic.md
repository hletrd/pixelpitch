# Critic — Cycle 54

**HEAD:** `93851b0`

## Multi-perspective critique

### Correctness perspective

After the C46-C53 round-trip hardening, the CSV parse/write/merge
chain is mostly defensive. The remaining gap is **F54-01**:
`_load_per_source_csvs` trusts the per-source CSV's
`matched_sensors` column verbatim. The codebase has two competing
intents for per-source CSVs:

1. **Cache**: regenerate freely from upstream + sensors.json on each
   `python pixelpitch.py source <name>` run.
2. **Authoritative store**: the CSV is the truth, downstream consumers
   trust it.

Today the code does (1) on write but (2) on read.

### Maintainability perspective

`pixelpitch.py` is 1378 lines (no growth this cycle). Deferred F32
(monolith refactor) still applicable.

### Doc/code consistency

- `_safe_int_id` docstring (lines 318-330) was updated in C53 to
  match the implementation. Re-checked, accurate.
- `merge_camera_data` docstring (lines 475-497) does NOT mention
  the matched_sensors preservation behavior added in C46. Cosmetic.

## Findings

### F54-01 (consensus, see code-reviewer): per-source CSV matched_sensors are stale — LOW

This is an architectural inconsistency: declare per-source CSVs as
caches and refresh on load, OR formalize them as the source of truth.

## Final sweep

No additional new findings.
