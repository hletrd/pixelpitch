# Tracer Review (Cycle 58, orchestrator cycle 11)

**Date:** 2026-04-29
**HEAD:** `aef726b`

## Causal trace: `source --limit -1` execution path

1. `main()` parses argv. `--limit` triggers
   `int(args[i + 1])` → `limit = -1`.
2. No range check on `limit`. Control flows to
   `fetch_source(name, limit, out_dir)`.
3. `fetch_source` builds `kwargs["limit"] = -1` and calls
   `module.fetch(**kwargs)`.
4. Per-source `fetch`:
   - `apotelyt`: `urls[:limit]` → `urls[:-1]` drops last URL.
   - `cined`: same, `urls[:limit]` truncates.
   - `gsmarena`: same, `slugs[:limit]` truncates.
   - `openmvg`: `if limit is not None and i >= limit: break`
     — `i >= -1` is true at `i=0`, breaks immediately,
     returns empty list.
5. `derive_specs([])` returns `[]`.
6. `write_csv([], out_file)` writes only the header row.
7. Process exits 0. **No error signal to the user.**

## Competing hypotheses considered

- H1 (rejected): "limit < 0 is intended as 'remove last N'."
  No documentation supports this. The README and `--help`
  describe `--limit N` as "fetch up to N records." Negative
  is undefined.
- H2 (rejected): "argparse-equivalent error elsewhere."
  Manual loop, no argparse. No error path exists.
- H3 (confirmed): "missing input validation; silent
  truncation / empty-file." Confirmed by trace + repro.

## New findings

- F58-T-01: corroborates `code-reviewer.F58-CR-01` and
  `critic.F58-CRIT-01`. Same root cause, same fix.

## Summary

Single causal chain from missing CLI validation to silent
empty CSV. Fix is one line in `main()`'s `--limit` branch.
