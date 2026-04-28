# Debugger — Cycle 50

**Date:** 2026-04-29
**HEAD:** `ed45eed`

## Latent bug surface scan

I traced every public function in `pixelpitch.py` and the source modules for failure modes:

- `derive_spec`: Exhaustive branch coverage for None size, None type, NaN/inf size, zero/negative pitch, zero/negative mpix. All paths produce well-defined output.
- `merge_camera_data`: All field-preservation paths post-C22-01/C46-01 correctly chained as independent `if`s.
- `parse_existing_csv`: Padded value lists prevent IndexError. has_id detection is robust.
- `write_csv`: All format strings guarded by `is not None and isfinite(...) and ... > 0`.
- `_select_main_lens`: Post-C45-01 decimal-MP split is correct.
- `_create_browser`: Cross-platform paths handled.
- `http_get`: Retry logic bounded; final failure returns None.

## New debugger findings this cycle

### F50-03 — matched_sensors `;` delimiter has no escape (LOW / MEDIUM)
- File: `pixelpitch.py:373` (split) and `pixelpitch.py:920-922` (join)
- Currently dormant: no sensor name in `sensors.json` contains `;`. If a future sensors.json entry includes `;` in its key (e.g. `"IMX455;v2"`), the round-trip would silently fragment it into `["IMX455", "v2", ...]`.
- Defense: assertion in `write_csv` that no element of `matched_sensors` contains `;`, OR migrate to a safer delimiter.

## Adjacent debt

F50-01 (`git pull --rebase || true` mask) means rebase-conflict failure modes are obscured in CI logs. Not a runtime bug, a diagnosability gap.

## Summary

One new dormant defect (F50-03). All previously suspicious surfaces remain remediated.
