# Debugger — Cycle 54

**HEAD:** `93851b0`

## Latent bug surface scan

### F54-D01 — same as F54-01 (stale matched_sensors) — LOW

Verified against the merge logic. Not a crash bug; data drift only.

### F54-D02 — `merge_camera_data` line 537 references uninitialized field on first miss — re-verified safe

`new_spec.size = existing_spec.size` is guarded by the outer
`if new_spec.spec.size is None` check on line 528. existing_spec is
always defined inside the `if key in existing_by_key` block. Safe.

### F54-D03 — None propagation through `derive_spec` — re-verified safe

`derive_spec` correctly returns `matched_sensors=None` when
`sensors_db` is unavailable or `size` is unknown. Sentinel
distinguishes "not checked" from "checked, found nothing".

## Failure-mode sweep

- `parse_existing_csv` broad except: lines 454-457 catches all and
  prints. Stable since C49.
- `_safe_int_id` and `_safe_year` are guarded against
  Excel-hand-edits; both have `isfinite` + range guards.
- `pixel_pitch` returns 0.0 for invalid inputs; `derive_spec`
  converts 0.0 to None.
- `match_sensors` requires positive width/height; guarded.

## No new crash bugs.
