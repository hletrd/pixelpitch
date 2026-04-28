# Tracer — Cycle 50

**Date:** 2026-04-29
**HEAD:** `ed45eed`

## Causal traces

### Trace A: matched_sensors round-trip integrity

- Source: `derive_spec` (`pixelpitch.py:816-823`) sets `matched_sensors` to `None` when sensors_db unavailable, `[]` when consulted-but-empty, or a sorted list of sensor names.
- Sink-write: `write_csv` (`pixelpitch.py:920-922`) joins with `";"` when truthy, else writes `""`.
- Sink-read: `parse_existing_csv` (`pixelpitch.py:373`) yields `[]` from `""` and `[s for s in str.split(";") if s]` from non-empty.
- Round-trip: `None` → `""` → `[]` (loses the "not consulted" signal). This is OK because `merge_camera_data:540-545` re-matches existing-only cameras using the live sensors_db, so the `None`-vs-`[]` distinction is reconstructed at merge time. **No bug**.
- However: if a sensor name ever contains `;`, the join/split round-trip silently fragments it. **F50-03**.

### Trace B: CI rebase-mask failure paths

- Step "Commit and push camera data" runs `git pull --rebase || true && git push`.
- Hypothesis 1: rebase succeeds → push succeeds. Workflow green. Common path.
- Hypothesis 2: rebase fails → `|| true` masks → `git push` fails on non-fast-forward → workflow step red.
- Hypothesis 3: rebase fails → `|| true` masks → `git push` succeeds (concurrent commit landed cleanly without conflict). Workflow green; local working tree was in mid-rebase state but is thrown away by the workflow runner exit. Idempotent because next month's run will re-add files.
- The masking obscures Hypothesis 2 from the workflow logs — operator must read both the rebase and push step outputs to diagnose. **F50-01**.

## Findings

- **F50-01** — see Trace B
- **F50-03** — see Trace A

## Confidence Summary
- HIGH on both traces' logic.
