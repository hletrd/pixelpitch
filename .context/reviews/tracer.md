# tracer Review (Cycle 51)

**Date:** 2026-04-29
**HEAD:** 3b35dcc

## Causal trace of `matched_sensors` through the pipeline

1. **Source:** `derive_spec` (`pixelpitch.py:817`) calls `match_sensors` → `List[str]`.
2. **Persistence:** `write_csv` (`pixelpitch.py:925-937`) joins on `;`, dropping any element
   containing `;` (with warning).
3. **Re-read:** `parse_existing_csv` (`pixelpitch.py:373`) splits on `;`, filters empty
   strings (no whitespace strip).
4. **Merge:** `merge_camera_data` preserves existing matched_sensors when new is None
   (`pixelpitch.py:512-513`).

### Hypothesis A: end-to-end round-trip is symmetric for clean data
- Verified by cycle-50 round-trip test. PASS.

### Hypothesis B: external CSV with whitespace breaks symmetry
- Not currently triggered (write_csv emits no whitespace).
- Latent fragility; aligns with code-reviewer F51-01.

### Hypothesis C: matched_sensors=None vs [] semantic distinction is preserved
- Trace: `parse_existing_csv` always produces `[]` for empty `sensors_str` (line 373).
  `derive_spec` produces `None` when `size` is missing (line 823) and a list otherwise.
  `merge_camera_data` uses the None vs not-None distinction for preservation logic.
- After CSV round-trip, an originally-`None` field becomes `[]`. The merge does NOT
  preserve existing matched_sensors when new is `[]`. By design — see lines 510-513.

No new actionable issues. The `matched_sensors` causal flow is consistent.
