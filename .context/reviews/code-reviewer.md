# Code-Reviewer Review (Cycle 59, orchestrator cycle 12)

**Date:** 2026-04-29
**HEAD:** `fa0ae66` (after C58-01 plan-completed marker)

## Status

Re-ran a full pass over `pixelpitch.py`, `sources/*.py`, `models.py`,
and `tests/test_parsers_offline.py`.

Both gates pass at HEAD:

- `python3 -m flake8 .` -> 0 errors.
- `python3 -m tests.test_parsers_offline` -> all sections green.

## New findings

### F59-CR-01 (defensive-parity, LOW): `write_csv` width/height columns lack the isfinite/positive guards used for area/mpix/pitch

- **File:** `pixelpitch.py:1018-1019`
- **Severity:** LOW. **Confidence:** HIGH.
- **Detail:** Lines 1020-1022 already guard
  `area_str`/`mpix_str`/`pitch_str` against non-finite or
  non-positive values:

  ```python
  area_str = f"{derived.area:.2f}" if derived.area is not None and isfinite(derived.area) and derived.area > 0 else ""
  mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and isfinite(spec.mpix) and spec.mpix > 0 else ""
  pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and isfinite(derived.pitch) and derived.pitch > 0 else ""
  ```

  But lines 1018-1019 only check truthiness:

  ```python
  width_str = f"{derived.size[0]:.2f}" if derived.size else ""
  height_str = f"{derived.size[1]:.2f}" if derived.size else ""
  ```

  `derived.size` is `Optional[Tuple[float, float]]`. A non-empty
  tuple is truthy regardless of element values, so a hypothetical
  `(0.0, 0.0)`, `(-1.0, -1.0)`, `(inf, inf)`, or `(nan, nan)`
  tuple would write `"0.00,0.00"` / `"-1.00,-1.00"` / `"inf,inf"`
  / `"nan,nan"` to the CSV. Today `derive_spec` (lines 900-906)
  filters non-finite/non-positive size and `parse_existing_csv`
  rejects non-positive dimensions (lines 430-433), so the
  pathological tuple cannot reach `write_csv` via the normal
  path. The fix hardens the contract symmetrically with the
  area/mpix/pitch guards, so a future `derive_spec` regression
  does not leak `inf`/`nan`/`0.00`/`-1.00` strings into the
  artifact CSV (which would then be parsed back on the next
  build via `parse_existing_csv` and either rejected as a row
  parse error or coerced to None - but the artifact-on-disk
  state would be visibly broken in the meantime).
- **Failure scenario:** A future derive_spec refactor (e.g.,
  removing the line-900 `isfinite` guard, or adding a new
  size-derivation path that bypasses the guard) silently leaks
  `inf` or `0.0` size into `dist/camera-data.csv`. The next
  build's `parse_existing_csv` rejects the row (width<=0 -> None,
  height<=0 -> None, so size becomes None and the row's
  `derived.size` is lost). Downstream consumers see "unknown
  sensor size" for an entry that previously had measured data.
- **Fix:** mirror the area/mpix/pitch guard:

  ```python
  if (derived.size
      and isfinite(derived.size[0]) and derived.size[0] > 0
      and isfinite(derived.size[1]) and derived.size[1] > 0):
      width_str = f"{derived.size[0]:.2f}"
      height_str = f"{derived.size[1]:.2f}"
  else:
      width_str = ""
      height_str = ""
  ```

  Add a regression test pinning the new behavior alongside
  `test_write_csv_nonfinite_guards`.

### F59-CR-02 (informational, LOW): `_load_per_source_csvs` "missing" log line is noisy on first build

- **File:** `pixelpitch.py:1085`
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Detail:** On a fresh `dist` directory, every registered
  source CSV prints "missing ... (skipped)". Wording could be
  softer ("no cached CSV at ...") to avoid misreading as a
  warning. Same severity class as F58-DOC-* informational fixes.
- **Disposition:** Defer (informational, no behavior change).

## Cycle 1-58 carry-over

All previous fixes confirmed still working at HEAD `fa0ae66`.
No regressions vs cycle 58.
