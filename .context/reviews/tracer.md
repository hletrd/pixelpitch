# Tracer Review (Cycle 37) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## Previous Findings Status

TR36-01 (NaN propagation from CSV to page) fixed. All `isfinite` guards verified working.

## New Findings

### TR37-01: `derive_spec` area=nan → `write_csv` → "nan" string → `_safe_float` → None: asymmetric round-trip

**Files:** `pixelpitch.py` (derive_spec, write_csv, parse_existing_csv)
**Severity:** LOW | **Confidence:** HIGH

**Causal trace:**
1. A `Spec` is constructed with `size=(nan, 24.0)` (from source code or in-memory)
2. `derive_spec` computes `area = nan * 24.0 = nan`
3. `write_csv` formats `f"{nan:.2f}"` = `"nan"` into the CSV area column
4. On next run, `parse_existing_csv` calls `_safe_float("nan")` → returns `None`
5. The area changes from `nan` to `None` across a write-read cycle

This is a data integrity asymmetry. The first write produces a CSV with "nan" in it; the read-back produces `None`. The CSV file is "corrupted" with a "nan" string on the first pass, then "corrected" to empty on the second pass.

**Competing hypothesis:** Can NaN area actually reach `write_csv`?
- `parse_existing_csv` now rejects NaN via `_safe_float` — NO from CSV
- Source parsers use `[\d.]+` regex which can't match "nan" — NO from sources
- `derive_spec` can produce NaN area from NaN size — YES from code-constructed Spec
- `openmvg.fetch` now guards inf dimensions — NO from openmvg
- Other source parsers use bare `float()` but regex excludes NaN — NO

The only remaining path is `derive_spec` computing `nan * valid = nan` from a code-constructed Spec with NaN size dimensions. This can only happen if a source parser somehow produces a NaN dimension, which the regex analysis shows is impossible from real HTML data.

**Fix:** Add `isfinite` guard in `derive_spec` for size dimensions. This closes the last remaining NaN propagation path.

---

## Summary

- TR37-01 (LOW): `derive_spec` area=nan writes "nan" to CSV, which round-trips as None
