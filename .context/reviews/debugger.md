# Debugger — Cycle 60 (Orchestrator Cycle 13)

**Date:** 2026-04-29
**HEAD:** `a0cd982`

## Latent Bug Surface Scan

Replayed the following hypothetical failure modes against the code:

1. **Empty/corrupt sensors.json** — handled (F55-01: cache fallback,
   line 1142-1145).
2. **Stale matched_sensors cache after sensor rename** — refreshed
   on next render (F54-01: re-match in `_load_per_source_csvs` and
   `merge_camera_data`).
3. **Excel-coerced floats in id/year** — handled (F51, F52, F53).
4. **Hand-edited blank rows** — handled (line 388, skip rows with
   no non-empty cells).
5. **UTF-8 BOM from Excel CSV save** — handled (line 375,
   `strip_bom`).
6. **Inf/NaN/zero/negative numeric columns** — handled at all three
   boundaries (derive_spec, parse_existing_csv, write_csv).
7. **Sensor names containing ';' delimiter** — handled (line
   1072-1077, drop with warning).

## Cycle 60 New Findings

### F60-D-01 (informational): `derive_spec` returns `size=None,
area=None` when input size is non-finite, but the original `spec.size`
is unchanged

- **File:** `pixelpitch.py:902-904`
- **Detail:** When the input `spec.size` has `(inf, 24)`, `derive_spec`
  sets `size = None` and `area = None`, but `spec.size` (the underlying
  Spec field) is not modified. This is correct — Spec is the raw input,
  SpecDerived is the cleaned output. But it means a downstream caller
  that reads `spec.size` directly (rather than `derived.size`) would
  still see the `(inf, 24)` value. No such caller exists in the
  codebase today; the template, write_csv, and JSON-LD all read
  `derived.size`. Documenting this contract more clearly in
  `derive_spec`'s docstring would aid future maintainers.
- **Severity:** LOW. **Confidence:** MEDIUM (DOC-only).
- **Disposition:** Defer (no live bug; `derive_spec` docstring already
  states "NaN or infinite dimensions in `spec.size` are treated as
  unknown (size and area set to None)" — the contract is documented,
  just not the asymmetry between Spec and SpecDerived).

## Summary

No actionable debugger findings for cycle 60.
