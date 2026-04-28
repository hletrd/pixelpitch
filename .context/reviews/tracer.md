# Tracer Review (Cycle 36) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

TR35-01 (derive_spec ValueError on negative area) fixed. TR35-02 (BOM literal) fixed.

## New Findings

### TR36-01: NaN propagation trace — silent data corruption from CSV to rendered page

**Files:** `pixelpitch.py`, `sources/openmvg.py`, `templates/pixelpitch.html`
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Corrupted CSV cell contains `nan` or `inf` string
2. `parse_existing_csv` calls `float("nan")` → produces `float('nan')`
3. `derive_spec` computes `area = nan * 24.0 = nan`
4. Since `spec.pitch is None`, calls `pixel_pitch(nan, 10.0)`
5. Guard `mpix <= 0 or area <= 0` evaluates to `False` (because `nan <= 0` is `False`)
6. `sqrt(nan / 10e6) = nan` — returns `nan` instead of `0.0`
7. `write_csv` writes `nan` to CSV — `f"{nan:.2f}"` = `"nan"`
8. Template renders `{{ nan|round(1) }}` = `nan` → visible "nan µm"
9. JS `isInvalidData`: `parseFloat("nan") || 0 = 0` → not flagged → row visible
10. User sees "nan µm" in the table

**Competing hypothesis:** Could NaN/inf actually reach the pipeline?
- `parse_existing_csv` accepts any `float()` string including `"nan"`, `"inf"` — YES
- Source parsers use regex-extracted values which never produce NaN from normal text — UNLIKELY from sources
- Manual CSV edits could introduce NaN — YES
- `derive_spec` arithmetic with very large floats produces inf — YES (e.g., `1e308 * 1e308`)

**Fix:** Add `math.isfinite()` guards in `pixel_pitch`, `parse_existing_csv`, and `openmvg.fetch`.

---

## Summary

- TR36-01 (MEDIUM): NaN propagates silently from CSV through to rendered "nan µm" — traced end-to-end
