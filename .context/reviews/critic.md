# Critic Review (Cycle 37) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## Previous Findings Status

All C36 findings fixed. NaN/inf guards are now complete. No regressions.

## New Findings

### CRIT37-01: `derive_spec` still has a NaN propagation path through partially-NaN size tuples

**File:** `pixelpitch.py`, lines 726-733
**Severity:** MEDIUM | **Confidence:** HIGH

The C36 fixes addressed NaN/inf entering through `parse_existing_csv` and `pixel_pitch`. However, `derive_spec` still computes `area = size[0] * size[1]` without checking whether the individual size dimensions are finite. If a `Spec` object is constructed in code (not through CSV) with `size=(nan, 24.0)`, the NaN propagates:

1. `area = nan * 24.0 = nan`
2. `pixel_pitch(nan, 10.0)` returns `0.0` (guarded)
3. `pitch = 0.0` — but the spec has invalid dimensions, not a valid zero-pitch camera

The `0.0` sentinel is ambiguous: it means "invalid input" but is indistinguishable from a legitimate 0.0 pitch. The guard in `pixel_pitch` correctly prevents NaN propagation to the template, but the intermediate `area=nan` is not caught.

While `parse_existing_csv` now rejects NaN (via `_safe_float`), other code paths (direct `Spec` construction in tests, or source parsers) could still produce NaN in size tuples. The `openmvg.fetch` path now guards against inf dimensions, but other source parsers (cined, apotelyt, imaging_resource) use `float()` without `isfinite` checks on the extracted dimensions.

**Fix:** Add `isfinite` validation in `derive_spec` for size dimensions, and consider adding it to source parser float extractions.

---

### CRIT37-02: Source parsers (cined, apotelyt, imaging_resource) still use bare `float()` for extracted dimensions

**File:** `sources/cined.py` line 98, `sources/apotelyt.py` lines 119/126/132, `sources/imaging_resource.py` lines 229/239/249
**Severity:** LOW | **Confidence:** MEDIUM

While `parse_existing_csv` and `openmvg.fetch` now use `_safe_float` / `math.isfinite` guards, the other source parsers still use bare `float()` calls on regex-extracted dimension strings. In theory, a corrupted or adversarial HTML page could contain text that matches the dimension regex with "nan" or "inf", which would produce NaN/inf values in the resulting Spec objects.

However, the probability of this happening in practice is extremely low because:
1. The regex patterns (SIZE_MM_RE, IR_SENSOR_SIZE_RE, PITCH_UM_RE) match numeric patterns like `([\d.]+)` which will not match "nan" or "inf"
2. The IR_MPIX_RE matches `(\d+\.?\d*)` which also cannot match "nan"
3. The data comes from legitimate websites, not adversarial input

The only theoretical risk is from GSMArena's `mpix` extraction which uses `float(mp_match.group(1))` where `mp_match` comes from a regex that matches `([\d.]+)` — again, cannot produce "nan" or "inf".

**Fix:** No immediate fix required. The regex patterns naturally exclude NaN/inf strings. Document this as a defense-in-depth gap.

---

## Summary

- CRIT37-01 (MEDIUM): `derive_spec` propagates NaN area from partially-NaN size tuples
- CRIT37-02 (LOW): Source parsers use bare `float()` but regex patterns exclude NaN/inf strings
