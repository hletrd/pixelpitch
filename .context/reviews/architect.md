# Architect Review (Cycle 38) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes

## Previous Findings Status

ARCH37-01 (derive_spec validation incomplete) fixed by isfinite guard.

## New Findings

### ARCH38-01: `0.0` sentinel value for "invalid" propagates through system as if legitimate — design concern

**File:** `pixelpitch.py` (pixel_pitch), `templates/pixelpitch.html` (template + JS)
**Severity:** MEDIUM | **Confidence:** HIGH

The `pixel_pitch` function returns `0.0` as a sentinel for "invalid input" (NaN, inf, negative, zero). This `0.0` propagates through `derive_spec`, `write_csv`, template rendering, and JS filtering as if it were a legitimate measurement value.

The fundamental issue is that `0.0` is a valid float but an invalid pixel pitch. Using `0.0` as a sentinel conflates "the computation produced zero" with "the input was invalid". The correct sentinel for "unknown/invalid" is `None`, which the template already handles by displaying "unknown".

The C37-02 fix tried to address this by adding `pitch === 0` to the JS filter, which hides zero-pitch rows. But this created a contradiction: the template still renders "0.0 µm" (as if valid) while JS hides it (as if invalid).

**Two-part fix (incremental, not a full refactor):**
1. **Template fix (immediate):** Add `spec.pitch != 0.0` to the pitch condition, so 0.0 renders as "unknown". This aligns template rendering with JS filtering.
2. **`pixel_pitch` refactor (deferred):** Change `pixel_pitch` to return `None` instead of `0.0` for invalid inputs. This requires updating `sorted_by` (which uses `-1` as fallback for None), `write_csv`, `merge_camera_data`, and all related tests. This is a larger change that should be done carefully.

The template fix is small, safe, and addresses the immediate UX issue. The `pixel_pitch` refactor can be deferred.

---

## Summary

- ARCH38-01 (MEDIUM): `0.0` sentinel value propagates as if legitimate — template should render "unknown" for 0.0 pitch/mpix
