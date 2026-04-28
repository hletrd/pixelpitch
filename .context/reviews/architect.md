# Architect Review (Cycle 37) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## Previous Findings Status

ARCH36-01 (validation layer incomplete) addressed by C36 fixes. `isfinite` guards now in `pixel_pitch`, `parse_existing_csv`, and `openmvg.fetch`.

## New Findings

### ARCH37-01: Validation layer still incomplete — `derive_spec` does not validate size dimensions

**File:** `pixelpitch.py`, lines 726-733
**Severity:** MEDIUM | **Confidence:** HIGH

The C36 fixes added `isfinite` guards at the data ingestion points (`parse_existing_csv`, `openmvg.fetch`, `pixel_pitch`). However, `derive_spec` — the central computation function — still does not validate its inputs. It trusts that `spec.size` contains finite, positive values.

The validation strategy is "guard at the boundaries" (CSV parser, source fetchers), which works for data entering through those paths. But `derive_spec` is also called from code paths that construct `Spec` objects directly (tests, merge logic, etc.). If any of these paths produce a Spec with NaN/inf/negative size, the validation is bypassed.

A more robust approach would be to also validate at the computation point (`derive_spec`), providing defense-in-depth. This is the same principle as the `pixel_pitch` guard — the function itself should reject invalid inputs, not just rely on callers to do so.

**Fix:** Add `isfinite` validation in `derive_spec` for size dimensions:
```python
if size is not None and spec.mpix is not None:
    if isfinite(size[0]) and isfinite(size[1]) and size[0] > 0 and size[1] > 0:
        area = size[0] * size[1]
    else:
        size = None
        area = None
```

---

## Summary

- ARCH37-01 (MEDIUM): Validation layer incomplete — `derive_spec` should validate size dimensions
