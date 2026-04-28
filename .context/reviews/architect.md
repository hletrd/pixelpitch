# Architect Review (Cycle 31) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## Previous Findings Status

ARCH30-01 and ARCH30-02 both fixed in C30. All previous architectural concerns remain deferred.

## New Findings

### ARCH31-01: Spec vs SpecDerived pitch duplication creates consistency risk

**File:** `pixelpitch.py`, `models.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The `Spec` dataclass has a `pitch` field (direct measurement), and `SpecDerived` has a separate `pitch` field (used for display). When `spec.pitch` is set, `derive_spec()` at line 693-694 copies it to `derived.pitch`. When `spec.pitch` is None but area+mpix are known, `derived.pitch` is computed from those.

The fundamental issue is that these two fields can diverge after merge_camera_data. The merge function preserves `spec.pitch` from existing (when new has None) but does not guarantee `derived.pitch` tracks this change. The template and write_csv both read `derived.pitch`, making it the single source of truth for display — but `spec.pitch` is the authoritative measurement.

This is an architectural coupling issue: `derived.pitch` should always be consistent with `spec.pitch` when the latter is set, but the merge logic treats them as independent fields.

**Fix:** After all Spec field preservation in merge_camera_data, recalculate `derived.pitch` to be consistent with the final `spec.pitch`. Specifically: if `spec.pitch` is not None after merge, set `derived.pitch = spec.pitch` regardless of what derived.pitch was computed to be.

---

### ARCH31-02: BOM check duplication across two modules

**File:** `pixelpitch.py` line 276; `sources/openmvg.py` line 67
**Severity:** LOW | **Confidence:** HIGH

The same BOM-stripping logic is duplicated in two files. This is a DRY violation. The BOM handling should be centralized, either in `sources/__init__.py` (as a utility function) or as a shared helper.

**Fix:** Extract `strip_bom(text: str) -> str` into `sources/__init__.py` and call it from both files.

---

## Summary

- ARCH31-01 (MEDIUM): Spec/SpecDerived pitch field duplication creates consistency risk in merge
- ARCH31-02 (LOW): BOM check logic duplicated across pixelpitch.py and openmvg.py
