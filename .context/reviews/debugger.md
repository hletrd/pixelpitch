# Debugger Review (Cycle 21) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## D21-01: C20-03 regression — SpecDerived fields stale after merge

**Severity:** HIGH | **Confidence:** HIGH (reproduced)

The C20-03 fix introduced a subtle data inconsistency. When `merge_camera_data` preserves `spec.type`, `spec.size`, or `spec.pitch` from existing data, it only sets the `Spec` attribute. The corresponding `SpecDerived` attributes (`size`, `area`, `pitch`) are NOT updated. This creates an internal inconsistency where `new_spec.spec.size != new_spec.size`.

**Failure mode:**
1. Camera "Test Cam" exists with `spec.size=(5.0, 3.7)`, `derived.size=(5.0, 3.7)`, `derived.area=18.5`
2. New source has `spec.size=None`, `derived.size=None`, `derived.area=None`
3. After merge: `spec.spec.size=(5.0, 3.7)`, `spec.size=None`, `spec.area=None`
4. The `spec.spec.size` and `spec.size` disagree — an internal invariant violation
5. Template reads `spec.size` -> shows "unknown"

**Root cause:** The C20-03 fix was implemented at the wrong layer. It should preserve `SpecDerived` fields alongside `Spec` fields, or re-derive after preservation.

---

## D21-02: Sony RX/DSC/HX/WX/TX/QX naming — latent bug from C20-02 incomplete fix

**Severity:** MEDIUM | **Confidence:** HIGH (reproduced)

The C20-02 fix only addressed the FX series. The same `.title()` issue affects all Sony multi-letter uppercase series. These cameras will be misnamed and may create dedup failures with other sources.

**Failure mode:** Camera "Sony RX100 VII" from IR is named "Sony Rx100 Vii". Apotelyt names it "Sony RX100 VII". Merge treats them as different cameras -> duplicate entry.

---

## Summary

- D21-01 (HIGH): SpecDerived fields stale after merge — internal invariant violation
- D21-02 (MEDIUM): Sony RX/DSC/HX/WX/TX/QX naming — same class of bug as C20-02
