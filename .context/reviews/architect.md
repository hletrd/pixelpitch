# Architect Review (Cycle 21) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## A21-01: SpecDerived stale fields reveal a deeper data model design issue

**Severity:** MEDIUM | **Confidence:** HIGH

The C21-01 bug (SpecDerived fields stale after merge) reveals a deeper architectural issue: the `Spec` / `SpecDerived` split creates two sources of truth for the same data. `Spec` holds input values; `SpecDerived` holds computed values. When merge modifies `Spec` fields, `SpecDerived` fields become stale.

The current fix (preserving both Spec and SpecDerived fields) is pragmatic but perpetuates the dual-source-of-truth problem. A more robust approach would be to re-derive `SpecDerived` from `Spec` after any merge operation. However, this has a subtle complication: `derive_spec` computes `SpecDerived.size` from `spec.type` when `spec.size` is None, which could produce different results than the preserved `spec.size`.

**Recommendation:** The pragmatic fix (preserving SpecDerived fields) is correct for now. If the source diversity grows, consider making `SpecDerived` a pure computed view with no independent state, so it's always consistent with `Spec`.

---

## Summary

- A21-01 (MEDIUM): Spec/SpecDerived dual-source-of-truth design issue — pragmatic fix is acceptable
