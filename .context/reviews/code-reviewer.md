# Code Review (Cycle 24) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-23 fixes, focusing on NEW issues

## Previous Findings Status

All previous findings confirmed addressed. C23-01 (body-category fallback tests) implemented and verified.

## New Findings

### CR24-01: `merge_camera_data` SpecDerived.size can become inconsistent with Spec.size after merge

**File:** `pixelpitch.py`, lines 405-428
**Severity:** MEDIUM | **Confidence:** HIGH

When a new spec has `spec.size` set (not None) but `derived.size` is None (which shouldn't happen in normal flow but can occur if `derive_spec` was called with inconsistent inputs), the merge logic only preserves `derived.size` from existing when `new_spec.size is None`. However, the condition checks `new_spec.size` (the Spec field) rather than `new_spec.size` (the SpecDerived field). Due to Python's attribute resolution, `new_spec.size` refers to the SpecDerived field since it shadows `spec.size`.

This is actually correct by accident — `new_spec.size` on a SpecDerived object accesses `SpecDerived.size` (which shadows `Spec.size`). But there is a subtle inconsistency: after the merge, `new_spec.spec.size` may differ from `new_spec.size` if only one was preserved. This can happen when:
- New data provides `spec.size` (the Spec attribute) but not `derived.size` (which is computed from `spec.size` or `spec.type`)
- OR vice versa: `derived.size` is set but `spec.size` is None

The `derive_spec` function always keeps them in sync (sets `derived.size` from `spec.size` or `spec.type`), so in normal flows they stay consistent. But after merge, if only one path preserves, they can diverge.

**Concrete failure scenario:** A camera is fetched by source A with `spec.size=(35.9, 23.9)` and `derived.size=(35.9, 23.9)`. Source B provides the same camera but with `spec.size=None` and `derived.size=None`. After merge, both are preserved from existing, so they stay consistent. But if source B somehow has `spec.size=(35.9, 23.9)` but `derived.size=None` (an edge case in the data pipeline), the merge would NOT preserve `derived.size` from existing because `new_spec.size` (the SpecDerived field) would be None... wait, no — `new_spec.size` IS the SpecDerived field. If it's None, the preservation kicks in. So this is correct.

Actually, on re-analysis: the merge is correct. The real concern is the shadowing — `SpecDerived.size` shadows `Spec.size`, making `spec.spec.size` vs `spec.size` confusing. This is a maintainability issue, not a correctness bug. Downgrading.

**Revised Severity:** LOW | **Confidence:** MEDIUM

**Fix:** Consider renaming `SpecDerived.size` to `SpecDerived.computed_size` or adding a property to clarify the relationship. This is purely a readability/maintainability concern.

---

### CR24-02: `_parse_fields` in imaging_resource.py uses `rstrip("</")` which strips individual chars, not the string

**File:** `sources/imaging_resource.py`, line 95
**Severity:** LOW | **Confidence:** HIGH

The line `value = html_lib.unescape(value).strip().rstrip("</")` uses `rstrip("</")` which strips any trailing `<`, `/`, or `"` characters individually, not the string `"</"`. For example, a value ending in `3.5"` would have the `"` stripped. While in practice most values don't end in these characters, this is a latent bug that could silently mangle data.

This finding was previously deferred as C3-08 and is still present. Noting it again as it remains unfixed.

---

## Summary

- CR24-01 (LOW): `SpecDerived.size` shadows `Spec.size` — maintainability concern (no correctness issue)
- CR24-02 (LOW): `rstrip("</")` strips chars not string — previously deferred as C3-08, still present
