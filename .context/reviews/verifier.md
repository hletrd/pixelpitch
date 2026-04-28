# Verifier Review (Cycle 38) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes

## V38-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C37 fixes verified working. No regressions.

## V38-02: `isInvalidData` with `pitch === 0` hides rows that `test_template_zero_pitch_rendering` expects to be visible — verified contradiction

**File:** `templates/pixelpitch.html`, lines 157, 277-279, 84-89
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:** Traced the data flow:
1. Template renders `0.0` pitch as "0.0 µm" (line 84-88: `spec.pitch is not none` is True for 0.0)
2. Test `test_template_zero_pitch_rendering` asserts "0.0 µm" appears in rendered HTML — PASSES
3. JS `isInvalidData` returns `true` for `pitch === 0` (line 277-279)
4. "Hide possibly invalid data" checkbox is checked by default (line 157)
5. `applyInvalidFilter()` is called on page load (line 327)
6. Result: rows with `pitch=0.0` are hidden by default

The template correctly renders the value, but JS hides it. Users with default settings never see "0.0 µm". If they uncheck the toggle, they see a physically impossible value displayed as if valid.

**Fix:** Template should render "unknown" for `pitch=0.0`, consistent with JS treating it as invalid.

---

## Summary

- V38-01: Gate tests pass
- V38-02 (MEDIUM): `isInvalidData` hides zero-pitch rows that template renders as "0.0 µm" — UX contradiction
