# Critic Review (Cycle 38) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes

## Previous Findings Status

All C37 findings fixed. `derive_spec` isfinite guard working. No regressions in core logic.

## New Findings

### CRIT38-01: C37-02 fix created UX contradiction — `isInvalidData` hides zero-pitch rows that template renders as valid numbers

**File:** `templates/pixelpitch.html`, lines 84-89, 277-279
**Severity:** MEDIUM | **Confidence:** HIGH

The C37-02 fix added `pitch === 0` to `isInvalidData`, hiding zero-pitch rows by default. But the template still renders `0.0 µm` as a number (not "unknown"). This is a classic case of a fix introducing a secondary issue: the template and JS filter now disagree on the semantics of `pitch=0.0`.

The test `test_template_zero_pitch_rendering` explicitly asserts that 0.0 pitch renders as "0.0 µm", confirming the template behavior. But the JS filter hides it, making the rendering invisible to users with the default toggle state.

The correct fix is to make the template treat `0.0` pitch the same as `None` — render "unknown". This aligns template rendering with JS filtering semantics. A zero pixel pitch is physically impossible; displaying it as a number is misleading even if visible.

**Fix:** Add `spec.pitch != 0.0` to the template's pitch-rendering condition, so zero pitch displays "unknown" just like None pitch. Keep the JS `pitch === 0` check as defense-in-depth.

---

## Summary

- CRIT38-01 (MEDIUM): C37-02 fix created UX contradiction — template renders "0.0 µm" but JS hides the row
