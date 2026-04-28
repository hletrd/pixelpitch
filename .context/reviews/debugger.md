# Debugger Review (Cycle 38) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes

## Previous Findings Status

All C37 findings fixed. No regressions in core logic.

## New Findings

### DBG38-01: C37-02 fix introduced behavioral regression — zero-pitch rows now hidden by default but template still renders "0.0 µm"

**File:** `templates/pixelpitch.html`, lines 84-89, 277-279
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** When a camera has `pitch=0.0` (from `pixel_pitch` returning 0.0 for invalid inputs):
1. Template renders "0.0 µm" in the table cell
2. JS `isInvalidData` returns true for `pitch === 0`
3. "Hide possibly invalid data" is checked by default
4. Row is hidden on page load
5. User never sees "0.0 µm" — it's invisible

If the user unchecks "Hide possibly invalid data", they see "0.0 µm" displayed as a legitimate value, which is misleading. A zero pixel pitch is physically impossible.

**Regression:** Before C37-02, zero-pitch rows were visible (rendered as "0.0 µm"). After C37-02, they're hidden by default. The template rendering wasn't updated to match, creating an inconsistency.

**Fix:** Update the template to render "unknown" for `pitch=0.0` and `mpix=0.0`, consistent with JS treating these as invalid.

---

## Summary

- DBG38-01 (MEDIUM): C37-02 introduced regression — zero-pitch rows hidden but template still renders "0.0 µm"
