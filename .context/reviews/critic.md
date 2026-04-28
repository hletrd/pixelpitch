# Critic Review (Cycle 39) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## Previous Findings Status

CRIT38-01 fixed. Template now renders "unknown" for 0.0 pitch/mpix. But the fix was narrow — only 0.0 was addressed, not the broader class of invalid values.

## New Findings

### CRIT39-01: C38-01 fix was incomplete — same class of bug exists for negative/NaN/inf values

**File:** `templates/pixelpitch.html`, lines 76-80, 84-88
**Severity:** MEDIUM | **Confidence:** HIGH

The C38-01 fix changed the template guard from `is not none` to `is not none and != 0.0`. This addressed the specific 0.0-sentinel case but missed the broader principle: **physically impossible values should not render as numbers**. The `!= 0.0` check is an allowlist of valid values by exclusion, when it should be a positivity check.

The same class of bug now exists for negative, NaN, and inf values:
- Negative pitch (`-1.0`) renders as "-1.0 µm" — physically impossible
- NaN pitch renders as "nan µm" — malformed
- NaN mpix renders as "nan MP" — malformed
- inf mpix renders as "inf MP" — malformed

This is a direct consequence of fixing C38-01 with a narrow `!= 0.0` guard instead of the more robust `> 0` check. The `> 0` guard would have handled 0.0, negative, and NaN in a single condition (since NaN > 0 evaluates to False in Python/Jinja2).

**Fix:** Replace `!= 0.0` with `> 0` in both the pitch and mpix template guards. This is the correct generalization of the C38-01 fix.

---

## Summary

- CRIT39-01 (MEDIUM): C38-01 fix was narrow — `!= 0.0` should be `> 0` to cover negative/NaN/inf
