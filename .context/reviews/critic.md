# Critic Review (Cycle 34) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-33 fixes, focusing on NEW issues

## Previous Findings Status

CRIT33-01 (systemic truthy-vs-None) fixed in C33 across derive_spec, sorted_by, prettyprint, template.

## New Findings

### CRIT34-01: Truthy-vs-None pattern still has residual instances — C33-01 fix was not fully exhaustive

**Files:** `pixelpitch.py` (line 1170, line 217, line 227)
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

The C33-01 fix addressed the 4 most visible truthy-vs-None locations. However, three additional instances remain:

1. **`list` command (line 1170):** `if spec.pitch:` — 0.0 pitch cameras silently omitted. Low impact (CLI only).

2. **match_sensors guard (line 217):** `if not width or not height:` — 0.0 dimensions treated as None. Low impact (physically meaningless).

3. **match_sensors inner loop (line 227):** `if not sensor_width or not sensor_height:` — same pattern for sensor_db entries.

While items 2 and 3 are physically meaningless (no sensor has 0.0 mm dimensions), item 1 is a consistency gap. The C33-01 fix should have been applied project-wide, not just to the 4 most visible locations.

### CRIT34-02: match_sensors ZeroDivisionError with mpix=0.0 — real crash risk

**File:** `pixelpitch.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

The division `abs(megapixels - mp) / megapixels` crashes when megapixels=0.0. This is a correctness bug, not just a consistency issue. The guard `if megapixels is not None and sensor_megapixels:` does not protect against 0.0.

**Fix:** Add `megapixels > 0` to the guard condition.

---

## Summary

- CRIT34-01 (LOW-MEDIUM): Residual truthy-vs-None instances in `list` command and match_sensors
- CRIT34-02 (MEDIUM): match_sensors ZeroDivisionError with mpix=0.0 — real crash risk
