# Critic Review (Cycle 17) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- CR16-01 (sensor_size_from_type crash): Fixed — try/except guard confirmed.
- CR16-02 (merge dedup): Fixed — `seen_new_keys` set confirmed working.
- CR16-03 (Pentax regex): Partially fixed — `K[-\s]?\d+[A-Za-z]*` now covers K3, K5, K-30, K100D etc., but still misses KP, KF, K-r, K-x (no digit after K or after hyphen).
- CR16-04 (digicamdb alias): Fixed — removed from SOURCE_REGISTRY.

## New Findings

### CR17-01: Pentax KP/KF/K-r/K-x STILL misclassified — C16-03 fix was incomplete
**File:** `sources/openmvg.py`, line 47
**Severity:** MEDIUM | **Confidence:** HIGH

The C16-03 fix changed the regex to `Pentax\s+K[-\s]?\d+[A-Za-z]*` which requires at least one digit. Pentax KP and KF have a letter directly after K (no digit). Pentax K-r and K-x have a hyphen followed by a letter (no digit). All four are DSLRs. This is a carry-over from the incomplete fix.

**Fix:** Change to `Pentax\s+K[-\s]?[\dA-Za-z]+[A-Za-z]*` to allow letters or digits after K[-\s]?.

---

### CR17-02: Nikon Df — a known DSLR missed by the regex
**File:** `sources/openmvg.py`, line 46
**Severity:** LOW | **Confidence:** HIGH

The Nikon Df is a well-known retro DSLR with no digit after "D". The regex `Nikon\s+D\d{1,4}` requires at least one digit. If the Df appears in the openMVG database, it would be classified as mirrorless.

**Fix:** Add `|Nikon\s+Df` to the regex alternation.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- CR17-01: Pentax KP/KF/K-r/K-x STILL misclassified — MEDIUM
- CR17-02: Nikon Df missed by DSLR regex — LOW
