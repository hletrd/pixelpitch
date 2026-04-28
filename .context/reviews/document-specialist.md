# Document Specialist Review (Cycle 17) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository doc/code review after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- DS16-01 (sensor_size_from_type docstring): Fixed — docstring now states "Invalid fractional types (e.g. "1/0", "1/") return None instead of raising ZeroDivisionError or ValueError."
- DS16-02 (merge_camera_data docstring): Fixed — docstring now states "If the same key appears multiple times in new_specs, only the first occurrence is kept and subsequent duplicates are silently dropped."

## New Findings

### DS17-01: openMVG docstring says "Pentax K-*, etc." but regex now covers K3, K5 without hyphen
**File:** `sources/openmvg.py`, lines 10-21
**Severity:** LOW | **Confidence:** HIGH

The docstring says "Pentax K-*" which implies only hyphenated models. After the C16-03 fix, the regex now matches K3, K5, K7, K1 (no hyphen) as well as K-30, K-50 (with hyphen). The docstring should be updated to reflect the broader coverage. Also, after fixing C17-01, it should mention KP, KF, K-r, K-x coverage.

**Fix:** Update the docstring to say "Pentax K-mount (K3, K-1, KP, etc.)" instead of "Pentax K-*".

---

### DS17-02: GSMArena SENSOR_FORMAT_RE pattern not documented as ASCII-only
**File:** `sources/gsmarena.py`, line 50
**Severity:** NEGLIGIBLE | **Confidence:** HIGH

The regex pattern only matches ASCII double-quote characters, not Unicode curly quotes. This limitation should be documented in a comment.

**Fix:** Add a comment noting the ASCII-quote limitation if the regex is not fixed.

---

## Summary
- NEW findings: 2 (1 LOW, 1 NEGLIGIBLE)
- DS17-01: openMVG docstring outdated after regex fix — LOW
- DS17-02: GSMArena quote limitation undocumented — NEGLIGIBLE
