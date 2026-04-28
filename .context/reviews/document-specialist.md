# Document Specialist Review (Cycle 18) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository doc/code review after cycles 1-17 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

- DS16-01 (sensor_size_from_type docstring): Fixed.
- DS16-02 (merge_camera_data docstring): Fixed.
- DS17-01 (openMVG docstring): Fixed — now says "Pentax K-mount (K3, K-1, KP, etc.)" and "Nikon D/Df".
- DS17-02 (GSMArena quote limitation): Fixed — Unicode quotes now supported.

## New Findings

### DS18-01: `SENSOR_TYPE_RE` in pixelpitch.py has no comment noting its ASCII-only limitation
**File:** `pixelpitch.py`, line 43
**Severity:** NEGLIGIBLE | **Confidence:** HIGH

After C17-03 fixed GSMArena's regex to support Unicode quotes, the `SENSOR_TYPE_RE` in pixelpitch.py remains ASCII-only. While this is likely intentional for Geizhals parsing, there's no comment explaining why it differs from `TYPE_FRACTIONAL_RE`.

**Fix:** Add a comment: `# Geizhals HTML uses ASCII double-quotes in title attributes`.

---

## Summary
- NEW findings: 1 (NEGLIGIBLE)
- DS18-01: SENSOR_TYPE_RE has no comment about ASCII-only limitation — NEGLIGIBLE
