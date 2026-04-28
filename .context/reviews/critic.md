# Critic Review (Cycle 25) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes, focusing on NEW issues

## Previous Findings Status

C24-01 (TYPE_FRACTIONAL_RE space+inch) and C24-02 (bare 1-inch type) implemented. All previous fixes stable.

## New Findings

### CRIT25-01: SIZE_RE/PITCH_RE inconsistency between pixelpitch.py and sources/__init__.py

**File:** `pixelpitch.py`, lines 42-43 vs `sources/__init__.py`, lines 65-66
**Severity:** MEDIUM | **Confidence:** HIGH

The Geizhals path uses more restrictive regex patterns than the shared source patterns. `SIZE_RE` only matches lowercase `x` (not `×` or spaces), and `PITCH_RE` only matches micro sign `µ` (not Greek mu `μ` or "microns"). If Geizhals changes their HTML format, these parsers silently fail while the other source parsers handle the same data correctly.

This is a consistency and robustness concern: the same type of data (sensor dimensions, pixel pitch) is parsed with different levels of robustness depending on which source it comes from.

**Fix:** Use the shared patterns from `sources/__init__.py` in `parse_sensor_field()`, or upgrade the local patterns to match the robustness of the shared ones.

---

### CRIT25-02: parse_sensor_field missing ValueError guard — entire category at risk

**File:** `pixelpitch.py`, lines 556, 561
**Severity:** MEDIUM | **Confidence:** MEDIUM

If a malformed sensor field produces a regex match that `float()` cannot parse (e.g., "36.0.1"), a `ValueError` propagates up and causes the entire Geizhals category to be dropped. The outer `try/except Exception` in `render_html` catches the error, but the result is that all cameras in that category are lost for that deployment.

Other parsing functions in the codebase (e.g., `parse_existing_csv`, `sensor_size_from_type`) use try/except to gracefully handle unparseable values. This function should follow the same pattern.

**Fix:** Add try/except ValueError around the float() calls in parse_sensor_field.

---

## Summary

- CRIT25-01 (MEDIUM): SIZE_RE/PITCH_RE inconsistency — less robust than shared patterns
- CRIT25-02 (MEDIUM): parse_sensor_field ValueError guard missing — category-wide data loss risk
