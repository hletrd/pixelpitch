# Debugger Review (Cycle 24) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-23 fixes

## Previous Findings Status

All previous bug fixes confirmed still working. No regressions detected.

## New Findings

### DBG24-01: TYPE_FRACTIONAL_RE silently fails on "1/x.y inch" format

**File:** `sources/__init__.py`, line 68
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** When a source page uses the format `1/2.3 inch` (space before "inch"), TYPE_FRACTIONAL_RE returns None, causing the sensor type to be lost. The camera then has no sensor size data unless explicit mm dimensions are present.

**Root cause:** The regex pattern has `inch` (no space) and `-inch` alternatives but lacks `\s*inch`. The `\s*type` alternative was added for the "type" suffix, but the corresponding `\s*inch` was not.

**Impact:** Camera appears with "unknown" sensor data on the website. Not a crash — silent data loss.

### DBG24-02: parse_sensor_field silently drops 1-inch sensor type

**File:** `pixelpitch.py`, lines 529-558
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** When Geizhals sensor field contains `CMOS 1"` without explicit mm dimensions, the 1-inch sensor type is not extracted. TYPE_FRACTIONAL_RE requires `1/x.y` prefix, which the bare `1"` format doesn't match.

**Root cause:** TYPE_FRACTIONAL_RE was designed only for fractional-inch formats. The bare 1-inch format (`1"`) is a separate designation that isn't handled.

**Impact:** Camera with 1-inch sensor shows "unknown" sensor size. Not a crash — silent data loss.

### DBG24-03: _parse_fields rstrip("</") silently strips valid characters

**File:** `sources/imaging_resource.py`, line 95
**Severity:** LOW | **Confidence:** HIGH

Previously deferred as C3-08. Re-confirming it remains present. The `rstrip("</")` strips individual characters `<`, `/`, and `"` from the end of values. A value ending in a double-quote (e.g., `3.5"`) would have it stripped.

**Real-world risk:** LOW — Imaging Resource values rarely end in these characters, and the regex/HTML processing usually cleans up tag remnants before this line.

---

## Summary

- DBG24-01 (LOW): TYPE_FRACTIONAL_RE fails on space+inch format — silent data loss
- DBG24-02 (LOW): parse_sensor_field drops 1-inch sensor type — silent data loss
- DBG24-03 (LOW): rstrip("</") strips chars not string — previously deferred as C3-08
