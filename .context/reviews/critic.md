# Critic Review (Cycle 24) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-23 fixes

## Previous Findings Status

C23-01 (body-category fallback tests) implemented and verified. All previous fixes stable.

## New Findings

### CRIT24-01: TYPE_FRACTIONAL_RE does not match "1/x.y inch" (space before "inch")

**File:** `sources/__init__.py`, line 68
**Severity:** LOW | **Confidence:** MEDIUM

The `TYPE_FRACTIONAL_RE` pattern is `(1/[\d.]+)(?:\"|inch|-inch|-type|\s*type|″)`. It matches `1/2.3-inch` and `1/2.3"` and `1/2.3 type` but NOT `1/2.3 inch` (space before "inch"). The pattern has `\s*type` but lacks the corresponding `\s*inch` alternative.

**Concrete failure scenario:** If a source page uses the format `1/2.3 inch` (with a space), the sensor type is not extracted, and `sensor_size_from_type` is never called. The camera would have `type=None` and `size=None` unless explicit mm dimensions are also present.

**Real-world risk:** LOW — all current sources use `1/2.3"` (quote) or `1/2.3-inch` (hyphenated) format. The space-before-inch format is uncommon but valid English.

**Fix:** Add `\s*inch` alternative to the regex: `(1/[\d.]+)(?:\"|\s*inch|-inch|-type|\s*type|″)`

---

### CRIT24-02: parse_sensor_field misses bare 1-inch sensor type ("1"")

**File:** `pixelpitch.py`, lines 529-558
**Severity:** LOW | **Confidence:** HIGH

The `parse_sensor_field` function uses `TYPE_FRACTIONAL_RE` to extract sensor types. This regex only matches fractional-inch formats (`1/x.y` suffix), not the bare 1-inch format (`1"`). When a Geizhals sensor field contains `CMOS 1"` without explicit mm dimensions, the type is not extracted, and no sensor size can be computed.

**Concrete failure scenario:** A Geizhals entry with sensor text `CMOS 1"` (no mm dims) would produce `type=None, size=None, pitch=None`. The camera appears with "unknown" sensor size on the website.

**Real-world risk:** LOW — Geizhals shopping site entries typically include explicit mm dimensions. But if a 1-inch sensor camera entry ever lacks mm dimensions, the sensor data is silently lost.

**Fix:** Add a separate regex or check for bare 1-inch format (`1"` or `1-inch`) after the `TYPE_FRACTIONAL_RE` check fails.

---

## Summary

- CRIT24-01 (LOW): TYPE_FRACTIONAL_RE misses `1/x.y inch` (space before inch)
- CRIT24-02 (LOW): parse_sensor_field misses bare 1-inch sensor type
