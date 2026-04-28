# Tracer Review (Cycle 24) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-23 fixes

## Previous Findings Status

All previously traced flows confirmed still correct. No regressions.

## New Findings

### TR24-01: TYPE_FRACTIONAL_RE space+inch gap — traced data flow impact

**File:** `sources/__init__.py` line 68, consumed by `pixelpitch.py` line 544, `gsmarena.py` line 135
**Severity:** LOW | **Confidence:** HIGH

**Causal trace:**
1. A source page contains sensor text with format `1/2.3 inch` (space before "inch")
2. TYPE_FRACTIONAL_RE.search() returns None (no match)
3. In `gsmarena._phone_to_spec()`: `fmt_match = TYPE_FRACTIONAL_RE.search(main)` returns None → `sensor_type = None`
4. `size = PHONE_TYPE_SIZE.get(None)` → `size = None`
5. Camera has no sensor size → lower data quality, shows "unknown" on website

**Alternative hypothesis:** The same sensor type might also appear in the Geizhals "Typ" field, but GSMArena doesn't have a "Typ" field, so there's no fallback.

**Competing hypothesis:** This format simply doesn't appear in any current source. LOW real-world risk.

### TR24-02: parse_sensor_field 1-inch gap — traced data flow impact

**File:** `pixelpitch.py` lines 529-558, consumed by `extract_specs()` line 593
**Severity:** LOW | **Confidence:** HIGH

**Causal trace:**
1. Geizhals sensor field contains `CMOS 1"` without explicit mm dimensions
2. `parse_sensor_field()` calls `TYPE_FRACTIONAL_RE.search()` → no match (requires `1/x.y` prefix)
3. `SIZE_RE.search()` also returns None (no mm dimensions in text)
4. Result: `{type: None, size: None, pitch: None}`
5. Camera enters `derive_spec()` with `spec.size=None, spec.type=None`
6. `sensor_size_from_type(None)` returns None
7. Camera has no sensor data at all — shows "unknown" on website

**Mitigating factor:** Geizhals typically includes mm dimensions, making this unlikely.

---

## Summary

- TR24-01 (LOW): TYPE_FRACTIONAL_RE space+inch gap traced — leads to missing sensor size
- TR24-02 (LOW): parse_sensor_field 1-inch gap traced — leads to all-unknown sensor data
