# Tracer Review (Cycle 25) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes

## Previous Findings Status

All previously traced flows confirmed still correct. No regressions.

## New Findings

### TR25-01: ValueError in parse_sensor_field traced — entire category lost

**File:** `pixelpitch.py` lines 556/561 → `extract_specs` line 598 → `get_category` line 713 → `render_html` lines 862-867
**Severity:** MEDIUM | **Confidence:** MEDIUM

**Causal trace:**
1. Geizhals HTML row has a corrupted sensor field text containing something like `"36.0.1x24.0mm"` (multi-dot value)
2. `parse_sensor_field()` → `SIZE_RE.search()` matches `"36.0.1"` as group(1)
3. `float("36.0.1")` raises `ValueError`
4. ValueError propagates: `parse_sensor_field` → `extract_specs` → `get_category` → `render_html`
5. `render_html` catches `Exception` and drops the entire category: `category_specs[category] = []`
6. All cameras in that category are lost for this deployment cycle

**Mitigating factor:** The outer `try/except Exception` in `render_html` prevents total failure. Previous data is preserved via the CSV merge. But the current cycle's Geizhals data for that category is completely lost.

**Competing hypothesis:** Geizhals HTML is well-structured and unlikely to produce multi-dot numbers. But defensive coding is warranted since the consequence is category-wide data loss.

**Fix:** Add try/except ValueError in parse_sensor_field around float() calls, returning None for unparseable values (consistent with `parse_existing_csv` and `sensor_size_from_type` patterns).

### TR25-02: SIZE_RE × gap traced — silent data loss on format change

**File:** `pixelpitch.py` line 42, consumed by `parse_sensor_field` line 554
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Geizhals sensor text contains `"CMOS 36.0×24.0mm"` (Unicode × instead of ASCII x)
2. `SIZE_RE.search()` returns None (only matches lowercase `x`)
3. `parse_sensor_field()` returns `size=None`
4. `derive_spec()` tries `sensor_size_from_type(type)` but type may also be None
5. Camera has `size=None, area=None, pitch=None` — shows "unknown" on website

**Alternative hypothesis:** Geizhals currently uses ASCII `x` and micro sign `µ` in their German-language site. A format change is unlikely but possible.

---

## Summary

- TR25-01 (MEDIUM): ValueError in parse_sensor_field → entire category lost
- TR25-02 (MEDIUM): SIZE_RE × gap → silent data loss on Geizhals format change
