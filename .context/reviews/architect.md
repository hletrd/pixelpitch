# Architect Review (Cycle 18) — Architectural/Design Risks, Coupling, Layering

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository architecture re-review after cycles 1-17 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

- A16-01 (merge dedup contract): Fixed.
- A16-02 (digicamdb registry DRY): Fixed.
- A16-03 (sensor_size_from_type validation): Fixed.
- A17-01 (DSLR regex maintenance): Acknowledged as inherent limitation.

## New Findings

### A18-01: Sensor-type regex is defined in three places — violates DRY
**Files:** `pixelpitch.py` line 43, `sources/__init__.py` line 68, `sources/gsmarena.py` line 50
**Severity:** LOW | **Confidence:** HIGH

The pattern to match fractional-inch sensor types (e.g., `1/2.3"`) is independently defined in three locations:
1. `SENSOR_TYPE_RE` in pixelpitch.py — ASCII quotes only
2. `TYPE_FRACTIONAL_RE` in sources/__init__.py — comprehensive (ASCII + Unicode + "inch" suffixes)
3. `SENSOR_FORMAT_RE` in gsmarena.py — ASCII + Unicode quotes

The three regexes have diverged in capability: `TYPE_FRACTIONAL_RE` handles the most formats, while `SENSOR_TYPE_RE` is the most limited. This violates the Single Source of Truth principle and increases maintenance burden.

The most robust version (`TYPE_FRACTIONAL_RE`) is in the shared module `sources/__init__.py`, making it naturally the canonical definition. The other two should either import it or be aligned.

**Fix:** Import `TYPE_FRACTIONAL_RE` from `sources/__init__.py` in both pixelpitch.py and gsmarena.py, or create a shared regex module.

---

## Summary
- NEW findings: 1 (LOW)
- A18-01: Three divergent sensor-type regexes — DRY violation — LOW
