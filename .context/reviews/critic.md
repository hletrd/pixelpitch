# Critic Review (Cycle 26) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes, focusing on NEW issues

## Previous Findings Status

C25-01 (SIZE_RE/PITCH_RE centralization) and C25-02 (ValueError guard) implemented. All previous fixes stable.

## New Findings

### CRIT26-01: MPIX_RE centralization was missed in C25-01 fix — incomplete DRY resolution

**File:** `pixelpitch.py`, line 42 vs `sources/__init__.py`, line 67
**Severity:** MEDIUM | **Confidence:** HIGH

The C25-01 aggregate review explicitly mentioned MPIX_RE: "Update the MPIX_RE in pixelpitch.py similarly (it only matches 'Megapixel', not 'MP' or 'Mega pixels')." But the implementation plan deferred it: "MPIX_RE.search() stays (no shared equivalent for 'Megapixel')". This was an error — `sources/__init__.py` line 67 exports `MPIX_RE` that handles "MP", "Mega pixels", "Megapixel" etc.

The centralization was incomplete. SIZE_RE and PITCH_RE were centralized, but MPIX_RE was not, despite being flagged in the same finding.

**Fix:** Import MPIX_RE from sources in pixelpitch.py, following the same pattern as SIZE_MM_RE/PITCH_UM_RE/TYPE_FRACTIONAL_RE.

---

### CRIT26-02: ValueError guard inconsistency — only applied to Geizhals path, not source modules

**File:** `sources/cined.py` line 98, `sources/apotelyt.py` lines 119-129, `sources/gsmarena.py` lines 130/133, `sources/imaging_resource.py` line 228
**Severity:** LOW | **Confidence:** MEDIUM

The C25-02 fix added ValueError guards in `parse_sensor_field()` but source modules that parse the same kind of data (sensor dimensions, pixel pitch, megapixels) from different websites still call `float()` on regex matches without any guard. The same multi-dot input that could crash `parse_sensor_field()` could crash these modules.

However, the blast radius is smaller: source modules process one camera at a time in a loop, and exceptions are caught by the outer try/except in each module. The individual camera is lost, but not the entire category.

**Fix:** Add try/except ValueError guards in source modules for consistency with the parse_sensor_field() pattern.

---

## Summary

- CRIT26-01 (MEDIUM): MPIX_RE centralization missed in C25-01 — incomplete DRY fix
- CRIT26-02 (LOW): ValueError guard inconsistency across source modules
