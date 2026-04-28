# Architect Review (Cycle 28) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes

## Previous Findings Status

ARCH27-01 (PITCH_UM_RE incomplete) addressed in C27. All previous architectural concerns remain deferred.

## New Findings

### ARCH28-01: DRY centralization incomplete — source modules maintain local regex copies

**File:** `sources/apotelyt.py`, `sources/gsmarena.py`, `sources/cined.py`
**Severity:** LOW | **Confidence:** HIGH

The C25-01 and C26-01 centralization established `sources/__init__.py` as the single source of truth for shared regex patterns (SIZE_MM_RE, PITCH_UM_RE, MPIX_RE, TYPE_FRACTIONAL_RE). However, 3 source modules still maintain their own local copies:

Current centralization status:
- `TYPE_FRACTIONAL_RE` — fully centralized (GSMArena imports shared) ✓
- `SIZE_MM_RE` — NOT fully centralized (Apotelyt, CineD have local SIZE_RE) ✗
- `PITCH_UM_RE` — NOT fully centralized (Apotelyt, GSMArena have local PITCH_RE) ✗
- `MPIX_RE` — NOT fully centralized (Apotelyt has local MPIX_RE matching only "Megapixel") ✗

The local copies diverge from the shared patterns:
- Apotelyt PITCH_RE: missing `um`, `&micro;m`, `&#956;m`
- Apotelyt MPIX_RE: only matches "Megapixel" (not "MP" or "Mega pixels")
- GSMArena PITCH_RE: has `um` but missing `microns`, `&micro;m`, `&#956;m`

**Fix:** Replace local regex copies with imports from `sources/__init__.py`. This completes the DRY centralization.

---

## Summary

- ARCH28-01 (LOW): DRY centralization incomplete — 3 source modules have divergent local regex copies
