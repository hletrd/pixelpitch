# Critic Review (Cycle 28) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes, focusing on NEW issues

## Previous Findings Status

C27-01 (PITCH_UM_RE "um") and C27-02 (year validation) both implemented. All previous fixes stable.

## New Findings

### CRIT28-01: imaging_resource.py pitch ValueError guard missing — C26-02 fix was incomplete

**File:** `sources/imaging_resource.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

The C26-02 fix added ValueError guards to `size` (line 229) and `mpix` (line 246) but missed `pitch` (line 238). This is the most significant new finding because it can crash the Imaging Resource scraper at runtime.

The `IR_PITCH_RE` pattern `([\d.]+)` matches multi-dotted strings like "5.1.2", and `float("5.1.2")` raises `ValueError`. If Imaging Resource ever serves a malformed pixel pitch value, the entire `fetch_one()` call crashes.

**Fix:** Add try/except ValueError around the pitch float() call, consistent with size and mpix.

---

### CRIT28-02: DRY inconsistency — source modules still have local regex copies not synchronized with shared patterns

**File:** `sources/apotelyt.py` line 35, `sources/gsmarena.py` line 50, `sources/cined.py` line 30
**Severity:** LOW | **Confidence:** HIGH

After the C25-01 and C26-01 centralization of shared regex patterns (SIZE_MM_RE, PITCH_UM_RE, MPIX_RE), several source modules still maintain local copies:

- `apotelyt.py` line 34 `SIZE_RE` — identical to shared `SIZE_MM_RE`
- `apotelyt.py` line 35 `PITCH_RE` — differs from shared `PITCH_UM_RE` (missing `um`, `&micro;m`, `&#956;m`)
- `apotelyt.py` line 36 `MPIX_RE` — differs from shared `MPIX_RE` (only matches "Megapixel", not "MP" or "Mega pixels")
- `cined.py` line 30 `SIZE_RE` — identical to shared `SIZE_MM_RE`
- `gsmarena.py` line 50 `PITCH_RE` — differs from shared `PITCH_UM_RE` (matches `um` but missing `microns`, `&micro;m`, `&#956;m`)

This is a DRY maintenance risk — if the shared patterns are updated, local copies may not be, leading to divergent behavior.

**Impact:** Currently no data is lost because each source module works correctly with its own local pattern. But future regex changes to the shared patterns won't propagate to the local copies.

**Fix:** Import and use the shared patterns from `sources/__init__.py`, removing the local copies. If a source needs source-specific behavior (e.g., Apotelyt only needs "Megapixel"), document why the shared pattern is used instead.

---

## Summary

- CRIT28-01 (MEDIUM): imaging_resource.py pitch ValueError guard missing — C26-02 incomplete
- CRIT28-02 (LOW): DRY inconsistency — local regex copies not synchronized with shared patterns
