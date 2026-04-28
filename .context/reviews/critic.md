# Critic Review (Cycle 31) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## Previous Findings Status

C30-01 and C30-02 both implemented. All previous fixes stable.

## New Findings

### CRIT31-01: merge_camera_data spec/derived pitch inconsistency after field preservation

**File:** `pixelpitch.py`, lines 413-432
**Severity:** MEDIUM | **Confidence:** HIGH

The merge function preserves `spec.pitch` from existing data when new has None (line 417-418), and separately preserves `derived.pitch` from existing when new has None (line 431-432). However, these two checks are independent. The scenario that falls through the cracks:

1. New data: `spec.pitch=None`, `spec.size=(5.0, 3.7)`, `spec.mpix=10.0`
2. After `derive_spec()`: `derived.pitch = pixel_pitch(18.5, 10.0)` ~= 1.36 (computed from area+mpix)
3. Existing data: `spec.pitch=2.0`, `derived.pitch=2.0` (direct measurement from Geizhals)
4. Merge: `spec.pitch` preserved from existing (2.0) because new has None -- CORRECT
5. Merge: `derived.pitch` NOT preserved because new has 1.36 (not None) -- WRONG
6. Template displays `derived.pitch=1.36`, losing the authoritative 2.0

This is a data integrity issue: the template reads `derived.pitch`, so the Geizhals-measured 2.0um pitch is silently replaced by the computed 1.36um pitch, and this incorrect value gets persisted to the CSV on next write.

**Fix:** After all Spec field preservation, check whether `spec.pitch` was updated from existing. If so, and if `derived.pitch` differs from `spec.pitch`, set `derived.pitch = spec.pitch`.

---

### CRIT31-02: BOM character literal fragility in two files

**File:** `pixelpitch.py` line 276; `sources/openmvg.py` line 67
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

Both files use `csv_content[0] == "﻿"` with the literal BOM character. If the source file is re-saved without BOM awareness (common in some editors/CI pipelines), the literal character disappears and the check silently breaks. Using `csv_content.startswith('﻿')` is robust against this.

**Fix:** Replace literal BOM with `'﻿'` escape sequence.

---

## Summary

- CRIT31-01 (MEDIUM): merge_camera_data spec/derived pitch inconsistency — computed value overwrites preserved measurement
- CRIT31-02 (LOW-MEDIUM): BOM literal character fragility
