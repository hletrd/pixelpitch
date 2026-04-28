# Debugger Review (Cycle 31) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## Previous Findings Status

DBG30-01 (GSMArena fetch per-phone try/except) fixed in C30.

## New Findings

### DBG31-01: merge_camera_data leaves derived.pitch inconsistent with spec.pitch

**File:** `pixelpitch.py`, lines 413-432
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** When openMVG provides a camera with `spec.pitch=None` and `spec.mpix`/`spec.size` set, `derive_spec()` computes `derived.pitch` from area+mpix. If existing CSV data has a direct `spec.pitch` measurement (e.g., from Geizhals), the merge preserves `spec.pitch` but NOT `derived.pitch` (because the new `derived.pitch` is not None). The template reads `derived.pitch`, so the computed value (which is an approximation) overwrites the authoritative measurement.

**Root cause:** Lines 431-432 check `new_spec.pitch is None` independently from lines 417-418 which check `new_spec.spec.pitch is None`. These two conditions can have different truth values when `derived.pitch` was computed from area+mpix.

**Fix:** After all Spec field preservation, ensure `derived.pitch` is consistent with `spec.pitch`. If `spec.pitch` is set after merge, `derived.pitch` should equal `spec.pitch`.

---

### DBG31-02: BOM literal character can be silently stripped by re-encoding

**File:** `pixelpitch.py` line 276; `sources/openmvg.py` line 67
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

**Failure mode:** If a Python source file containing the literal BOM character U+FEFF is re-encoded by a tool that strips or normalizes Unicode, the BOM literal becomes an empty string or incorrect character. The BOM-stripping code then fails to detect BOM-prefixed CSVs, causing mangled header detection and 0-row parses.

**Root cause:** Using a literal Unicode character instead of the escape sequence `'﻿'`.

**Fix:** Replace `csv_content[0] == "﻿"` with `csv_content.startswith('﻿')`.

---

## Summary

- DBG31-01 (MEDIUM): merge_camera_data derived.pitch inconsistent with spec.pitch after field preservation
- DBG31-02 (LOW-MEDIUM): BOM literal character fragility in two files
