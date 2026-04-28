# Verifier Review (Cycle 45) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28

## Previous Findings Status

C44-01, C44-02 — COMPLETED. CineD dead code removed.

## Verification Results

### V45-VERIFY-01: GSMArena _select_main_lens with decimal MP

- Input: `cam = '12.2 MP, f/1.9, 25mm (wide), 1/2.55", 1.25µm, dual pixel PDAF, OIS 12 MP, f/2.2, 114mm (ultrawide), 1/2.9, 1.25µm'`
- `_select_main_lens(cam)` returns `'2 MP, f/1.9, 25mm (wide), 1/2.55", 1.25µm, dual pixel PDAF, OIS'`
- Extracted mpix: 2.0 (WRONG — should be 12.2)
- Extracted pitch: 1.25 (correct by coincidence — pitch data is after the split point)
- Extracted type: None (WRONG — `1/2.55"` lost its quote suffix in the split)
- Status: **BUG CONFIRMED** — mpix and type are corrupted

### V45-VERIFY-02: GSMArena _select_main_lens with integer MP (no bug)

- Input: `cam = '200 MP, f/1.7, 24mm (wide), 1/1.3", 0.6µm, OIS 10 MP, f/2.4, 67mm (telephoto), 1/3.52", 1.12µm'`
- `_select_main_lens(cam)` returns `'200 MP, f/1.7, 24mm (wide), 1/1.3", 0.6µm, OIS'`
- Extracted mpix: 200.0 (correct)
- Extracted type: '1/1.3' (correct)
- Status: OK — integer MP values work correctly

### V45-VERIFY-03: Regex split analysis

- Pattern: `r'(?=\b\d+(?:\.\d+)?\s*MP\b)'`
- Input: `'12.2 MP, f/1.7, (wide)'`
- Split positions: after "12." (at the `\b` between "0" and "."), and before "2 MP"
- Result: `['12.', '2 MP, f/1.7, (wide)']` — **INCORRECT**
- Root cause: `\b` matches between digit and decimal point, causing the greedy `\d+` to match only "0" before the decimal, not the full "12"

## New Findings

### V45-01: GSMArena _select_main_lens regex split corrupts decimal MP data

**File:** `sources/gsmarena.py, line 82`
**Severity:** HIGH | **Confidence:** HIGH (verified by live execution)

Verified with direct Python execution that `_select_main_lens('12.2 MP, f/1.9, 25mm (wide), 1/2.55", 1.25µm, ...')` returns a fragment starting with "2 MP" instead of "12.2 MP". The mpix extraction then produces 2.0 instead of 12.2, and the sensor type is lost because the fractional-inch format's quote suffix is severed.

**Fix:** Remove `\b` from the start of the split regex: `r'(?=\d+(?:\.\d+)?\s*MP\b)'`

---

## Summary

- V45-01 (HIGH): GSMArena _select_main_lens regex split corrupts decimal MP data — verified by execution
