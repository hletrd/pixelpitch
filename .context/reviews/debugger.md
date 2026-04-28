# Debugger Review (Cycle 28) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes

## Previous Findings Status

DBG27-01 (PITCH_UM_RE "um") and DBG27-02 (year=0) both fixed in C27.

## New Findings

### DBG28-01: imaging_resource.py pitch float() — unhandled ValueError (C26-02 regression by omission)

**File:** `sources/imaging_resource.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** The C26-02 fix added ValueError guards to `size` and `mpix` but missed `pitch`. If Imaging Resource serves a pixel pitch value containing multiple dots (e.g., "5.1.2 microns"), `float("5.1.2")` raises `ValueError`, which is NOT caught. The exception propagates through `fetch_one()`, aborting the current camera and potentially the entire `fetch()` loop (which has no per-camera try/except).

**Root cause:** Incomplete C26-02 fix — the guard was added to two of three float() calls.

**Verified:** `IR_PITCH_RE.search("5.1.2 microns")` matches "5.1.2"; `float("5.1.2")` raises `ValueError`.

### DBG28-02: CineD year regex can produce invalid years without validation

**File:** `sources/cined.py`, line 114
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** The regex `r"Release Date.{0,40}?(\d{4})"` matches any 4-digit number, including years like 0000, 1234, or 9999. The `int()` conversion succeeds, and the year is stored in Spec without validation. The template renders this year verbatim.

The C27-02 fix added range validation to `parse_existing_csv()` (1900-2100), but `cined._parse_camera_page()` produces years before they reach the CSV parser.

**Root cause:** Missing year range validation in the source module.

---

## Summary

- DBG28-01 (MEDIUM): imaging_resource.py pitch float() unhandled ValueError — C26-02 incomplete
- DBG28-02 (LOW): CineD year regex produces unvalidated years
