# Tracer Review (Cycle 30) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-29 fixes

## Previous Findings Status

TR29-01 and TR29-02 both fixed in C29. All previous fixes stable.

## New Findings

### TR30-01: GSMArena fetch() crash propagation — no per-phone try/except

**File:** `sources/gsmarena.py`, lines 246-252
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. `fetch_phone()` raises an unhandled exception (e.g., from unexpected HTML structure, or a future code change)
2. Exception propagates through `fetch()` loop — no per-phone try/except
3. The entire GSMArena scrape aborts
4. `camera-data-gsmarena.csv` is not written
5. Existing data from previous runs is preserved via `merge_camera_data`, but new data is lost

**Competing hypothesis:** Is it realistic for `fetch_phone()` to raise? The function already handles most parsing errors gracefully, but `_phone_to_spec()` could raise an `AttributeError` or `TypeError` from unexpected HTML structure (e.g., a missing spec table row that causes a `None` value to be indexed).

**Fix:** Add per-phone try/except in `fetch()`, consistent with the CineD/IR/Apotelyt pattern.

---

## Summary

- TR30-01 (MEDIUM): GSMArena fetch() crash propagation — no per-phone try/except
