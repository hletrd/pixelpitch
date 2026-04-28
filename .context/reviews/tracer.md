# Tracer Review (Cycle 28) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes

## Previous Findings Status

TR27-01 (PITCH_UM_RE "um" gap) and TR27-02 (parse_existing_csv year=0) both fixed in C27.

## New Findings

### TR28-01: imaging_resource.py pitch float() — unhandled ValueError crash trace

**File:** `sources/imaging_resource.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Imaging Resource serves a camera spec page with malformed pixel pitch text, e.g., "Approximate Pixel Pitch: 5.1.2 microns"
2. `IR_PITCH_RE.search("5.1.2 microns")` matches group(1) = "5.1.2"
3. `float("5.1.2")` raises `ValueError: could not convert string to float: '5.1.2'`
4. Exception propagates through `fetch_one()` → `fetch()` loop
5. The `fetch()` function has no per-camera try/except — the entire scrape aborts
6. Zero Imaging Resource cameras are fetched; `camera-data-imaging-resource.csv` is not written

**Competing hypothesis:** Is it realistic for IR to serve "5.1.2 microns"? The `([\d.]+)` pattern matches any sequence of digits and dots. If the page has "5.1.2 microns" or if the HTML has an embedded decimal (e.g., from formatting), this could happen. More likely, the site could have a value like "5 µm" with no decimal (matched as just "5"), which would work fine. The malformed case is unlikely but not impossible.

**Root cause:** The C26-02 fix was incomplete — it added guards to `size` and `mpix` but missed `pitch`.

---

## Summary

- TR28-01 (MEDIUM): imaging_resource.py pitch float() unhandled ValueError — can abort entire scrape
