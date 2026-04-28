# Tracer Review (Cycle 26) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes

## Previous Findings Status

C25-01 and C25-02 fixes verified. TR25-01 (ValueError trace) and TR25-02 (SIZE_RE × gap) both addressed.

## New Findings

### TR26-01: MPIX_RE gap traced — silent data loss if Geizhals format changes

**File:** `pixelpitch.py` line 42, consumed by `extract_specs()` line 596
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace:**
1. Geizhals HTML contains a megapixel field with text like `"33.0 MP"` (abbreviated instead of full "Megapixel")
2. `MPIX_RE.search("33.0 MP")` returns None (local pattern only matches "Megapixel")
3. `mpix` stays None
4. `derive_spec()` computes `area` from size, but `pixel_pitch()` is not called because `spec.mpix is None`
5. Camera shows "unknown" resolution and computed pixel pitch is missing
6. Camera appears in the "Unknown Pixel Pitch" section on the website

**Competing hypothesis:** Geizhals currently uses "Megapixel" in German-language HTML. A format change is unlikely but possible, especially since the shared `MPIX_RE` in `sources/__init__.py` already handles "MP".

**Root cause:** The C25-01 fix centralized SIZE_RE and PITCH_RE but missed MPIX_RE, leaving an inconsistency.

### TR26-02: ValueError in source modules traced — individual camera lost

**File:** `sources/apotelyt.py` lines 119-120, `sources/cined.py` line 98
**Severity:** LOW | **Confidence:** MEDIUM

**Causal trace:**
1. Apotelyt HTML contains a malformed sensor size like `"35.9.1x23.9 mm"`
2. `SIZE_RE.search()` matches group(1) = `"35.9.1"`, group(2) = `"23.9"`
3. `float("35.9.1")` raises ValueError
4. ValueError propagates out of `fetch_one()` → caught by outer loop in `fetch()`
5. That camera record is lost, but other cameras in the batch continue

**Mitigating factor:** The outer `try/except Exception` in each source's `fetch()` catches the error for individual cameras. Blast radius is limited to one camera.

---

## Summary

- TR26-01 (MEDIUM): MPIX_RE gap — silent data loss if Geizhals uses "MP" abbreviation
- TR26-02 (LOW): ValueError in source modules — individual camera lost on malformed input
