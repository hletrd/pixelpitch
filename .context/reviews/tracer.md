# Tracer Review (Cycle 20) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## T20-01: Data flow from source to pixel_pitch — crash path traced

**Trace:**
1. `fetch_source()` calls `module.fetch()` which returns `list[Spec]`
2. `derive_specs()` calls `derive_spec(spec, sensors_db)` for each
3. `derive_spec()` checks `if spec.mpix is not None and area is not None`
4. If true, calls `pixel_pitch(area, spec.mpix)`
5. `pixel_pitch()` computes `sqrt(area / (mpix * 10**6))`
6. If `mpix == 0`: ZeroDivisionError
7. If `mpix < 0`: ValueError (sqrt of negative)

**Root cause:** No guard against non-positive mpix in `pixel_pitch()`.
**Fix point:** Add guard at step 5 (in `pixel_pitch` itself, not in `derive_spec`).

---

## T20-02: Sony FX name corruption flow traced

**Trace:**
1. GSMArena phone page: `_parse_camera_name` not called (GSMArena uses `<h1>` tag)
2. IR spec page: `_parse_camera_name({'Model Name': 'Sony FX3'}, url)`
3. Name starts with "Sony" -> enters Sony-specific branch
4. URL slug extracted: "sony-fx3" -> `.replace("-", " ")` -> "sony fx3"
5. `.title()` -> "Sony Fx3" (lowercase 'x' capitalized)
6. ZV normalization: `cleaned.replace("Sony Zv ", "Sony ZV-")` — no match
7. No FX normalization -> returns "Sony Fx3"

**Root cause:** `.title()` capitalizes every word-initial letter including 'x' in 'fx'.
**Fix point:** Add FX normalization after ZV normalization.

---

## Summary

- T20-01: pixel_pitch crash path fully traced — fix at `pixel_pitch()` function
- T20-02: Sony FX name corruption path fully traced — fix at `_parse_camera_name()`
