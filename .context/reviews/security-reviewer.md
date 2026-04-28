# Security Review (Cycle 43)

**Reviewer:** security-reviewer
**Date:** 2026-04-28

## Previous Findings Status

C42-02 (CLI `--limit` validation) implemented and verified. All prior security findings remain deferred (LOW severity). Jinja2 autoescape enabled, SRI hashes present, importlib whitelist in place.

## New Findings

### SR43-01: GSMArena and CineD set `spec.size` from lookup tables — data integrity risk masquerading as measured data

**File:** `sources/gsmarena.py`, line 146; `sources/cined.py`, lines 94-102
**Severity:** MEDIUM | **Confidence:** HIGH

GSMArena's `_phone_to_spec` sets `spec.size` from `PHONE_TYPE_SIZE` (a lookup table derived from `TYPE_SIZE`). CineD's `_parse_camera_page` sets `spec.size` from `FORMAT_TO_MM`. These approximate values are stored in the `spec.size` field which is treated as authoritative measured data by `merge_camera_data`.

From a data integrity perspective, this means:
1. Approximate values override measured values in the merge without any provenance tracking
2. The CSV stores these as if they were measured values
3. Users see them as accurate sensor dimensions in the template

This is not a traditional security vulnerability (XSS, injection, etc.) but a data integrity issue — the system presents approximate data as fact, silently replacing more accurate measurements.

**Fix:** Same as CR43-02 — GSMArena and CineD should not set `spec.size` from lookup tables. Use `spec.type` and let `derive_spec` compute `derived.size`.

---

## Summary

- SR43-01 (MEDIUM): Data integrity risk — GSMArena/CineD lookup-table sizes masquerade as measured data in merge
- No new high/critical security findings
