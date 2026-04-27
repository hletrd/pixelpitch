# Architect Review (Cycle 12) — Architectural/Design Risks, Coupling, Layering

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository architecture review after cycles 1-11 fixes

## Previously Noted (Deferred, Still Valid)
- F32: `pixelpitch.py` is a ~1057-line monolith — DEFERRED
- F31: No source Protocol/base class — DEFERRED
- A5-02: Template description blocks DRY violation — DEFERRED

## Previously Fixed (Cycles 1-11)
- A11-01: create_camera_key year coupling — FIXED (year removed from key)

## New Findings

### A12-01: `parse_existing_csv` field stripping is inconsistent — only some fields are stripped
**File:** `pixelpitch.py`, lines 267-301
**Severity:** LOW | **Confidence:** HIGH

The CSV parser applies `.strip()` to the type field (C10-01) and the category field (C11-02), but NOT to the name field. This inconsistency suggests a pattern where each field was fixed reactively rather than proactively. The architectural concern is: there's no systematic approach to field sanitization in the CSV parser. Each field's whitespace handling depends on whether a bug was found for that specific field.

**Fix:** Apply `.strip()` to ALL string fields consistently in `parse_existing_csv`. This would also fix C12-01 (name field whitespace) as a side effect.

---

### A12-02: `_parse_camera_name` Sony branch has different URL parsing logic than non-Sony fallback — fragile
**File:** `sources/imaging_resource.py`, lines 151-167
**Severity:** MEDIUM | **Confidence:** HIGH

The Sony branch uses `rsplit('/', 2)[-2]` for slug extraction, while the non-Sony fallback uses `rsplit('/', 1)[-1]`. These are different parsing strategies that assume different URL formats. The Sony branch assumes the URL has a `/specifications/` suffix (which is only true for modern review URLs, not legacy spec URLs). This creates a fragile coupling between `_spec_url` behavior and `_parse_camera_name` assumptions.

The architectural issue is that `_parse_camera_name` makes assumptions about the URL structure that aren't guaranteed by the interface contract. If `_gather_review_urls` returns a URL format that doesn't match the assumption, the name extraction silently produces wrong results.

**Fix:** Use a single robust slug extraction strategy that works for all URL formats. Extract the last path component with `rsplit('/', 1)[-1]`, then strip known suffixes (review, specifications, etc.).

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- A12-01: Inconsistent field stripping in CSV parser — LOW
- A12-02: Sony URL parsing assumes specific URL structure — MEDIUM
