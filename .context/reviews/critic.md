# Critic Review (Cycle 12) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-11 fixes

## Previously Fixed (Cycles 1-11) — Confirmed Resolved
All previous findings addressed or properly deferred. No regressions. CR11-01 (year in key) was fixed in cycle 11. CR11-02 (about title) was fixed in cycle 11.

## New Findings

### CR12-01: `parse_existing_csv` name field not stripped — inconsistency with category/type handling
**File:** `pixelpitch.py`, lines 277, 291
**Severity:** MEDIUM | **Confidence:** HIGH

C10-01 fixed the type field, C11-02 fixed the category field — both by adding `.strip()`. The name field was never fixed. This inconsistency means that while category and type are resilient to whitespace in CSV data, the name field is not. A manually edited CSV with leading/trailing whitespace in the name column would produce a camera with visible whitespace in its name on the website.

The `create_camera_key` function applies `.lower().strip()` to the name, so deduplication still works correctly. But the display name in the table, search links, and page title would all show the whitespace. This is a user-facing defect.

**Fix:** Add `.strip()` to the name field parsing at lines 277 and 291.

---

### CR12-02: `_parse_camera_name` Sony slug extraction incorrect for legacy IR URLs — produces "Cameras" as name
**File:** `sources/imaging_resource.py`, line 151
**Severity:** MEDIUM | **Confidence:** HIGH

The Sony-specific branch of `_parse_camera_name` extracts the camera slug using `fallback_url.rstrip('/').rsplit('/', 2)[-2]`. This was designed assuming the URL always has the form `.../slug/specifications/` (the format produced by `_spec_url`). However, `_gather_review_urls` also matches legacy spec URLs via `LEGACY_SPEC_URL_RE`, which have the form `.../slug-specifications/` (no `/specifications/` suffix). For these legacy URLs, `rsplit('/', 2)[-2]` returns the parent path segment `'cameras'` instead of the slug.

For example:
- Modern spec URL: `.../sony-a7-iv-review/specifications/` → `rsplit('/', 2)[-2]` = `'sony-a7-iv-review'` (correct)
- Legacy spec URL: `.../sony-a7-iv-specifications/` → `rsplit('/', 2)[-2]` = `'cameras'` (wrong)

The non-Sony fallback at line 165 uses `rsplit('/', 1)[-1]` which works correctly for both URL formats.

**Fix:** Use `rsplit('/', 1)[-1]` consistently, or add format detection to handle legacy URLs.

---

### CR12-03: `_load_per_source_csvs` doesn't catch `UnicodeDecodeError` — crashes render pipeline
**File:** `pixelpitch.py`, line 754
**Severity:** LOW | **Confidence:** HIGH

The function catches `OSError` but not `UnicodeDecodeError`. Since `UnicodeDecodeError` is a subclass of `ValueError` (not `OSError`), a corrupt UTF-8 CSV file would crash `render_html`. The CI's `continue-on-error` expects source failures to be graceful. Self-written CSVs are always valid UTF-8, so this only affects manually edited or corrupted files.

**Fix:** Add `UnicodeDecodeError` to the except clause.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- CR12-01: Name field whitespace not stripped — MEDIUM
- CR12-02: Sony slug extraction fails for legacy IR URLs — MEDIUM
- CR12-03: UnicodeDecodeError not caught — LOW
