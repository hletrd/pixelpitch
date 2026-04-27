# Code Review (Cycle 12) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-11 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-11) — Confirmed Resolved
All previous fixes remain intact. C11-01 (year in key) through C11-10 (year overwrite) all verified as fixed or properly deferred.

## New Findings

### C12-01: `parse_existing_csv` doesn't strip whitespace from name field
**File:** `pixelpitch.py`, lines 277, 291
**Severity:** MEDIUM | **Confidence:** HIGH

C10-01 fixed the type field and C11-02 fixed the category field by adding `.strip()`. The name field has the same pattern: `name = values[1]` (has-id) or `name = values[0]` (no-id) without `.strip()`. A name like `" Sony A7 IV "` would display with leading/trailing whitespace on the website. While `write_csv` never introduces whitespace, manually edited CSVs could. More importantly, `create_camera_key` does `.lower().strip()` so the key would be correct, but the displayed name would have whitespace — confusing to users.

**Failure scenario:** A manually edited CSV has `" Sony A7 IV "` as the name. The camera appears on the website with visible leading/trailing spaces in its name. The search link and table display both show the whitespace.

**Fix:** Add `.strip()` to the name field parsing, consistent with category and type fields.

---

### C12-02: `_parse_camera_name` Sony slug extraction fails for legacy spec URLs
**File:** `sources/imaging_resource.py`, line 151
**Severity:** MEDIUM | **Confidence:** HIGH

The Sony branch of `_parse_camera_name` extracts the camera slug using `fallback_url.rstrip('/').rsplit('/', 2)[-2]`. This assumes the URL has the form `.../camera-slug/specifications/` (modern spec URL). However, `_gather_review_urls` also matches legacy spec URLs via `LEGACY_SPEC_URL_RE`, which have the form `.../camera-slug-specifications/`. For these legacy URLs, `rsplit('/', 2)[-2]` returns `'cameras'` (the parent path segment) instead of the slug, producing the camera name `"Cameras"` — clearly wrong.

**Failure scenario:** A Sony camera discovered via a legacy Imaging Resource spec URL (e.g. `https://www.imaging-resource.com/cameras/sony-zv-e10-specifications/`) would be named `"Cameras"` on the website, making it indistinguishable from other similarly broken entries and breaking the data.

**Fix:** Use `rsplit('/', 1)[-1]` consistently (like the non-Sony fallback at line 165), then strip the suffix with the existing regex. Or detect whether the URL ends with `/specifications/` and adjust the extraction accordingly.

---

### C12-03: `_load_per_source_csvs` doesn't catch `UnicodeDecodeError`
**File:** `pixelpitch.py`, line 754
**Severity:** LOW | **Confidence:** HIGH

`_load_per_source_csvs` catches `OSError` at line 755-756 but not `UnicodeDecodeError`. The `path.read_text(encoding='utf-8')` call raises `UnicodeDecodeError` (a subclass of `ValueError`, not `OSError`) if the file contains invalid UTF-8 bytes. Since `continue-on-error` in the CI workflow expects transient source failures to be gracefully handled, this unhandled exception could crash the entire render pipeline.

**Failure scenario:** A manually edited or corrupted source CSV with non-UTF-8 bytes causes `UnicodeDecodeError`, crashing `render_html` and preventing the site from being built.

**Fix:** Add `UnicodeDecodeError` to the except clause at line 755, or use `errors='replace'` in `read_text()`.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- C12-01: Name field whitespace not stripped in parse_existing_csv — MEDIUM
- C12-02: Sony slug extraction fails for legacy IR URLs — MEDIUM
- C12-03: UnicodeDecodeError not caught in _load_per_source_csvs — LOW
- All cycle 1-11 fixes remain intact
