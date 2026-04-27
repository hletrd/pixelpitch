# Debugger Review (Cycle 12) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-11 fixes

## Previously Fixed (Cycles 1-11) — Confirmed Resolved
All previous fixes verified. No regressions detected. D11-01 (year in key) fixed. D11-02 (category strip) fixed. D11-03 (year overwrite logging) fixed.

## New Findings

### D12-01: `_parse_camera_name` Sony slug extraction produces "Cameras" for legacy IR spec URLs
**File:** `sources/imaging_resource.py`, line 151
**Severity:** MEDIUM | **Confidence:** HIGH

Failure mode:
1. `_gather_review_urls` matches a legacy spec URL: `.../cameras/sony-zv-e10-specifications/`
2. `_spec_url` returns it unchanged (already has `-specifications`)
3. `fetch_one` receives the legacy URL
4. `_parse_camera_name` Sony branch: `fallback_url.rstrip('/').rsplit('/', 2)[-2]` = `'cameras'`
5. `slug = 'cameras'` → `.title()` = `'Cameras'` → name = `'Cameras'`
6. Camera appears on website with name "Cameras" — clearly wrong and confusing

This only happens when the source discovers a legacy spec URL (matched by `LEGACY_SPEC_URL_RE`). For modern review URLs, `_spec_url` adds `/specifications/` and the extraction works correctly.

**Failure scenario:** Fetch from Imaging Resource with legacy spec URLs present. Sony cameras from legacy URLs appear as "Cameras" on the website.

**Fix:** Use `rsplit('/', 1)[-1]` in the Sony branch, then strip suffixes with regex.

---

### D12-02: `parse_existing_csv` name field not stripped — latent whitespace bug
**File:** `pixelpitch.py`, lines 277, 291
**Severity:** MEDIUM | **Confidence:** HIGH

Same pattern as the fixed C10-01 (type) and C11-02 (category) bugs. The name field is the last string field without `.strip()`. While `write_csv` never introduces whitespace, manually edited CSVs could. The `create_camera_key` applies `.strip()` so deduplication still works, but the displayed name on the website would have visible whitespace.

**Failure scenario:** CSV with `" Sony A7 IV "` as name → camera displays with leading/trailing spaces in table, search links, and scatter plot tooltips.

**Fix:** Add `.strip()` to name field parsing.

---

### D12-03: `_load_per_source_csvs` crashes on UnicodeDecodeError
**File:** `pixelpitch.py`, line 754
**Severity:** LOW | **Confidence:** HIGH

`path.read_text(encoding='utf-8')` raises `UnicodeDecodeError` for corrupt files. The `except OSError` at line 755 doesn't catch it. This would crash `render_html` and prevent site generation.

**Failure scenario:** A source CSV file gets corrupted (non-UTF-8 bytes introduced). The next CI run crashes during render.

**Fix:** Add `UnicodeDecodeError` to the except clause.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- D12-01: Sony slug extraction fails for legacy IR URLs — MEDIUM
- D12-02: Name field not stripped in parse_existing_csv — MEDIUM
- D12-03: UnicodeDecodeError not caught in source CSV loading — LOW
