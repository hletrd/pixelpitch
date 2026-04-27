# Tracer Review (Cycle 12) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-11 fixes

## Traced Flows

### Flow 1: Legacy IR spec URL through _parse_camera_name (Sony branch)
**Path:** `_gather_review_urls` matches legacy URL via `LEGACY_SPEC_URL_RE` → URL like `.../cameras/sony-zv-e10-specifications/` → `fetch_one` receives this URL → `_spec_url` returns it unchanged (already has `-specifications`) → `_parse_camera_name` called with this URL as `fallback_url` → Sony branch: `fallback_url.rstrip('/').rsplit('/', 2)[-2]` → For URL `.../cameras/sony-zv-e10-specifications/`, rsplit('/', 2) produces `['https://www.imaging-resource.com', 'cameras', 'sony-zv-e10-specifications']` → `[-2]` = `'cameras'` → slug = `'cameras'` → `.title()` = `'Cameras'` → camera name = `'Cameras'` → WRONG

Compared with modern URL: `.../cameras/sony-zv-e10-review/specifications/` → rsplit('/', 2) = `['https://www.imaging-resource.com/cameras', 'sony-zv-e10-review', 'specifications']` → `[-2]` = `'sony-zv-e10-review'` → correct

**FINDING: T12-01** — Sony slug extraction broken for legacy IR spec URLs. The `rsplit('/', 2)[-2]` assumes an extra path segment after the slug, which legacy URLs don't have.

**Severity:** MEDIUM | **Confidence:** HIGH

---

### Flow 2: Name field whitespace through CSV parse to display
**Path:** Manually edited CSV has `name = " Sony A7 IV "` → `parse_existing_csv` line 277: `name = values[1]` (no `.strip()`) → `Spec(name=' Sony A7 IV ', ...)` → `create_camera_key` at line 336: `spec.name.lower().strip()` = `'sony a7 iv'` (key is correct) → but `spec.name` still has whitespace → `write_csv` writes ` Sony A7 IV ` → template renders `{{ spec.spec.name }}` = ` Sony A7 IV ` with visible spaces

**FINDING: T12-02** — Name field not stripped in parse_existing_csv. While deduplication keys are correct (key applies `.strip()`), the display name preserves whitespace.

**Severity:** MEDIUM | **Confidence:** HIGH

---

### Flow 3: Corrupt UTF-8 source CSV through render pipeline
**Path:** Manually corrupted source CSV with non-UTF-8 bytes → `_load_per_source_csvs` at line 754: `path.read_text(encoding='utf-8')` raises `UnicodeDecodeError` → `except OSError` at line 755 does NOT catch it → exception propagates up → `render_html` crashes → site not built

**FINDING: T12-03** — UnicodeDecodeError not caught in _load_per_source_csvs, crashing the render pipeline.

**Severity:** LOW | **Confidence:** HIGH

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- T12-01: Sony slug extraction broken for legacy IR URLs — MEDIUM
- T12-02: Name field not stripped in parse_existing_csv — MEDIUM
- T12-03: UnicodeDecodeError not caught in source CSV loading — LOW
