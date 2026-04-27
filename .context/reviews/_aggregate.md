# Aggregate Review (Cycle 12) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-11 Status

All previous fixes confirmed still working. No regressions.
Deferred items: see `.omc/plans/C7-02-deferred.md` and other deferred files.

## Cross-Agent Agreement Matrix (Cycle 12 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `parse_existing_csv` name field not stripped | code-reviewer, critic, verifier, tracer, debugger, architect, test-engineer | MEDIUM |
| `_parse_camera_name` Sony slug extraction fails for legacy IR URLs | code-reviewer, critic, verifier, tracer, debugger, architect, test-engineer | MEDIUM |
| `_load_per_source_csvs` doesn't catch UnicodeDecodeError | code-reviewer, critic, verifier, tracer, debugger | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C12-01: `parse_existing_csv` name field not stripped — whitespace in camera names
**Sources:** C12-01, CR12-01, V12-01, T12-02, D12-02, A12-01, TE12-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

`pixelpitch.py` lines 277, 291: The name field is parsed without `.strip()`. This is the same pattern that was fixed for the type field (C10-01) and the category field (C11-02). While `create_camera_key` applies `.lower().strip()` so deduplication still works, the displayed name on the website would show leading/trailing whitespace.

**Concrete failure:** A CSV with `" Sony A7 IV "` as the name would display with visible whitespace in the table, search links, and scatter plot tooltips.

**Fix:** Add `.strip()` to the name field parsing at lines 277 and 291.

---

### C12-02: `_parse_camera_name` Sony slug extraction fails for legacy IR spec URLs
**Sources:** C12-02, CR12-02, V12-02, T12-01, D12-01, A12-02, TE12-02
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

`sources/imaging_resource.py` line 151: The Sony branch uses `fallback_url.rstrip('/').rsplit('/', 2)[-2]` to extract the camera slug. This assumes the URL has an extra path segment after the slug (e.g. `/specifications/`). However, legacy spec URLs matched by `LEGACY_SPEC_URL_RE` have the form `.../slug-specifications/` without a trailing `/specifications/` segment. For these URLs, `rsplit('/', 2)[-2]` returns the parent directory name `'cameras'` instead of the slug.

**Concrete failure:** A Sony camera discovered via a legacy Imaging Resource spec URL (e.g. `https://www.imaging-resource.com/cameras/sony-zv-e10-specifications/`) would be named `"Cameras"` on the website.

**Fix:** Use `rsplit('/', 1)[-1]` in the Sony branch (consistent with the non-Sony fallback at line 165), then strip known suffixes with the existing regex.

---

### C12-03: `_load_per_source_csvs` doesn't catch `UnicodeDecodeError`
**Sources:** C12-03, CR12-03, V12-03, T12-03, D12-03
**Severity:** LOW | **Confidence:** HIGH (5-agent consensus)

`pixelpitch.py` line 754: `path.read_text(encoding='utf-8')` can raise `UnicodeDecodeError` for corrupt files. The `except OSError` at line 755 doesn't catch it (`UnicodeDecodeError` is a subclass of `ValueError`, not `OSError`). This would crash `render_html`.

**Concrete failure:** A corrupt source CSV file (non-UTF-8 bytes) causes `UnicodeDecodeError` during the render pipeline, preventing site generation.

**Fix:** Add `UnicodeDecodeError` to the except clause at line 755.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 3 (2 MEDIUM, 1 LOW)
- Cross-agent consensus findings (3+ agents): 3
- All cycle 1-11 fixes remain intact
- Remaining deferred items: see `.omc/plans/C7-02-deferred.md`
