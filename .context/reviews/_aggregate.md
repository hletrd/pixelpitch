# Aggregate Review (Cycle 11) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-10 Status

All previous fixes confirmed still working. No regressions.
Deferred items: see `.context/plans/deferred.md`.

## Cross-Agent Agreement Matrix (Cycle 11 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `create_camera_key` year mismatch causes duplicates across sources | code-reviewer, critic, verifier, test-engineer, tracer, architect, debugger | MEDIUM |
| Category field whitespace not stripped in parse_existing_csv | code-reviewer, tracer, debugger | LOW |
| about.html no `title` in render context (defensive) | critic | LOW |
| No test for create_camera_key year mismatch | test-engineer | LOW (MEDIUM for test gap) |
| No integration test for render_html | test-engineer | LOW |
| sensors.json schema undocumented | document-specialist | LOW |
| openMVG not listed as data source | document-specialist | LOW |
| Scatter plot year axis label overlap | designer | LOW |
| "Hide possibly invalid data" label unclear | designer | LOW |
| CSV parser positional indexing — schema change fragility | verifier | LOW |
| merge_camera_data year overwrite without warning | debugger | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C11-01: `create_camera_key` year mismatch causes duplicate cameras across sources
**Sources:** C11-01, CR11-01, V11-01, TE11-01, T11-01, A11-01, D11-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

`pixelpitch.py` lines 313-315: `create_camera_key` includes the year in the key: `f"{spec.name.lower().strip()}-{spec.category}-{year}"`. When `year is None`, the key uses the string `"unknown"`. Since openMVG always provides `year=None`, any camera that appears in both openMVG and another source (which provides the year) will have different keys and thus be treated as separate cameras by `merge_camera_data`.

**Concrete failure:** Camera "Canon EOS 5D" from Geizhals (year=2005) gets key `"canon eos 5d-dslr-2005"`, but from openMVG (year=None) gets key `"canon eos 5d-dslr-unknown"`. Both appear in the merged output, producing duplicates on the website.

**Fix:** Remove the year from `create_camera_key`. The name+category combination is sufficient for deduplication. The merge code already handles year preservation (line 343-344).

---

### C11-02: Category field whitespace not stripped in `parse_existing_csv`
**Sources:** C11-02, T11-02, D11-02
**Severity:** LOW | **Confidence:** HIGH (3-agent consensus)

`pixelpitch.py` lines 262-263, 275-276: Same pattern as the C10-01 type field fix. The category field is parsed without `.strip()`. A category like `" mirrorless"` would not match the category filter in `render_html` (line 793: `if s.spec.category == cat`), causing the camera to be excluded from its category page.

**Fix:** Add `.strip()` to category field parsing, same as was done for the type field.

---

### C11-03: about.html render context missing `title` parameter — defensive improvement
**Sources:** CR11-02
**Severity:** LOW | **Confidence:** HIGH

`pixelpitch.py` line 899: The about.html template is rendered without a `title` variable in the context. The template works correctly via Jinja2 block overrides, but passing `title="About Pixel Pitch"` would provide a safety net if block overrides were accidentally removed.

**Fix:** Add `title="About Pixel Pitch"` to the about.html render context.

---

### C11-04: No test for `create_camera_key` year mismatch — duplicate cameras
**Sources:** TE11-01
**Severity:** MEDIUM (test gap) | **Confidence:** HIGH

No test covers the scenario where the same camera from two sources has different years (e.g., year=2021 vs year=None). The existing `test_merge_camera_data` only tests overlapping cameras with the SAME year.

**Fix:** Add a test that merges the same camera from two sources where one has a year and the other doesn't, asserting that only one entry appears in the merged result.

---

### C11-05: No integration test for `render_html` output
**Sources:** TE11-02
**Severity:** LOW | **Confidence:** MEDIUM

No integration test calls `render_html` (or a subset) and verifies the HTML output. Regressions in template rendering (e.g., broken template blocks, missing variables) would not be caught by gate tests.

**Fix:** Add a minimal integration test that renders `pixelpitch.html` with a small set of specs.

---

### C11-06: `sensors.json` schema undocumented
**Sources:** DS11-01
**Severity:** LOW | **Confidence:** HIGH

The expected JSON schema of `sensors.json` is not documented. From code inspection, it requires `sensor_width_mm`, `sensor_height_mm`, and `megapixels` fields. If someone adds a sensor with different field names, `match_sensors` would silently skip it.

**Fix:** Add a docstring or comment documenting the expected schema.

---

### C11-07: openMVG not listed as a data source on the about page
**Sources:** DS11-02
**Severity:** LOW | **Confidence:** MEDIUM

The about page and the pixelpitch.html alert text list geizhals.eu, Imaging Resource, Apotelyt, GSMArena, and CineD as sources but omit openMVG/CameraSensorSizeDatabase, which is a primary bulk data source.

**Fix:** Add openMVG to the source list on both pages.

---

### C11-08: Scatter plot year axis label overlap
**Sources:** UX11-01
**Severity:** LOW | **Confidence:** MEDIUM

With many years of data (20+), the band scale on the x-axis produces overlapping year labels that become unreadable.

**Fix:** Rotate x-axis labels 45 degrees when there are many years, or use a time-based scale.

---

### C11-09: CSV parser positional indexing — schema change fragility
**Sources:** V11-02
**Severity:** LOW | **Confidence:** MEDIUM

`parse_existing_csv` uses positional indexing to map CSV columns. If a column is added or reordered in the schema, the mapping silently breaks. The parser detects `has_id` from the header but doesn't validate other column names.

**Fix:** Low priority — add header validation or switch to DictReader.

---

### C11-10: `merge_camera_data` year overwrite without warning
**Sources:** D11-03
**Severity:** LOW | **Confidence:** MEDIUM

The merge logic only preserves the existing year when the new data has no year. If the new data has a DIFFERENT year, the new year silently overwrites the existing year without any warning.

**Fix:** Consider logging when years differ during merge.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 10 (1 MEDIUM + 1 MEDIUM test gap, 8 LOW)
- Cross-agent consensus findings (3+ agents): 2
- All cycle 1-10 fixes remain intact
- Remaining deferred items: see `.context/plans/deferred.md`
