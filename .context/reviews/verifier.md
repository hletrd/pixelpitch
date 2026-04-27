# Verifier Review (Cycle 11) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository verification after cycles 1-10 fixes

## Previously Fixed (Cycles 1-10) — Verification Status
All previous fixes verified still working. Gate tests pass cleanly.

## New Findings

### V11-01: `create_camera_key` year mismatch produces duplicate cameras — verified via code trace
**File:** `pixelpitch.py`, lines 313-315, 318-367
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by tracing the code path:

1. openMVG `fetch()` returns specs with `year=None` for every camera
2. `fetch_source` calls `derive_specs(raw_specs)` which calls `derive_spec` for each
3. `derive_spec` preserves `year=None` from the Spec
4. `render_html` calls `_load_per_source_csvs` which calls `parse_existing_csv`
5. `parse_existing_csv` preserves the year column — openMVG CSVs have empty year
6. In `merge_camera_data`, `create_camera_key(spec)` produces keys like `"canon eos 5d-mirrorless-unknown"` for openMVG cameras
7. The same camera from Geizhals produces `"canon eos 5d-mirrorless-2005"`
8. These are different keys, so the merge treats them as different cameras
9. Both cameras are added to the merged output

The resulting duplicate cameras appear on the website. I verified that `create_camera_key` is the ONLY deduplication mechanism in `merge_camera_data` — there's no secondary check by name+category.

**Concrete failure:** The digicamdb source (which wraps openMVG) will produce cameras with year=None. If any of these cameras also exist in the Geizhals data (which has years), they will be duplicated in the output.

**Fix:** Remove year from `create_camera_key`. The name+category is sufficient.

---

### V11-02: `parse_existing_csv` doesn't validate header column names — silent misparse on schema change
**File:** `pixelpitch.py`, lines 230-310
**Severity:** LOW | **Confidence:** MEDIUM

The CSV parser detects `has_id` based on `header[0] == "id"`, but then uses positional indexing for all other fields. If a column is added or reordered in the CSV schema, the positional mapping silently breaks. For example, if a "sensor_type" column were inserted between "type" and "sensor_width_mm", all fields after "type" would shift by one position, and the parser would read wrong values without any error.

This is unlikely in practice because `write_csv` and `parse_existing_csv` are maintained together, but it's a fragility worth noting. The fix would be to validate that header names match expected values.

**Fix:** Low priority — add header validation or switch to DictReader.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- V11-01: create_camera_key year mismatch — verified duplicates — MEDIUM
- V11-02: CSV parser positional indexing — schema change fragility — LOW
