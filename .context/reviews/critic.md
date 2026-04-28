# Critic Review (Cycle 20) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## C20-CR01: Sony FX series misnaming causes dedup failures with other sources
**Severity:** MEDIUM | **Confidence:** HIGH

The `_parse_camera_name` function in imaging_resource.py produces "Sony Fx3" instead of "Sony FX3". This is not just a cosmetic issue — it affects data quality. Users searching for "Sony FX3" won't find it. Other sources (Apotelyt, GSMArena) may produce the correct name, creating a name mismatch that prevents proper deduplication in the merge step. This amplifies the bug from a cosmetic issue to a data integrity issue.

**Amplification:** If Imaging Resource names it "Sony Fx3" but Apotelyt names it "Sony FX3", the merge treats them as different cameras (different keys), creating a duplicate entry on the All Cameras page.

---

## C20-CR02: `pixel_pitch` crash risk on corrupted source data
**Severity:** MEDIUM | **Confidence:** HIGH

The `pixel_pitch()` ZeroDivisionError (C20-01) is a crash risk that could halt the entire CI pipeline. While mpix=0 is unlikely from valid data, corrupted or malformed HTML from a source page could produce unexpected values. The `derive_spec` function has no try/except around the `pixel_pitch` call, so any error propagates up and crashes `render_html`.

**Concrete scenario:** A source page has "0.0 Megapixels" (perhaps from a placeholder or data entry error). The regex matches it, producing mpix=0.0. `derive_spec` calls `pixel_pitch(area, 0.0)` and crashes.

---

## C20-CR03: Merge field preservation inconsistency — year is special-cased but other fields are not
**Severity:** LOW | **Confidence:** HIGH

The merge function has explicit year-preservation logic (if new year is None, keep existing year). But type, size, and pitch have no such logic. This creates an inconsistency: year is treated as a "best known value" while other fields are treated as "newest value wins". This design choice should be intentional and documented, or all fields should have consistent preservation behavior.

---

## Summary

- C20-CR01 (MEDIUM): Sony FX misnaming causes dedup failures with other sources
- C20-CR02 (MEDIUM): pixel_pitch crash risk on corrupted data
- C20-CR03 (LOW): Inconsistent field preservation in merge
