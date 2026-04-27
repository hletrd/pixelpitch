# Tracer Review (Cycle 16) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
All previous fixes remain intact.

## New Findings

### T16-01: Crash propagation path: malformed sensor type -> derive_spec -> derive_specs -> render_html
**File:** `pixelpitch.py`, lines 152-165, 611-635, 638-640
**Severity:** MEDIUM | **Confidence:** HIGH

Traced crash path:
1. Source HTML contains sensor type `1/0"` (malformed)
2. `SENSOR_TYPE_RE` matches and extracts `1/0`
3. Source parser sets `spec.type = "1/0"`, `spec.size = None`
4. `derive_spec(spec)` calls `sensor_size_from_type("1/0")`
5. `1 / float("0")` raises `ZeroDivisionError`
6. `derive_specs()` list comprehension propagates the crash
7. `render_html()` calls `derive_specs()` -> crash
8. No HTML output, CI build fails

The fix should be at step 5: `sensor_size_from_type` must catch arithmetic errors.

---

### T16-02: Duplicate propagation path: same camera from multiple sources -> merge_camera_data -> visible duplicate
**File:** `pixelpitch.py`, lines 349-407
**Severity:** MEDIUM | **Confidence:** HIGH

Traced duplicate path:
1. Geizhals fetches "Canon EOS 250D" as DSLR
2. openMVG classifies "Canon EOS 250D" as DSLR (after C15-01 fix)
3. Both appear in `new_specs_all` with key "canon eos 250d-dslr"
4. `merge_camera_data` iterates new_specs linearly
5. First entry: not in existing_by_key -> appended
6. Second entry: not in existing_by_key -> also appended
7. Result: 2 entries for "Canon EOS 250D" in merged data
8. Both appear on All Cameras page

The fix should be at step 4-6: track seen keys among new_specs.

---

### T16-03: Pentax classification gap: models without hyphen -> missed by regex -> mirrorless misclassification
**File:** `sources/openmvg.py`, line 47
**Severity:** LOW | **Confidence:** HIGH

Traced misclassification path:
1. openMVG CSV contains "Pentax K3" (no hyphen)
2. `_DSLR_NAME_RE.search("Pentax K3")` returns None
3. Category defaults to "mirrorless"
4. On All Cameras page, "Pentax K3" appears under Mirrorless instead of DSLR
5. If Geizhals also lists "Pentax K3" as DSLR, duplicate entries appear

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- T16-01: Crash propagation from malformed sensor type — MEDIUM
- T16-02: Duplicate propagation from multi-source same-category cameras — MEDIUM
- T16-03: Pentax classification gap — LOW
