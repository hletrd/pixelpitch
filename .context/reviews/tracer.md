# Tracer Review (Cycle 11) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-10 fixes

## Traced Flows

### Flow 1: openMVG camera through merge_camera_data with year=None
**Path:** openMVG `fetch()` → Spec with `year=None` → `fetch_source` → `derive_specs` → `derive_spec` preserves year=None → `write_csv` writes empty year → `parse_existing_csv` reads empty year as `year=None` → `merge_camera_data` → `create_camera_key(spec)` produces `"name-category-unknown"` → same camera from Geizhals with year=2021 produces `"name-category-2021"` → different keys → both added to merged result → DUPLICATE

**FINDING: T11-01** — The year component in `create_camera_key` causes duplicate cameras when sources disagree on the year. This is a real bug that affects every camera that appears in both openMVG and Geizhals/IR.

**Severity:** MEDIUM | **Confidence:** HIGH

---

### Flow 2: CSV round-trip with empty category field
**Path:** `write_csv` writes `category=""` when spec.category is None (but Spec.category is required, so it's always present) → `parse_existing_csv` reads `category = values[2]` → no `.strip()` applied → if whitespace present, it passes through unchanged → `render_html` filters by `spec.spec.category == "mirrorless"` → whitespace-prefixed category like `" mirrorless"` would not match → camera excluded from category page

**FINDING: T11-02** — Same pattern as the C10-01 type whitespace fix, but for the category field. Low risk because `write_csv` never introduces whitespace.

**Severity:** LOW | **Confidence:** HIGH

---

### Flow 3: about.html title rendering through template inheritance
**Path:** `render_html` line 899 → `_get_env().get_template("about.html").render(page="about", date=date)` → about.html overrides `{% block title %}About Pixel Pitch{% endblock %}` → base template `index.html` renders `{% block title %}{{ title | default('Camera Sensor Pixel Pitch List') }}{% endblock %}` → about.html's override takes precedence → "About Pixel Pitch" is rendered correctly

No `title` variable is passed in the render context, but it's not needed because the block override works. However, if the about.html template's block override were accidentally removed, the fallback would produce the generic title.

**FINDING:** No bug, but a defensive improvement would be to pass `title="About Pixel Pitch"` in the render context.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- T11-01: create_camera_key year mismatch — verified duplicate camera flow — MEDIUM
- T11-02: Category field whitespace not stripped — LOW (same as C10-01 pattern)
