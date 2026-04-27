# Critic Review (Cycle 11) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-10 fixes

## Previously Fixed (Cycles 1-10) — Confirmed Resolved
All previous findings addressed or properly deferred. No regressions.

## New Findings

### CR11-01: `create_camera_key` includes year — cross-source year mismatch causes duplicates
**File:** `pixelpitch.py`, lines 313-315
**Severity:** MEDIUM | **Confidence:** HIGH

This is the most significant new finding. `create_camera_key` includes the year: `f"{spec.name.lower().strip()}-{spec.category}-{year}"`. When `year is None`, it uses the string `"unknown"`. Since openMVG always has `year=None`, any camera that appears in both openMVG and another source (which provides the year) will have different keys and thus be treated as separate cameras by `merge_camera_data`.

Concrete example: "Canon EOS 5D" from Geizhals (year=2005) gets key `"canon eos 5d-dslr-2005"`, but from openMVG (year=None) gets key `"canon eos 5d-dslr-unknown"`. These are treated as different cameras, so both appear in the output.

The openMVG dataset contains thousands of cameras, many of which overlap with Geizhals and other sources. This means a significant number of duplicates could exist in the final merged dataset.

**Fix:** Remove the year from `create_camera_key`. The name+category combination is sufficient for deduplication. Year can be preserved through the merge logic's existing year-preservation code (line 343-344).

---

### CR11-02: `render_html` about page doesn't pass `title` to base template's og:title/twitter:title
**File:** `pixelpitch.py`, line 898-899
**Severity:** LOW | **Confidence:** HIGH

Line 899: `_get_env().get_template("about.html").render(page="about", date=date)`. The about.html template overrides `{% block title %}`, `{% block og_title %}`, and `{% block twitter_title %}` with "About Pixel Pitch", so these work correctly via template inheritance. However, no `title` variable is passed in the render context. If the about.html template's block overrides were ever removed or broken, the fallback `{{ title | default('Camera Sensor Pixel Pitch List') }}` would produce the generic title instead of "About Pixel Pitch".

This is defensive — the current code works correctly because the block overrides are in place. But passing `title="About Pixel Pitch"` in the render context would be an additional safety net.

**Fix:** Add `title="About Pixel Pitch"` to the about.html render context.

---

### CR11-03: `fetch_source` passes `limit` to module.fetch but gsmarena.fetch also accepts `sleep_seconds`, `brands`, `max_pages_per_brand`
**File:** `pixelpitch.py`, lines 933-955
**Severity:** LOW | **Confidence:** MEDIUM

`fetch_source` calls `module.fetch(limit=limit) if limit is not None else module.fetch()`. The gsmarena.fetch function accepts `sleep_seconds`, `brands`, and `max_pages_per_brand` kwargs, but `fetch_source` has no way to pass these through. The CI workflow (github-pages.yml) sets `GSMARENA_MAX_PAGES_PER_BRAND` as an env var but this env var is never read by the fetch code — it's dead config in the CI workflow (already noted as F23 in deferred).

This is not a bug (defaults are reasonable), but it means the CI workflow's `GSMARENA_PAGES` env var has no effect, which could confuse maintainers.

**Fix:** Already deferred as F23. No new action needed.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 2 LOW)
- CR11-01: create_camera_key year mismatch — duplicate cameras across sources — MEDIUM
- CR11-02: about.html no title in render context — LOW (defensive)
- CR11-03: fetch_source doesn't pass gsmarena kwargs — LOW (already deferred F23)
