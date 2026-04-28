# Architect Review (Cycle 17) — Architectural/Design Risks, Coupling, Layering

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository architecture re-review after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- A16-01 (merge input dedup contract): Fixed — `seen_new_keys` set and updated docstring.
- A16-02 (digicamdb registry DRY): Fixed — removed from SOURCE_REGISTRY.
- A16-03 (sensor_size_from_type validation): Fixed — try/except with proper docstring.

## New Findings

### A17-01: DSLR classification heuristic in openMVG is fragile — regex-based approach has ongoing maintenance burden
**File:** `sources/openmvg.py`, lines 42-53
**Severity:** LOW | **Confidence:** HIGH

The openMVG source has no body-type field, so DSLR classification relies on a regex heuristic (`_DSLR_NAME_RE`). Each new camera naming pattern requires a regex update. The C16-03 fix partially addressed Pentax models, but KP, KF, K-r, K-x are still missed (C17-01). Nikon Df is also missed (C17-02). This pattern of incremental regex fixes will continue as new camera models are released.

From an architectural standpoint, the regex approach is inherently fragile. A more robust approach would be:
1. Maintain a curated DSLR name list alongside the regex
2. Or use the openMVG dataset's own metadata if it adds a body-type field in the future
3. Or accept the heuristic limitation and classify edge cases as "mirrorless" (which is less wrong than the alternative)

However, for the current data scale and update frequency, the regex approach is pragmatic. The ongoing maintenance cost is low (a few model names per year).

**Fix (if desired):** No architectural change recommended. Continue with regex fixes for known gaps (C17-01, C17-02).

---

## Summary
- NEW findings: 1 (LOW)
- A17-01: DSLR regex heuristic has ongoing maintenance cost — LOW (informational)
- No architectural regressions
