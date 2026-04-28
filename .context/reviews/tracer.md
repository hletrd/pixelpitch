# Tracer Review (Cycle 44) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## Previous Findings Status

All C43 findings resolved. No regressions.

## New Findings

### TR44-01: CineD format extraction regex runs but result is discarded — dead code path after C43-01

**File:** `sources/cined.py, _parse_camera_page`
**Severity:** LOW | **Confidence:** HIGH

Causal trace: fmt_m regex at line 92-97 matches format class in HTML → fmt assigned → `if size is None and fmt:` block at lines 106-119 contains only pass → fmt is never used to set spec.type or spec.size. The entire format detection path is dead code.

**Fix:** Remove the fmt_m regex, fmt assignment, and the `if size is None and fmt:` block.

---


## Summary

- TR44-01 (LOW): CineD format extraction regex runs but result is discarded — dead code path after C43-01
