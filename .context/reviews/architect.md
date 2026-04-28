# Architect Review (Cycle 24) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-23 fixes

## Previous Findings Status

All previously identified architectural concerns remain deferred (LOW severity). C22-05 (ad-hoc field preservation) and F32 (monolith) are tracked with exit criteria.

## New Findings

### ARCH24-01: TYPE_FRACTIONAL_RE is a shared regex with evolving requirements

**File:** `sources/__init__.py`, line 68
**Severity:** LOW | **Confidence:** MEDIUM

The `TYPE_FRACTIONAL_RE` regex is imported by both `pixelpitch.py` (line 50) and `gsmarena.py` (line 24). It was designed for fractional-inch formats (`1/x.y"`) but is now also used for bare 1-inch format detection (which it doesn't support). As new source formats are encountered, this single regex accumulates more suffix alternatives, making it harder to reason about.

**Current pattern:** `(1/[\d.]+)(?:\"|inch|-inch|-type|\s*type|″)`
**Gap:** Missing `\s*inch` (space before "inch") and bare `1"` (1-inch sensor) support.

**Fix:** Either extend the regex with `\s*inch` alternative, or split into two patterns: one for fractional-inch (1/x.y") and one for bare sensor formats (1"). The latter is a more maintainable approach if more formats are expected.

---

## Summary

- ARCH24-01 (LOW): TYPE_FRACTIONAL_RE regex accumulating alternatives — consider splitting
