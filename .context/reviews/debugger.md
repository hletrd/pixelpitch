# Debugger Review (Cycle 44) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## Previous Findings Status

DBG43-01 (GSMArena spec.size from TYPE_SIZE) — COMPLETED. GSMArena now sets spec.size=None.
DBG43-02 (CineD FORMAT_TO_MM lookup) — COMPLETED. CineD no longer sets spec.size from FORMAT_TO_MM.
DBG43-03 (redundant derived.pitch write) — COMPLETED. Write was restored with improved comments.

## New Findings

### DBG44-03: CineD fmt/fmt_m variables and `if size is None and fmt:` block are dead code after C43-01 fix

**File:** `sources/cined.py, _parse_camera_page function`
**Severity:** LOW | **Confidence:** HIGH

After C43-01 removed `size = FORMAT_TO_MM.get(fmt.lower())`, the fmt variable is computed via regex but never used. The `if size is None and fmt:` block contains only a `pass` statement with a long comment. This is dead code that could confuse maintainers.

**Fix:** Remove the fmt_m regex, fmt assignment, and the `if size is None and fmt:` block. Also remove the FORMAT_TO_MM dict if no test references it.

---


## Summary

- DBG44-03 (LOW): CineD fmt/fmt_m variables and `if size is None and fmt:` block are dead code after C43-01 fix
