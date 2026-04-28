# Critic Review (Cycle 44) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## Previous Findings Status

CRIT43-01 (GSMArena/CineD spec.size provenance) — COMPLETED. Both sources now leave spec.size=None.
CRIT43-02 (redundant derived.pitch write) — COMPLETED. Write was restored with improved comments.

## New Findings

### CRIT44-01: FORMAT_TO_MM dict in cined.py is dead code after C43-01 fix — no code references it

**File:** `sources/cined.py, lines 37-51`
**Severity:** LOW | **Confidence:** HIGH

After C43-01 removed `size = FORMAT_TO_MM.get(fmt.lower())`, the FORMAT_TO_MM dict is defined but never referenced by any executable code. It's only mentioned in comments and the docstring. The module docstring says 'The FORMAT_TO_MM table is kept for the regex coverage test only' but no test actually references it. This is dead code that could confuse maintainers into thinking it's still used.

**Fix:** Remove the FORMAT_TO_MM dict. If a regex coverage test is desired, add one that explicitly references it. Otherwise the format extraction regex itself is sufficient documentation.

---


## Summary

- CRIT44-01 (LOW): FORMAT_TO_MM dict in cined.py is dead code after C43-01 fix — no code references it
