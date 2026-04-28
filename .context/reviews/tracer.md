# Tracer Review (Cycle 22) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## T22-01: Year-change log flow traced — `elif` misattachment from C21-01

**Trace:**
1. Pre-C21-01 code structure (after C20-03):
   ```
   if spec.year is None: preserve year       (line ~418)
   elif years differ:      print year change  (line ~419)
   if spec.size is None:  preserve size       (new in C20-03)
   if spec.pitch is None: preserve pitch      (new in C20-03)
   ```
2. C21-01 fix inserted SpecDerived preservation BETWEEN spec.year and the year elif:
   ```
   if spec.year is None: preserve year               (line 417)
   if spec.size is None:  preserve spec.size          (line 411)
   if spec.pitch is None: preserve spec.pitch         (line 413)
   if spec.mpix is None:  preserve spec.mpix          (line 415)
   if derived.size is None:  preserve derived.size    (line 424)
   if derived.area is None:  preserve derived.area    (line 426)
   if derived.pitch is None: preserve derived.pitch   (line 428)
   elif years differ:      print year change          (line 429) ← WRONG ATTACHMENT
   ```
3. The `elif` is now syntactically part of the `if derived.pitch is None` block
4. Year change log only fires when `derived.pitch is NOT None` (elif condition)

**Root cause:** The C21-01 fix inserted code above the year-change `elif` without recognizing it was part of a conditional chain. The insertion changed the `elif`'s parent from the year-preservation `if` to the pitch-preservation `if`.

**Fix point:** Convert the `elif` to a standalone `if` after all preservation logic.

---

## Summary

- T22-01: Year-change `elif` misattachment traced to C21-01 insertion — fix: convert to standalone `if`
