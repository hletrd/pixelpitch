# Critic Review (Cycle 22) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## C22-CR01: Year-change `elif` misattachment — C21-01 regression

**Severity:** MEDIUM | **Confidence:** HIGH

The C21-01 fix inserted SpecDerived field preservation code between the Spec year-preservation `if` and the year-change `elif`. This broke the conditional chain: the `elif` is now attached to the SpecDerived pitch preservation `if` instead of the year preservation `if`.

This is a classic code-insertion bug: adding code in the middle of a conditional chain without recognizing that the `elif` is part of that chain. The fix should have either:
1. Converted the year-change `elif` to a standalone `if` before inserting new code, or
2. Inserted the SpecDerived preservation AFTER the entire year-preservation block

**Amplification:** While this only affects a diagnostic log (not data correctness), it demonstrates that the merge function is becoming fragile. Multiple rounds of fixes (C20-03, C21-01) have added preservation logic in an ad-hoc manner. A cleaner design would be a generic "preserve None fields" helper that iterates over all field names.

---

## Summary

- C22-CR01 (MEDIUM): `elif` misattachment is a code-insertion regression — suggests merge logic needs refactoring
