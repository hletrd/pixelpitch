# Document Specialist Review (Cycle 24) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-23 fixes

## Previous Findings Status

All previously identified doc/code mismatches addressed. No regressions.

## New Findings

### DOC24-01: TYPE_FRACTIONAL_RE comment says "ASCII/Unicode quotes" but also matches -inch and -type suffixes

**File:** `pixelpitch.py`, lines 47-49
**Severity:** LOW | **Confidence:** MEDIUM

The comment on lines 47-49 says:
```
# Canonical fractional-inch sensor type regex — matches "1/x.y" followed by
# any recognized suffix (ASCII/Unicode quotes, "inch", "-type", etc.).
```

While the comment does mention "inch" and "-type", it labels them as "etc." and the primary focus is on quotes. The regex also matches `\s*type` (space+type) which is not mentioned. This is a minor documentation precision issue.

**Fix:** Update comment to explicitly list all suffix alternatives.

---

## Summary

- DOC24-01 (LOW): TYPE_FRACTIONAL_RE comment could more precisely list all matched suffixes
