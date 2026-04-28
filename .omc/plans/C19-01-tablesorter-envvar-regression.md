# Plan: Cycle 19 Findings — Tablesorter Column Regression + Env Var Error Handling

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** C19-01, C19-02 (aggregate)

---

## Task 1: Fix tablesorter column indices for non-"all" pages — C19-01 — DONE
**Finding:** C19-01 (6-agent consensus: code-reviewer, debugger, designer, critic, tracer, verifier)
**Severity:** MEDIUM | **File:** `templates/pixelpitch.html`, lines 228-258
**Commit:** 3529498
**Root Cause:** C18-08 fix added `sensor-width` parser to column index 2 unconditionally. On non-"all" pages (8 of 9 pages), the Category column is absent, so Sensor Size is at index 1, not 2.

### What was done
Replaced hardcoded column indices with conditional Jinja2 blocks based on `{% if page == "all" %}`:
- "all" page: Sensor Size at column 2 (after Category at column 1)
- non-"all" page: Sensor Size at column 1 (no Category column)

Both `#table_with_pitch` and `#table_without_pitch` configs updated.

### Verification
- Gate tests pass (105 checks)
- Rendered DSLR page HTML: `sensor-width` parser correctly on column 1
- Rendered All Cameras page HTML: `sensor-width` parser correctly on column 2

---

## Task 2: Add error handling for GSMARENA_MAX_PAGES_PER_BRAND env var — C19-02 — DONE
**Finding:** C19-02 (4-agent consensus: code-reviewer, critic, verifier, debugger)
**Severity:** LOW | **File:** `pixelpitch.py`, line 1046
**Commit:** 29535ee
**Root Cause:** C18-04 fix wired the env var but didn't add try/except for `int()` conversion.

### What was done
Wrapped `int(os.environ.get("GSMARENA_MAX_PAGES_PER_BRAND", "2"))` in try/except with fallback to default value 2.

### Verification
- Gate tests pass (105 checks)

---

## Deferred Findings

### TE19-01: No JS-side test for tablesorter column config — NEGLIGIBLE
**File:** `templates/pixelpitch.html`
**Original Severity:** NEGLIGIBLE | **Confidence:** MEDIUM
**Reason for deferral:** The tablesorter configuration is JavaScript that runs in the browser. The offline Python test suite cannot exercise it. Adding browser automation tests (Playwright, Selenium) is a significant infrastructure addition beyond the scope of a bug fix.
**Re-open if:** Browser automation tests are added to the project.
