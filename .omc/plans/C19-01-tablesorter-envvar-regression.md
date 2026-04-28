# Plan: Cycle 19 Findings — Tablesorter Column Regression + Env Var Error Handling

**Created:** 2026-04-28
**Status:** IN PROGRESS
**Source Reviews:** C19-01, C19-02 (aggregate)

---

## Task 1: Fix tablesorter column indices for non-"all" pages — C19-01
**Finding:** C19-01 (6-agent consensus: code-reviewer, debugger, designer, critic, tracer, verifier)
**Severity:** MEDIUM | **File:** `templates/pixelpitch.html`, lines 228-258
**Root Cause:** C18-08 fix added `sensor-width` parser to column index 2 unconditionally. On non-"all" pages (8 of 9 pages), the Category column is absent, so Sensor Size is at index 1, not 2.

### What needs to be done
Replace the hardcoded column indices with conditional Jinja2 blocks based on `{% if page == "all" %}`:

**For `#table_with_pitch`:**
- "all" page: Name(0), Category(1), Sensor Size(2), Resolution(3), Pixel Pitch(4), Year(5)
- non-"all" page: Name(0), Sensor Size(1), Resolution(2), Pixel Pitch(3), Year(4)

**For `#table_without_pitch`:**
- "all" page: Name(0), Category(1), Sensor Size(2), Resolution(3), Year(4)
- non-"all" page: Name(0), Sensor Size(1), Resolution(2), Year(3)

### Verification
- Gate tests pass
- Render DSLR page HTML and verify `sensor-width` parser is on column 1
- Render "All Cameras" page HTML and verify `sensor-width` parser is on column 2

---

## Task 2: Add error handling for GSMARENA_MAX_PAGES_PER_BRAND env var — C19-02
**Finding:** C19-02 (4-agent consensus: code-reviewer, critic, verifier, debugger)
**Severity:** LOW | **File:** `pixelpitch.py`, line 1046
**Root Cause:** C18-04 fix wired the env var but didn't add try/except for `int()` conversion.

### What needs to be done
Wrap `int(os.environ.get("GSMARENA_MAX_PAGES_PER_BRAND", "2"))` in try/except with fallback to default value 2.

### Verification
- Gate tests pass
- Verify empty string env var falls back to 2

---

## Deferred Findings

### TE19-01: No JS-side test for tablesorter column config — NEGLIGIBLE
**File:** `templates/pixelpitch.html`
**Original Severity:** NEGLIGIBLE | **Confidence:** MEDIUM
**Reason for deferral:** The tablesorter configuration is JavaScript that runs in the browser. The offline Python test suite cannot exercise it. Adding browser automation tests (Playwright, Selenium) is a significant infrastructure addition beyond the scope of a bug fix.
**Re-open if:** Browser automation tests are added to the project.
