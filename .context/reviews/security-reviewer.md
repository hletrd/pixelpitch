# Security Review (Cycle 11) — OWASP Top 10, Secrets, Unsafe Patterns

**Reviewer:** security-reviewer
**Date:** 2026-04-28
**Scope:** Full repository security re-review after cycles 1-10 fixes

## Previously Fixed (Cycles 1-10) — Confirmed Resolved
- All SRI hashes present on all 7 CDN resources
- `data-name` attribute `|e` filter — FIXED
- All `target="_blank"` links have `rel="noopener noreferrer"` — FIXED

## Deferred Items Still Valid
- C10-07: HTTP redirect chain SSRF risk — DEFERRED
- C10-08: Remote debugging port — DEFERRED

## New Findings

### S11-01: `_parse_fields` in imaging_resource.py strips HTML tags unsafely — regex XSS risk
**File:** `sources/imaging_resource.py`, lines 90-99
**Severity:** LOW | **Confidence:** MEDIUM

`_parse_fields` strips HTML from values using `re.sub(r"<[^>]+>", " ", m.group(2))`. While this data is only written to CSV (not rendered as HTML directly), the stripped text could contain unescaped `<` or `>` characters from malformed HTML that don't match the regex, potentially carrying through to the CSV. The CSV writer properly quotes fields, so there's no injection risk in the CSV itself. The Jinja2 templates use `{{ spec.spec.name }}` without the `|e` filter in some locations... wait, let me check.

Actually, Jinja2 autoescape is enabled (`select_autoescape(["html", "xml"])`), so all `{{ }}` expressions are auto-escaped. The `data-name` attribute explicitly uses `|e` for extra safety in attribute contexts. No XSS risk.

Retracting this finding — Jinja2 autoescape protects against this.

---

## Summary
- NEW findings: 0 (1 investigated and retracted)
- No security regressions
- Deferred items remain appropriate
