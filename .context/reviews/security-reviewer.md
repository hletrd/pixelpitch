# Security Review (Cycle 57)

**Reviewer:** security-reviewer
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## Inventory

- `pixelpitch.py`, `sources/*`, `templates/*`, `.github/workflows/*`.
- Secrets exposure: scanned `*.py`, `*.html`, `*.json`, `.github/`.

All examined.

## OWASP-style sweep

### A01 Broken Access Control — N/A (static site)
### A02 Cryptographic Failures — N/A (no crypto)
### A03 Injection
- Templates use Jinja2 with autoescape=True
  (`pixelpitch.py:_get_jinja_env`). Verified — no `|safe` on user
  data found.
- CSV emission uses the stdlib `csv.writer` — proper quoting.
- No subprocess shell=True usages.

### A04 Insecure Design — no notable change
### A05 Security Misconfiguration
- F57-SR-old (carry-over of C10-08): debug Chrome opens
  port 9222 on macOS host; development-only. Re-defer.

### A06 Vulnerable Components
- requirements.txt: `jinja2`, `requests`, `DrissionPage`,
  `beautifulsoup4`. All commonly used; no pin specified —
  resolved at install time.

### A07 ID & Auth — N/A
### A08 SSRF / Software Integrity
- HTTP client (`sources/__init__.py:http_get`) follows redirects
  by default. URLs are hardcoded. No user-controlled URL fetch.
  OK.

### A09 Logging Failures — using `print()`; F21 deferred.
### A10 SSRF — covered by A08.

## New findings (cycle 57)

### F57-SR-01: `Path.read_text` on per-source CSVs accepts any UTF-8 — LOW (informational)

- **File:** `pixelpitch.py:1067`
- **Detail:** A maliciously crafted source CSV could embed control
  characters or extremely long lines that pass `parse_existing_csv`
  but produce surprising HTML. Mitigated because:
  - Jinja2 autoescape=True stops HTML injection.
  - The CSV files are produced by our own scrapers, committed to
    the repo, and reviewed before deployment.
  - `csv.reader` handles malformed input gracefully.
- **Severity:** LOW. **Confidence:** LOW (theoretical).
- **Disposition:** No action; trust boundary is the repo itself.

## Confirmed-still-good

- No hardcoded secrets in source.
- `.github/workflows/github-pages.yml` uses GITHUB_TOKEN via
  contexts only.
- robots.txt and sitemap.xml hand-curated; no template injection.

## Confidence summary

- 0 actionable findings.
- 1 informational (F57-SR-01: trust boundary already enforced
  by repo review).
