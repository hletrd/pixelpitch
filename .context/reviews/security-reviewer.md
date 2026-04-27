# Security Review (Cycle 16) — OWASP Top 10, Secrets, Unsafe Patterns

**Reviewer:** security-reviewer
**Date:** 2026-04-28
**Scope:** Full repository security re-review after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
- All SRI hashes present on all 7 CDN resources
- `data-name` attribute `|e` filter — FIXED
- All `target="_blank"` links have `rel="noopener noreferrer"` — FIXED
- `UnicodeDecodeError` now caught in both `load_csv` and `_load_per_source_csvs` — FIXED
- BOM defense in both `parse_existing_csv` and `openmvg.fetch()` — FIXED
- `importlib.import_module` uses only hardcoded `SOURCE_REGISTRY` values — NOT a vulnerability (input is validated)

## Deferred Items Still Valid
- C10-07: HTTP redirect chain SSRF risk — DEFERRED
- C10-08: Remote debugging port — DEFERRED
- F34: `importlib.import_module` with user-controllable input — DEFERRED

## New Findings

### S16-01: `sensor_size_from_type` unhandled exception on malicious/malformed sensor type — DoS vector
**File:** `pixelpitch.py`, lines 152-165
**Severity:** MEDIUM | **Confidence:** HIGH

Same issue as C16-01 from code-reviewer perspective. A crafted sensor type value of `"1/0"` or `"1/0.0"` in source HTML would cause an unhandled `ZeroDivisionError` that crashes the entire render pipeline. From a security perspective, this is a denial-of-service vector: if an attacker could control the HTML returned by a source (e.g., compromised GSMArena page, or MITM on unencrypted HTTP), they could crash the CI build.

The `http_get` function uses `urllib.request.urlopen` without verifying SSL certificates beyond the default. A network-level attacker could inject a `1/0"` sensor type into the response.

**Fix:** Add try/except in `sensor_size_from_type` for `ZeroDivisionError` and `ValueError`.

---

### S16-02: `http_get` does not catch `ConnectionResetError` or `ssl.SSLError` — unhandled exception on network errors
**File:** `sources/__init__.py`, lines 48-61
**Severity:** LOW | **Confidence:** MEDIUM

The `http_get` function catches `URLError`, `HTTPError`, and `TimeoutError`. However, `ConnectionResetError` and `ssl.SSLError` are subclasses of `OSError`, not `URLError`. While `urllib.request.urlopen` typically wraps these as `URLError`, there are edge cases where the underlying socket error leaks through, particularly on SSL handshake failures. An uncaught exception would crash the source fetch step.

**Concrete failure:** If the remote server resets the connection during an SSL handshake, the fetch could crash instead of retrying and returning None gracefully.

**Fix:** Add `OSError` to the except clause in `http_get` (catches both `ConnectionResetError` and `ssl.SSLError`).

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- S16-01: sensor_size_from_type crash is a DoS vector — MEDIUM
- S16-02: http_get misses some OSError subclasses — LOW
- No security regressions
- Deferred items remain appropriate
