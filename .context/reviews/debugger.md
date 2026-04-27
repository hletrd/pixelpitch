# Debugger Review (Cycle 16) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository latent bug review after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
All previous fixes remain intact. No regressions.

## New Findings

### D16-01: `sensor_size_from_type` — unhandled ZeroDivisionError/ValueError on malformed input
**File:** `pixelpitch.py`, lines 152-165
**Severity:** MEDIUM | **Confidence:** HIGH

Same as C16-01. The function crashes on `1/0`, `1/0.0`, `1/` sensor type values. This is a latent bug that has never triggered in production because source data has never contained these values, but it represents a crash vector.

**Failure mode:** CI build fails silently — no HTML output, no error reporting to the user.

---

### D16-02: `merge_camera_data` — duplicate entries when same camera appears in multiple sources with same category
**File:** `pixelpitch.py`, lines 349-407
**Severity:** MEDIUM | **Confidence:** HIGH

Same as C16-02. This is a regression risk from the C15-01 fix: now that openMVG correctly classifies Canon EOS xxxD cameras as DSLR, they overlap with Geizhals DSLR data and produce duplicate entries.

**Failure mode:** User sees the same camera twice on the All Cameras page.

---

### D16-03: Pentax DSLR regex misses multiple model families
**File:** `sources/openmvg.py`, line 47
**Severity:** LOW | **Confidence:** HIGH

Same as C16-03. 10+ Pentax DSLR models are missed by the regex.

**Failure mode:** Pentax K3, K5, K7, KP, KF etc. appear under Mirrorless instead of DSLR.

---

### D16-04: `http_get` does not catch OSError subclasses (ConnectionResetError, SSLError)
**File:** `sources/__init__.py`, lines 48-61
**Severity:** LOW | **Confidence:** MEDIUM

Same as S16-02. While urllib typically wraps these as URLError, edge cases exist where the underlying socket error leaks through.

**Failure mode:** Source fetch step crashes with unhandled exception, causing `continue-on-error: true` to skip the source, but the error message is confusing.

---

## Summary
- NEW findings: 4 (2 MEDIUM, 2 LOW)
- D16-01: sensor_size_from_type crash — MEDIUM
- D16-02: merge_camera_data duplicate — MEDIUM
- D16-03: Pentax regex misses — LOW
- D16-04: http_get exception gap — LOW
