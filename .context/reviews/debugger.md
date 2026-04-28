# Debugger Review (Cycle 17) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository latent bug review after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- D16-01 (sensor_size_from_type crash): Fixed — try/except guard confirmed.
- D16-02 (merge dedup): Fixed — seen_new_keys set confirmed.
- D16-03 (Pentax regex): Partially fixed — K3, K5 etc. work, but KP/KF/K-r/K-x still missed.
- D16-04 (http_get OSError): Fixed — OSError in except clause.

## New Findings

### D17-01: Pentax KP, KF, K-r, K-x still misclassified — C16-03 fix incomplete
**File:** `sources/openmvg.py`, line 47
**Severity:** MEDIUM | **Confidence:** HIGH

Same root cause as C17-01/V17-01. The regex `Pentax\s+K[-\s]?\d+[A-Za-z]*` requires at least one digit after K[-\s]?. Pentax KP and KF have a letter directly after K; K-r and K-x have a letter after the hyphen. All are DSLRs.

**Failure mode:** These cameras appear under "Mirrorless" on the website instead of "DSLR". If also present in Geizhals as DSLR, they create duplicate entries.

**Fix:** Change to `Pentax\s+K[-\s]?[\dA-Za-z]+[A-Za-z]*`.

---

### D17-02: Nikon Df — letter-suffix DSLR not matched by regex
**File:** `sources/openmvg.py`, line 46
**Severity:** LOW | **Confidence:** HIGH

The regex `Nikon\s+D\d{1,4}` requires digits after D. Nikon Df has no digit.

**Failure mode:** Nikon Df appears under "Mirrorless" instead of "DSLR" on the website.

**Fix:** Add `|Nikon\s+Df` to the regex alternation.

---

### D17-03: GSMArena SENSOR_FORMAT_RE doesn't match Unicode quotes — silent data loss
**File:** `sources/gsmarena.py`, line 50
**Severity:** LOW | **Confidence:** MEDIUM

The regex `r'(1/[\d.]+)"'` requires ASCII double-quote. Unicode curly quotes (U+2033) are not matched.

**Failure mode:** A GSMArena page using curly quotes for sensor format silently loses the sensor type and size data for that phone. The phone still appears but with "unknown" sensor size.

**Fix:** Add Unicode quote variant to the regex.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 2 LOW)
- D17-01: Pentax KP/KF/K-r/K-x misclassification — MEDIUM
- D17-02: Nikon Df misclassification — LOW
- D17-03: GSMArena Unicode quote data loss — LOW
