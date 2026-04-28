# Code Review (Cycle 17) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-16 fixes, focusing on NEW issues missed or introduced by previous fixes

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

All previous fixes remain intact and working. Gate tests pass (98 checks). C16-01 through C16-05 all verified as correctly applied.

## New Findings

### C17-01: Pentax KP, KF, K-r, K-x still misclassified as mirrorless — C16-03 fix was incomplete
**File:** `sources/openmvg.py`, line 47 (`Pentax\s+K[-\s]?\d+[A-Za-z]*`)
**Severity:** MEDIUM | **Confidence:** HIGH

The C16-03 fix changed `Pentax\s+K[-\s]\d` to `Pentax\s+K[-\s]?\d+[A-Za-z]*`. This regex requires at least one digit after K (or K-). It still misses Pentax models that have NO digit between K and the suffix:
- **Pentax KP** (letter directly after K, no digit)
- **Pentax KF** (letter directly after K, no digit)
- **Pentax K-r** (hyphen + letter, no digit)
- **Pentax K-x** (hyphen + letter, no digit)

All four are DSLRs. Verified by testing `_DSLR_NAME_RE.search()` against these names — all return None.

**Concrete failure:** If any of these cameras appear in the openMVG database with sensor width >= 20mm, they are classified as "mirrorless" instead of "dslr".

**Fix:** Change `Pentax\s+K[-\s]?\d+[A-Za-z]*` to `Pentax\s+K[-\s]?[\dA-Za-z]+[A-Za-z]*` to allow letters or digits immediately after K[-\s]?.

---

### C17-02: Nikon Df and Nikon D-suffix models (D3X, D3S, D4S, D5X) missed by DSLR regex
**File:** `sources/openmvg.py`, line 46 (`Nikon\s+D\d{1,4}`)
**Severity:** LOW | **Confidence:** HIGH

The Nikon pattern `Nikon\s+D\d{1,4}` only matches D followed by 1-4 digits. It misses:
- **Nikon Df** — no digits after D (DSLR)
- **Nikon D3X** — letter suffix (the regex `\d{1,4}` matches "3" but the "X" is not captured, which is fine for classification but means the match stops at "D3" not "D3X". Actually, this IS classified correctly since the regex only needs to match the prefix.)
- **Nikon D3S** — same as above, classified correctly

Wait - actually `Nikon\s+D\d{1,4}` would match "Nikon D3" in "Nikon D3X" since the regex doesn't require word boundary at the end. And `\b` at the start ensures it doesn't match inside other words. So D3X, D3S etc. are actually classified correctly.

However, **Nikon Df** is genuinely missed — it has no digits after D.

**Concrete failure:** If "Nikon Df" appears in the openMVG database with sensor width >= 20mm, it is classified as "mirrorless" instead of "dslr". The Df is a well-known retro-style DSLR.

**Fix:** Add `|Nikon\s+Df` to the DSLR regex alternation, or change `Nikon\s+D\d{1,4}` to `Nikon\s+D[\d]{1,4}[A-Za-z]?|Nikon\s+Df`.

---

### C17-03: GSMArena SENSOR_FORMAT_RE doesn't match Unicode curly quotes
**File:** `sources/gsmarena.py`, line 50 (`SENSOR_FORMAT_RE`)
**Severity:** LOW | **Confidence:** MEDIUM

The regex `r'(1/[\d.]+)"'` requires a literal ASCII double-quote character after the fractional format. Some web pages use Unicode right double quotation mark (`″`, U+2033) or other quote variants. The central `TYPE_FRACTIONAL_RE` in `sources/__init__.py` handles this correctly with `(?:\"|inch|-inch|-type|\s*type|″)`.

Verified: `SENSOR_FORMAT_RE.search('1/1.3″')` returns None, while `TYPE_FRACTIONAL_RE.search('1/1.3″')` matches.

**Concrete failure:** A GSMArena page that uses curly quotes for sensor format would fail to extract the sensor type, losing the `type` field and potentially the `size` if it's not in PHONE_TYPE_SIZE.

**Fix:** Change the regex to `r'(1/[\d.]+)(?:\"|″)'` or reuse `TYPE_FRACTIONAL_RE` from `sources/__init__.py`.

---

### C17-04: deduplicate_specs loses year information for color variants
**File:** `pixelpitch.py`, lines 596-597
**Severity:** LOW | **Confidence:** MEDIUM

When color variants are unified (same specs but different names like "Sony A7 IV schwarz" and "Sony A7 IV silber"), the code takes `year = min(years)`. But if the color variants have different years (e.g., black version released 2021, silver version released 2022), the min() is taken. This is arguably correct (earliest release year), but it silently drops the later year without any indication.

Verified: `deduplicate_specs` with two color variants with years 2021 and 2022 produces year=2021.

This is a design decision rather than a bug. The earliest year is arguably the correct one for "first available" semantics. Keeping as informational.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 2 LOW)
- C17-01: Pentax KP/KF/K-r/K-x still misclassified — incomplete C16-03 fix — MEDIUM
- C17-02: Nikon Df missed by DSLR regex — LOW
- C17-03: GSMArena SENSOR_FORMAT_RE doesn't match Unicode quotes — LOW
- C17-04: deduplicate_specs year selection — informational only (not a bug)
