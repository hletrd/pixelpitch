# Tracer Review (Cycle 17) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- T16-01 (crash propagation): Fixed — sensor_size_from_type now catches ZeroDivisionError/ValueError.
- T16-02 (duplicate propagation): Fixed — seen_new_keys set prevents duplicate appends.
- T16-03 (Pentax classification): Partially fixed — K3, K5, K7 etc. now match, but KP, KF, K-r, K-x still fail.

## New Findings

### T17-01: Pentax KP/KF/K-r/K-x misclassification path — C16-03 fix incomplete
**File:** `sources/openmvg.py`, line 47
**Severity:** MEDIUM | **Confidence:** HIGH

Traced misclassification path:
1. openMVG CSV contains "Pentax KP" (letter directly after K)
2. `_DSLR_NAME_RE.search("Pentax KP")` returns None — regex requires at least one digit
3. Category defaults to "mirrorless"
4. On All Cameras page, "Pentax KP" appears under Mirrorless instead of DSLR
5. If Geizhals also lists "Pentax KP" as DSLR, duplicate entries appear on All Cameras page

Same path for KF, K-r, K-x. The C16-03 fix changed `K[-\s]\d` to `K[-\s]?\d+[A-Za-z]*` but the `\d+` requirement still blocks letter-only suffixes.

**Fix:** Change `Pentax\s+K[-\s]?\d+[A-Za-z]*` to `Pentax\s+K[-\s]?[\dA-Za-z]+[A-Za-z]*`.

---

### T17-02: Nikon Df misclassification path
**File:** `sources/openmvg.py`, line 46
**Severity:** LOW | **Confidence:** HIGH

Traced path:
1. openMVG CSV contains "Nikon Df" (letter after D, no digit)
2. `_DSLR_NAME_RE.search("Nikon Df")` returns None — regex requires `\d{1,4}`
3. Category defaults to "mirrorless"
4. Nikon Df is a well-known DSLR but appears as mirrorless

**Fix:** Add `|Nikon\s+Df` to the regex alternation.

---

### T17-03: GSMArena curly-quote sensor format loss path
**File:** `sources/gsmarena.py`, line 50
**Severity:** LOW | **Confidence:** MEDIUM

Traced data loss path:
1. GSMArena page contains `1/1.3″` (Unicode right double quote U+2033)
2. `SENSOR_FORMAT_RE` pattern `r'(1/[\d.]+)"'` requires ASCII `"` — no match
3. `sensor_type` remains None
4. `size` also remains None (not in PHONE_TYPE_SIZE lookup since type is None)
5. Camera appears with "unknown" sensor size on the page

**Fix:** Add Unicode quote variant to SENSOR_FORMAT_RE.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 2 LOW)
- T17-01: Pentax KP/KF/K-r/K-x misclassification — MEDIUM
- T17-02: Nikon Df misclassification — LOW
- T17-03: GSMArena curly-quote data loss — LOW
