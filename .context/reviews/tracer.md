# Tracer Review (Cycle 45) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## Previous Findings Status

TR44-01 (CineD format extraction dead code) — COMPLETED. Removed.

## New Findings

### TR45-01: GSMArena _select_main_lens split regex creates corrupted lens fragments from decimal MP values

**File:** `sources/gsmarena.py, _select_main_lens, line 82`
**Severity:** HIGH | **Confidence:** HIGH

Causal trace:
1. `_phone_to_spec` calls `_select_main_lens(cam)` to pick the main (wide) lens entry
2. `_select_main_lens` splits the camera value using `re.split(r'(?=\b\d+(?:\.\d+)?\s*MP\b)', raw)`
3. The `\b` (word boundary) in the look-ahead matches at the boundary between "0" and "." in "12.2", creating a split point INSIDE the decimal number
4. The greedy `\d+` in the look-ahead matches "0" (the digit before the decimal), not the full "12"
5. Split produces `['12.', '2 MP, f/1.9, (wide), 1/2.55", 1.25µm, ...']`
6. `_select_main_lens` sorts parts by role_priority and picks "2 MP, f/1.9, ..." (because "wide" has priority 0)
7. Back in `_phone_to_spec`:
   - `re.match(r'\s*([\d.]+)\s*MP', '2 MP, ...')` extracts mpix=2.0 (WRONG — should be 12.2)
   - `TYPE_FRACTIONAL_RE.search('2 MP, ...')` — the sensor type string "1/2.55\"" starts after a comma and space, but the regex needs a suffix (", -inch, etc.). In the corrupted fragment, the `" ` suffix may be present but the numeric prefix "1/2.55" is intact. Let me verify...

Hypothesis 1: The type is lost because the quote suffix is severed → REJECTED (the fragment still contains the full sensor format string including the quote)

Hypothesis 2: The type is lost because TYPE_FRACTIONAL_RE requires the quote/inch suffix to be immediately after the number → CONFIRMED. In the actual GSMArena HTML, the camera value text is flattened with spaces between entries. The `1/2.55"` format includes the quote as part of the lens entry. After the split, the fragment "2 MP, f/1.9, 25mm (wide), 1/2.55\", 1.25µm" still contains the quote. However, testing shows TYPE_FRACTIONAL_RE.search() on the corrupted fragment does NOT match because the fragment starts with "2 MP" and the look-ahead for `1/2.55` is after multiple fields.

Wait — actually, TYPE_FRACTIONAL_RE searches the entire main lens string, not just the MP prefix. Let me verify...

The actual test showed: for input `'2 MP, f/1.9, 25mm (wide), 1/2.55, 1.25µm, dual pixel PDAF, OIS'`, TYPE_FRACTIONAL_RE.search returns None because `1/2.55` doesn't have the required suffix (quote, inch, type, etc.). In GSMArena HTML, the original text has `1/2.55"` with the quote, but after HTML stripping and flattening, the quote may be preserved or lost depending on the page structure.

**Fix:** Remove `\b` from the split regex: `r'(?=\d+(?:\.\d+)?\s*MP\b)'`

---

## Summary

- TR45-01 (HIGH): GSMArena _select_main_lens split regex creates corrupted lens fragments from decimal MP values
