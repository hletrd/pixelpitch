# Debugger Review (Cycle 45) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## Previous Findings Status

DBG44-03 (CineD fmt/fmt_m dead code) — COMPLETED. Removed.

## New Findings

### DBG45-01: GSMArena _select_main_lens regex split corrupts decimal MP — latent data corruption bug

**File:** `sources/gsmarena.py, _select_main_lens, line 82`
**Severity:** HIGH | **Confidence:** HIGH

**Failure mode:** When a phone's camera spec page lists a main camera with a decimal MP value (e.g., "12.2 MP"), the `re.split(r'(?=\b\d+(?:\.\d+)?\s*MP\b)', raw)` regex incorrectly splits the value at the decimal point. The word boundary `\b` fires between the integer digit "2" and the decimal point ".", causing the split to produce two fragments: "12." and "2 MP, f/1.7, ...". The function then selects "2 MP" as the main lens, extracting mpix=2.0 instead of 12.2.

**Concrete failure scenario:**
1. GSMArena fetch runs for Google Pixel 7 (12.2 MP main camera)
2. _select_main_lens receives "12.2 MP, f/1.9, 25mm (wide), 1/2.55\", 1.25µm, ..."
3. Split produces: ["12.", "2 MP, f/1.9, 25mm (wide), 1/2.55\", 1.25µm, ..."]
4. _select_main_lens sorts and picks "2 MP, f/1.9, ..." (wide has priority 0)
5. _phone_to_spec extracts mpix=2.0 (wrong), type=None (lost), pitch=1.25 (correct by coincidence)
6. derive_spec cannot compute sensor size because spec.type=None and spec.size=None
7. The camera appears in the database with mpix=2.0, unknown sensor size, and unknown derived.size

**Why it was missed for 44 cycles:** The existing test fixture (Galaxy S25 Ultra) only has integer MP values (200, 10, 50, 50). No test exercises decimal MP, and the bug only manifests at runtime when scraping phones with decimal MP main cameras.

**Fix:** Remove `\b` from the start of the split regex: `r'(?=\d+(?:\.\d+)?\s*MP\b)'`

---

## Summary

- DBG45-01 (HIGH): GSMArena _select_main_lens regex split corrupts decimal MP — latent data corruption bug
