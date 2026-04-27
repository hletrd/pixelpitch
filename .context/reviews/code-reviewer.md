# Code Review (Cycle 14) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-13 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-13) — Confirmed Resolved
All previous fixes remain intact. C13-01 (load_csv UnicodeDecodeError) and C13-02 (Sony fallback normalizations) both verified as correctly applied and tested.

## New Findings

### C14-01: openMVG classifies all interchangeable-lens cameras as "mirrorless" — causes visible duplicates on All Cameras page
**File:** `sources/openmvg.py`, lines 63-69
**Severity:** MEDIUM | **Confidence:** HIGH

openMVG's category heuristic uses sensor width only:
```python
if size[0] >= 20:
    category = "mirrorless"
else:
    category = "fixed"
```

Any DSLR with sensor width >= 20mm (i.e., all APS-C and full-frame DSLRs) gets `category="mirrorless"`. When the same camera also exists in Geizhals data with `category="dslr"`, `create_camera_key` produces different keys (`canon eos 5d-mirrorless` vs `canon eos 5d-dslr`), so `merge_camera_data` treats them as different cameras and preserves BOTH. This results in visible duplicates on the "All Cameras" page.

**Concrete failure:** Canon EOS 5D appears twice: once under "mirrorless" (from openMVG) and once under "dslr" (from Geizhals). The same happens for every DSLR that openMVG covers (Canon 5D Mark II/III/IV, Nikon D850, etc.).

**Fix:** In `openmvg.fetch`, add a simple name-based heuristic for DSLR detection (e.g., if the name contains "EOS" and "D" but not "R", or contains "D8" / "D7" / "D6" / "D5" / "D4" / "D3" / "D2" / "D1" followed by digit, classify as "dslr"). Alternatively, normalize openMVG categories during merge by matching against Geizhals camera names that have different categories.

---

### C14-02: UTF-8 BOM in camera-data.csv causes complete parse failure
**File:** `pixelpitch.py`, lines 250-330 (`parse_existing_csv`); lines 238-247 (`load_csv`)
**Severity:** MEDIUM | **Confidence:** HIGH

If `camera-data.csv` (or any source CSV) is saved with a UTF-8 BOM (byte order mark, `﻿` at start), `parse_existing_csv` fails completely:

1. `read_text(encoding="utf-8")` preserves the BOM character
2. `header[0]` becomes `"﻿id"` instead of `"id"`
3. `has_id = header[0] == "id"` evaluates to `False`
4. The parser uses the wrong schema (no-id instead of has-id)
5. Column alignment is wrong, causing `ValueError` on each row
6. Result: 0 rows parsed, entire render pipeline produces an empty site

A common scenario: opening `camera-data.csv` in Excel and saving as "CSV UTF-8" adds a BOM.

**Fix:** Strip BOM from the content at the start of `parse_existing_csv`:
```python
if csv_content and csv_content[0] == '﻿':
    csv_content = csv_content[1:]
```
Or use `encoding="utf-8-sig"` in `read_text` calls (which strips BOM automatically).

---

### C14-03: CineD `FORMAT_TO_MM` has unreachable entries — dead code
**File:** `sources/cined.py`, lines 33-47; lines 88-89 (regex)
**Severity:** LOW | **Confidence:** HIGH

Three entries in `FORMAT_TO_MM` are unreachable because the regex in `_parse_camera_page` never captures their string patterns:

| Entry in table | Regex pattern that should match | Why it's unreachable |
|---|---|---|
| `"super35"` | Would need `Super35` (no space) | Regex requires `Super 35` (with space) |
| `"1 inch"` | Would need `1 inch` (no hyphen, no quote) | Regex matches `1"` or `1-inch` only |
| `"2/3-inch"` | Would need `2/3-inch` | Regex matches `2/3"` only |

These entries were presumably added for defense-in-depth but are never exercised. The `super35` entry is particularly notable — a CineD page containing "Super35" (no space) would not be matched by the regex, and the fallback lookup would fail.

**Fix:** Either add the missing regex alternatives to `fmt_re` (e.g., `Super[- ]?35(?:\s*mm)?`, `1[- ]inch`), or remove the unreachable entries from the table to reduce confusion.

---

### C14-04: `PHONE_TYPE_SIZE` is a mutable alias of `TYPE_SIZE` — accidental mutation risk
**File:** `sources/gsmarena.py`, line 58
**Severity:** LOW | **Confidence:** HIGH

```python
PHONE_TYPE_SIZE: dict[str, tuple[float, float]] = SENSOR_TYPE_SIZE
```

This is a direct reference to the same dict object, not a copy. Any mutation to `PHONE_TYPE_SIZE` (e.g., `PHONE_TYPE_SIZE["custom"] = (1.0, 1.0)`) would corrupt the central `TYPE_SIZE` table in `pixelpitch.py`. The code comment warns against mutation, but the type system doesn't enforce it.

**Fix:** Either use `types.MappingProxyType(SENSOR_TYPE_SIZE)` for read-only access, or copy the dict: `PHONE_TYPE_SIZE = dict(SENSOR_TYPE_SIZE)`.

---

## Summary
- NEW findings: 4 (2 MEDIUM, 2 LOW)
- C14-01: openMVG DSLR misclassification causes duplicates — MEDIUM
- C14-02: UTF-8 BOM in CSV causes complete parse failure — MEDIUM
- C14-03: CineD unreachable FORMAT_TO_MM entries — LOW
- C14-04: PHONE_TYPE_SIZE mutable alias — LOW
- All cycle 1-13 fixes remain intact
