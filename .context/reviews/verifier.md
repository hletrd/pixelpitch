# Verifier Review (Cycle 14) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository verification after cycles 1-13 fixes

## Previously Fixed (Cycles 1-13) — Verification Status
All previous fixes verified still working. Gate tests pass cleanly. C13-01 and C13-02 both verified as correctly implemented.

## New Findings

### V14-01: openMVG DSLR cameras produce duplicates in merge — verified by runtime test
**File:** `sources/openmvg.py`, lines 63-69; `pixelpitch.py`, line 340
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by runtime test:
```python
# openMVG: Canon EOS 5D → category='mirrorless'
# Geizhals: Canon EOS 5D → category='dslr'
# create_camera_key('mirrorless') ≠ create_camera_key('dslr')
# → merge preserves BOTH → duplicate on All Cameras page
merged = pp.merge_camera_data([derived_om], [derived_gz])
# Result: 2 records for "Canon EOS 5D"
```

The duplicate is confirmed. This affects every DSLR in the openMVG dataset that also appears in Geizhals data (likely dozens of cameras).

**Fix:** Add DSLR detection heuristic to openMVG's category assignment.

---

### V14-02: UTF-8 BOM in CSV causes 0-row parse — verified by runtime test
**File:** `pixelpitch.py`, lines 250-330
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by runtime test:
```python
# CSV with BOM prefix
content_with_bom = '﻿id,name,category,...\n0,Test Cam,...\n'
parsed = pp.parse_existing_csv(content_with_bom)
# Result: 0 rows parsed (complete failure)
# header[0] = '﻿id' ≠ 'id' → has_id = False → wrong schema → ValueError on each row
```

Without BOM, the same content parses correctly. The fix is to strip BOM at the entry point of `parse_existing_csv`.

---

### V14-03: CineD `FORMAT_TO_MM` unreachable entries — verified by regex test
**File:** `sources/cined.py`, lines 33-47; lines 88-89
**Severity:** LOW | **Confidence:** HIGH

Verified by regex test:
- `"Super35 camera"` → NO MATCH (regex requires space between "Super" and "35")
- `"1 inch sensor"` → NO MATCH (regex matches `1"` or `1-inch`, not `1 inch`)
- `"2/3-inch sensor"` → NO MATCH (regex matches `2/3"`, not `2/3-inch`)

The corresponding `FORMAT_TO_MM` entries (`super35`, `1 inch`, `2/3-inch`) are unreachable dead code.

**Fix:** Extend regex or remove dead entries.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- V14-01: openMVG DSLR duplicates — verified by runtime test — MEDIUM
- V14-02: BOM parse failure — verified by runtime test — MEDIUM
- V14-03: CineD unreachable FORMAT_TO_MM — verified by regex test — LOW
