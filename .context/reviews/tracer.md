# Tracer Review (Cycle 14) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-13 fixes

## Traced Flows

### Flow 1: openMVG Canon EOS 5D through merge → duplicate on All Cameras page
**Path:**
1. `openmvg.fetch()` → sensor width 36.0mm → `size[0] >= 20` → `category="mirrorless"` (line 64)
2. `Geizhals extract_specs` → `category="dslr"` (from CATEGORIES lookup)
3. `merge_camera_data` called with new=[openMVG Canon 5D (mirrorless)] + existing=[Geizhals Canon 5D (dslr)]
4. `create_camera_key(openMVG)` → `"canon eos 5d-mirrorless"` ≠ `create_camera_key(Geizhals)` → `"canon eos 5d-dslr"`
5. Keys don't match → both records preserved in `merged_specs`
6. `render_html` → `specs_all = sorted_by(merged_specs, "pitch")` → contains 2 entries for Canon EOS 5D
7. "All Cameras" page renders with both entries → visible duplicate

**FINDING: T14-01** — openMVG DSLR misclassification causes duplicate entries on All Cameras page.
**Severity:** MEDIUM | **Confidence:** HIGH

Competing hypotheses:
- H1: Fix openMVG's category heuristic (add DSLR name detection) — addresses root cause, pragmatic
- H2: Change `create_camera_key` to exclude category — too broad, would merge cameras that genuinely have different categories (e.g., a compact and mirrorless with same name)
- H3: Normalize categories during merge based on existing data — adds complexity, but most robust

Recommended: H1 (fix openMVG heuristic) as the primary fix, with H3 as a potential future improvement.

---

### Flow 2: BOM-prefixed CSV through parse_existing_csv → 0 rows
**Path:**
1. Developer opens `camera-data.csv` in Excel, saves as "CSV UTF-8"
2. Excel adds UTF-8 BOM (`\xef\xbb\xbf`) at start of file
3. `load_csv` → `path.read_text(encoding="utf-8")` → BOM character preserved as `﻿`
4. `parse_existing_csv(content)` → `csv.reader` → `header[0] = "﻿id"` (with BOM)
5. `has_id = header[0] == "id"` → `False` (BOM character present)
6. Parser uses no-id schema → column misalignment → `int("5.12")` → ValueError
7. Exception caught → row skipped → ALL rows skipped → 0 records parsed
8. `render_html` → `previous_csv` has content but `parse_existing_csv` returns [] → `existing_specs = []`
9. Build proceeds with no existing data → site regenerates from scratch (losing preserved cameras)

**FINDING: T14-02** — UTF-8 BOM in CSV causes complete parse failure and loss of preserved camera data.
**Severity:** MEDIUM | **Confidence:** HIGH

---

### Flow 3: CineD "Super35" format string through _parse_camera_page → missed lookup
**Path:**
1. CineD camera page body contains "Super35" (no space)
2. `fmt_re.search(body_text)` → regex `Super 35(?:\s*mm)?` requires space → NO MATCH
3. `fmt = ""` → `FORMAT_TO_MM.get("")` → `None`
4. `size = None` (if no explicit mm dimensions found)
5. Camera record has no sensor size → may be skipped by `if not (size or mpix): return None`
6. Camera lost from CineD data

**FINDING: T14-03** — CineD "Super35" format (no space) not captured by regex, camera may be lost.
**Severity:** LOW | **Confidence:** MEDIUM

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- T14-01: openMVG DSLR duplicate flow — MEDIUM
- T14-02: BOM parse failure flow — MEDIUM
- T14-03: CineD "Super35" missed lookup flow — LOW
