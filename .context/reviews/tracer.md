# Tracer Review (Cycle 15) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-14 fixes

## Traced Flows

### Flow 1: Canon EOS 250D through openMVG → misclassified as mirrorless → duplicate on All Cameras
**Path:**
1. `openmvg.fetch()` → sensor width 22.3mm → `size[0] >= 20` → enters interchangeable-lens branch
2. `_DSLR_NAME_RE.search("Canon EOS 250D")` → pattern `EOS[-\s]+\dD` requires single digit before D → NO MATCH
3. `category = "mirrorless"` (default for non-DSLR interchangeable-lens)
4. `merge_camera_data` → `create_camera_key` → `"canon eos 250d-mirrorless"` ≠ `"canon eos 250d-dslr"` (from Geizhals)
5. Both records preserved → duplicate on All Cameras page

**FINDING: T15-01** — Canon EOS xxxD DSLRs misclassified as mirrorless by incomplete regex.
**Severity:** MEDIUM | **Confidence:** HIGH

Competing hypotheses:
- H1: Change `\dD` to `\d+D` in Canon pattern — addresses root cause, minimal risk
- H2: Add separate pattern for xxxD/xxxD series — more explicit but more patterns to maintain
- H3: Make openMVG category a "suggestion" that merge overrides — architectural change, high risk

Recommended: H1 (simple regex fix).

---

### Flow 2: Samsung NX300 through openMVG → misclassified as DSLR → wrong page
**Path:**
1. `openmvg.fetch()` → sensor width 23.5mm → `size[0] >= 20` → enters interchangeable-lens branch
2. `_DSLR_NAME_RE.search("Samsung NX300")` → pattern `Samsung\s+NX\d{3}` → MATCH
3. `category = "dslr"` (WRONG — NX300 is mirrorless)
4. Camera appears on DSLR page instead of Mirrorless page

**FINDING: T15-02** — Samsung NX pattern incorrectly classifies mirrorless cameras as DSLR.
**Severity:** MEDIUM | **Confidence:** HIGH

Root cause: The C14-01 fix added Samsung NX with the comment "some were DSLR-style" but all Samsung NX cameras are mirrorless. "DSLR-style" refers to body shape, not camera type.

Recommended fix: Remove Samsung NX from the DSLR regex entirely.

---

### Flow 3: Canon EOS R5 through Geizhals rangefinder → triple-duplicate on All Cameras
**Path:**
1. Geizhals DSLR URL → `category="dslr"` for Canon EOS R5... wait, R5 is mirrorless, not DSLR
2. Geizhals Mirrorless URL → `category="mirrorless"` for Canon EOS R5
3. Geizhals Rangefinder URL → `category="rangefinder"` for Canon EOS R5 (misclassified by Geizhals)
4. `create_camera_key` → 3 different keys: `"canon eos r5-dslr"`, `"canon eos r5-mirrorless"`, `"canon eos r5-rangefinder"`
5. All 3 preserved in merge → 3 entries on All Cameras page

Wait — Canon EOS R5 is mirrorless, not DSLR. Let me re-trace:
1. Geizhals Mirrorless URL → `category="mirrorless"` → `create_camera_key` → `"canon eos r5-mirrorless"`
2. Geizhals Rangefinder URL → `category="rangefinder"` → `create_camera_key` → `"canon eos r5-rangefinder"`
3. openMVG → `category="mirrorless"` (correct) → `create_camera_key` → `"canon eos r5-mirrorless"` (merges with #1)
4. Result: 2 entries (mirrorless + rangefinder) — NOT 3 for mirrorless cameras

For DSLRs like Canon EOS 5D Mark IV:
1. Geizhals DSLR URL → `category="dslr"` → key `"canon eos 5d mark iv-dslr"`
2. Geizhals Rangefinder URL → `category="rangefinder"` → key `"canon eos 5d mark iv-rangefinder"`
3. openMVG → `category="mirrorless"` (wrong from regex) → key `"canon eos 5d mark iv-mirrorless"`
4. Result: 3 entries — ALL 3 categories

**FINDING: T15-03** — Geizhals rangefinder + openMVG misclassification produces up to 3 entries per camera.
**Severity:** MEDIUM | **Confidence:** HIGH

This is a compound issue: the Geizhals rangefinder misclassification (data issue) AND the openMVG DSLR misclassification (code issue) combine to produce 3-way duplicates.

---

### Flow 4: BOM-prefixed openMVG CSV → 0 records fetched
**Path:**
1. GitHub raw CSV gets BOM (e.g., maintainer saves with Excel)
2. `http_get` returns BOM-prefixed string
3. `csv.DictReader(io.StringIO(body))` → first fieldname = `"﻿CameraMaker"`
4. `row.get("CameraMaker")` → returns None (key doesn't match)
5. `if not maker or not model: continue` → every row skipped
6. `specs = []` → 0 records returned
7. Site generated with no openMVG cameras → data loss

**FINDING: T15-04** — openMVG CSV fetch has no BOM defense, causing potential 0-record parse.
**Severity:** LOW | **Confidence:** HIGH

---

## Summary
- NEW findings: 4 (3 MEDIUM, 1 LOW)
- T15-01: Canon EOS xxxD misclassified as mirrorless — MEDIUM
- T15-02: Samsung NX misclassified as DSLR — MEDIUM
- T15-03: Geizhals rangefinder + openMVG DSLR misclassification compound — MEDIUM
- T15-04: openMVG CSV BOM vulnerability — LOW
