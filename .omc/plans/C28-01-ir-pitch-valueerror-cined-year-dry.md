# Plan: Cycle 28 Findings — IR Pitch ValueError Guard, CineD Year Validation, DRY Consistency

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR28-01, CRIT28-01, V28-02, TR28-01, DBG28-01, TE28-01, V28-03, DBG28-02, CRIT28-02, ARCH28-01, CR28-02, DOC28-01

---

## Task 1: Add ValueError guard to imaging_resource.py pitch float() — C28-01

**Finding:** C28-01 (6-agent consensus: code-reviewer, critic, verifier, tracer, debugger, test-engineer)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `sources/imaging_resource.py`, `tests/test_parsers_offline.py`

### Problem

The C26-02 fix added ValueError guards to `size` (line 229) and `mpix` (line 246) but missed `pitch` (line 238). The `IR_PITCH_RE` pattern `([\d.]+)` can match malformed values like "5.1.2", and `float("5.1.2")` raises `ValueError`, which would crash the entire Imaging Resource scrape.

### Implementation

1. In `sources/imaging_resource.py` line 237-238, wrap the pitch float() in try/except:
   ```python
   m = IR_PITCH_RE.search(pitch_text)
   if m:
       try:
           pitch = float(m.group(1))
       except ValueError:
           pitch = None
   ```

2. Run gate tests to verify no regressions.

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- Manually verified: `IR_PITCH_RE.search("5.1.2 microns")` matches "5.1.2", and the try/except catches the ValueError
- Commit: 62c8123

---

## Task 2: Add year range validation to cined._parse_camera_page — C28-02

**Finding:** C28-02 (2-agent consensus: verifier, debugger)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/cined.py`

### Problem

The CineD year parsing uses `int(year_m.group(1))` without range validation. The regex `r"Release Date.{0,40}?(\d{4})"` matches any 4-digit number (0000-9999). A page with "Release Date: model1234" would produce year=1234.

### Implementation

1. In `sources/cined.py` line 114, add range validation:
   ```python
   year_m = re.search(r"Release Date.{0,40}?(\d{4})", body_text, re.IGNORECASE)
   if year_m:
       y = int(year_m.group(1))
       year = y if 1900 <= y <= 2100 else None
   else:
       year = parse_year(body_text[:500])
   ```

2. Run gate tests to verify no regressions.

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- The CineD source is browser-dependent, so offline testing is limited. The `parse_year` fallback already validates 19xx/20xx.
- Commit: 707013a

---

## Task 3: Replace local regex copies with shared imports — C28-03

**Finding:** C28-03 (2-agent consensus: critic, architect)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/apotelyt.py`, `sources/cined.py`, `sources/gsmarena.py`

### Problem

After the C25-01 and C26-01 centralization of shared regex patterns, 3 source modules still maintain local regex copies that diverge from the shared patterns:
- `apotelyt.py` SIZE_RE — identical to shared SIZE_MM_RE (can be replaced)
- `apotelyt.py` PITCH_RE — missing `um`, `&micro;m`, `&#956;m` (should use shared PITCH_UM_RE)
- `apotelyt.py` MPIX_RE — only matches "Megapixel" (shared MPIX_RE matches "MP", "Mega pixels" too — Apotelyt pages only use "Megapixel" so the shared pattern is a strict superset)
- `cined.py` SIZE_RE — identical to shared SIZE_MM_RE (can be replaced)
- `gsmarena.py` PITCH_RE — has `um` but missing `microns`, `&micro;m`, `&#956;m` (should use shared PITCH_UM_RE)

### Implementation

1. In `sources/apotelyt.py`:
   - Add import: `from . import SIZE_MM_RE, PITCH_UM_RE, MPIX_RE as SHARED_MPIX_RE`
   - Replace local `SIZE_RE` with `SIZE_MM_RE`
   - Replace local `PITCH_RE` with `PITCH_UM_RE`
   - Replace local `MPIX_RE` with `SHARED_MPIX_RE` (or keep local alias for readability)
   - Remove the local regex definitions

2. In `sources/cined.py`:
   - Add import: `from . import SIZE_MM_RE`
   - Replace local `SIZE_RE` with `SIZE_MM_RE`
   - Remove the local regex definition

3. In `sources/gsmarena.py`:
   - Add `PITCH_UM_RE` to the existing import from `.`
   - Replace local `PITCH_RE` with `PITCH_UM_RE`
   - Remove the local regex definition

4. Run gate tests to verify no regressions.

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- Each source module's existing tests still pass (Apotelyt fixture, GSMArena fixture)
- The shared patterns are strict supersets of the local patterns, so no data loss
- Commit: 7d86a85 (combined with Tasks 4-5)

---

## Task 4: Fix GSMArena fallback µm regex to use PITCH_RE — C28-04

**Finding:** C28-04 (1-agent: code-reviewer)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/gsmarena.py`

### Problem

The fallback camera detection on line 121 uses `re.search(r"\d+\s*MP.*?µm", v)` which only matches micro-sign `µm`, not Greek `μm` or ASCII `um`. If GSMArena changes their HTML format, phones could be silently dropped.

### Implementation

After Task 3 replaces local `PITCH_RE` with `PITCH_UM_RE`, update line 121 to use the shared pattern:

```python
if "MP" in v and PITCH_UM_RE.search(v):
    cam = v
    break
```

This handles all µm/μm/um variants consistently.

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- The GSMArena fixture test still passes
- Commit: 7d86a85 (combined with Task 3-5)

---

## Task 5: Add inline documentation comment for PITCH_UM_RE in sources/__init__.py — C28-05

**Finding:** C28-05 (1-agent: document-specialist)
**Severity:** LOW | **Confidence:** MEDIUM
**Files:** `sources/__init__.py`

### Problem

The PITCH_UM_RE pattern on line 66 is the canonical definition but has no inline comment documenting the supported formats.

### Implementation

Add a comment above line 66:
```python
# PITCH_UM_RE matches pixel pitch values with these suffixes:
# µm (micro sign), um (ASCII), microns/micron, μm (Greek mu),
# &micro;m and &#956;m (HTML entities).
PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|um|microns?|μm|&micro;m|&#0?956;m)", re.IGNORECASE)
```

### Verification — DONE

- Visual check: comment matches the regex pattern
- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- Commit: 7d86a85 (combined with Tasks 3-4)

---

## Deferred Findings

None. All findings are scheduled for implementation.
