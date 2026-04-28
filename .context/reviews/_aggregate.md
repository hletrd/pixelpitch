# Aggregate Review (Cycle 28) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-27 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (all checks). C27-01 (PITCH_UM_RE "um" fix) and C27-02 (year validation) implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 28 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| imaging_resource.py pitch float() missing ValueError guard | code-reviewer, critic, verifier, tracer, debugger, test-engineer | MEDIUM |
| CineD year regex produces unvalidated years | verifier, debugger | LOW |
| DRY inconsistency — source modules have divergent local regex copies | critic, architect | LOW |
| GSMArena fallback µm regex only matches micro-sign | code-reviewer | LOW |
| sources/__init__.py PITCH_UM_RE lacks inline doc comment | document-specialist | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C28-01: imaging_resource.py pitch float() missing ValueError guard — incomplete C26-02 fix

**Sources:** CR28-01, CRIT28-01, V28-02, TR28-01, DBG28-01, TE28-01
**Severity:** MEDIUM | **Confidence:** HIGH (6-agent consensus)

The C26-02 fix added ValueError guards to `size` (line 229) and `mpix` (line 246) in `sources/imaging_resource.py`'s `fetch_one()`, but **missed** `pitch` at line 238:

```python
# Line 238 — NO try/except:
pitch = float(m.group(1))

# Line 229 — HAS try/except:
try:
    size = (float(m.group(1)), float(m.group(2)))
except ValueError:
    size = None

# Line 246 — HAS try/except:
try:
    mpix = float(m.group(1))
except ValueError:
    mpix = None
```

The `IR_PITCH_RE` pattern `([\d.]+)` can match malformed values like "5.1.2" from text "5.1.2 microns". `float("5.1.2")` raises `ValueError`, which would crash `fetch_one()` and propagate to `fetch()`, aborting the entire Imaging Resource scrape.

**Verified:** `IR_PITCH_RE.search("5.1.2 microns")` matches "5.1.2"; `float("5.1.2")` raises `ValueError`.

**Fix:** Wrap line 238 in try/except ValueError:
```python
if m:
    try:
        pitch = float(m.group(1))
    except ValueError:
        pitch = None
```

Add a test for malformed pitch values.

---

### C28-02: CineD year regex produces unvalidated years without range check

**Sources:** V28-03, DBG28-02
**Severity:** LOW | **Confidence:** HIGH (2-agent consensus)

**File:** `sources/cined.py`, line 114

The CineD year parsing uses `int(year_m.group(1))` without range validation:
```python
year_m = re.search(r"Release Date.{0,40}?(\d{4})", body_text, re.IGNORECASE)
year = int(year_m.group(1)) if year_m else parse_year(body_text[:500])
```

The regex matches any 4-digit number (0000-9999). The `parse_year()` fallback validates 19xx/20xx, but the primary path does not. A CineD page with text "Release Date: model1234" would produce year=1234, which would display on the website.

The C27-02 fix added year range validation to `parse_existing_csv()` (1900-2100), but the source module itself produces years before they reach the CSV parser.

**Fix:** Add a range check after the int conversion:
```python
y = int(year_m.group(1))
year = y if 1900 <= y <= 2100 else None
```

---

### C28-03: DRY inconsistency — source modules maintain divergent local regex copies

**Sources:** CRIT28-02, ARCH28-01
**Severity:** LOW | **Confidence:** HIGH (2-agent consensus)

After the C25-01 and C26-01 centralization of shared regex patterns, 3 source modules still maintain their own local copies that diverge from the shared patterns:

- `apotelyt.py` SIZE_RE — identical to shared SIZE_MM_RE
- `apotelyt.py` PITCH_RE — missing `um`, `&micro;m`, `&#956;m` (vs shared PITCH_UM_RE)
- `apotelyt.py` MPIX_RE — only matches "Megapixel" (vs shared MPIX_RE which also matches "MP", "Mega pixels")
- `cined.py` SIZE_RE — identical to shared SIZE_MM_RE
- `gsmarena.py` PITCH_RE — has `um` but missing `microns`, `&micro;m`, `&#956;m`

**Impact:** Currently no data is lost because each source works with its own local pattern. But future shared pattern changes won't propagate to local copies.

**Fix:** Replace local regex copies with imports from `sources/__init__.py`.

---

### C28-04: GSMArena fallback camera detection regex only matches micro-sign µm

**Sources:** CR28-02
**Severity:** LOW | **Confidence:** HIGH (1-agent)

**File:** `sources/gsmarena.py`, line 121

The fallback camera detection uses `re.search(r"\d+\s*MP.*?µm", v)` which only matches micro-sign `µm`, not Greek `μm` or ASCII `um`. If GSMArena changes their HTML to use a different mu variant, phones without a "Main Camera" field would be silently dropped.

**Impact:** Currently GSMArena uses `µm`, so no data is lost. This is a latent format-change risk.

**Fix:** Replace with `PITCH_RE.search(v)` or add `μm`/`um` to the inline regex.

---

### C28-05: sources/__init__.py PITCH_UM_RE lacks inline documentation comment

**Sources:** DOC28-01
**Severity:** LOW | **Confidence:** MEDIUM (1-agent)

**File:** `sources/__init__.py`, line 66

The PITCH_UM_RE pattern is the canonical definition but has no inline comment documenting the supported formats. A developer reading only `sources/__init__.py` would need to read the regex itself to understand what it matches. The comment in `pixelpitch.py` line 44 documents the pattern, but `sources/__init__.py` is the canonical location.

**Fix:** Add a comment above the PITCH_UM_RE definition listing supported formats.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 5 (1 MEDIUM, 4 LOW)
- Cross-agent consensus findings (3+ agents): 1 (C28-01: 6-agent)
- 2-agent consensus findings: 2 (C28-02, C28-03)
- 1-agent findings: 2 (C28-04, C28-05)
