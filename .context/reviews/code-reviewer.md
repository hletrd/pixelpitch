# Code Review (Cycle 28) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes, focusing on NEW issues

## Previous Findings Status

C27-01 (PITCH_UM_RE "um" fix) and C27-02 (parse_existing_csv year validation) both implemented and verified. All previous fixes stable. Gate tests pass.

## New Findings

### CR28-01: imaging_resource.py pitch float() missing ValueError guard — incomplete C26-02 fix

**File:** `sources/imaging_resource.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

The C26-02 fix added ValueError guards to `size` (line 229, try/except) and `mpix` (line 246, try/except) in `fetch_one()`, but **missed** `pitch` at line 238:

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

The `IR_PITCH_RE` pattern uses `([\d.]+)` which can match malformed values like `"5.1.2"` from text such as "5.1.2 microns". `float("5.1.2")` raises `ValueError`, which would crash `fetch_one()` and propagate up to `fetch()`, aborting the entire Imaging Resource scrape.

**Verified:** `IR_PITCH_RE.search("5.1.2 microns")` matches `"5.1.2"`, and `float("5.1.2")` raises `ValueError`.

**Fix:** Wrap line 238 in try/except ValueError, consistent with size and mpix:
```python
if m:
    try:
        pitch = float(m.group(1))
    except ValueError:
        pitch = None
```

---

### CR28-02: GSMArena fallback camera detection only matches micro-sign µm — silent phone drop if format changes

**File:** `sources/gsmarena.py`, line 121
**Severity:** LOW | **Confidence:** HIGH

The fallback camera detection in `_phone_to_spec()` uses a hardcoded regex with literal `µm`:

```python
if "MP" in v and re.search(r"\d+\s*MP.*?µm", v):
    cam = v
    break
```

This only matches `µm` (U+00B5 micro sign), not `μm` (U+03BC Greek mu) or `um` (ASCII). The GSMArena local `PITCH_RE` (line 50) correctly handles all three variants, but this fallback detection regex does not.

**Impact:** If GSMArena changes their HTML to use Greek `μm` instead of micro-sign `µm` in the camera section values, phones that lack a "Main Camera" field would be silently dropped (cam stays empty, `_phone_to_spec` returns None). Currently, GSMArena uses `µm`, so no data is lost. This is a DRY consistency concern and a latent format-change risk.

**Fix:** Replace the inline regex with `PITCH_RE.search(v)` or at minimum add `μm` and `um` to the alternation: `re.search(r"\d+\s*MP.*?(?:µm|μm|um)", v)`.

---

## Summary

- CR28-01 (MEDIUM): imaging_resource.py pitch float() missing ValueError guard — incomplete C26-02 fix
- CR28-02 (LOW): GSMArena fallback µm regex only matches micro-sign — latent format-change risk
