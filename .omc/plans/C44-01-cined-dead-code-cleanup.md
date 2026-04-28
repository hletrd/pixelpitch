# Plan: Cycle 44 Findings — CineD Dead Code Cleanup

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR44-01, CR44-02, CRIT44-01, CRIT44-02, CRIT44-03, ARCH44-01, ARCH44-02, TR44-01, DBG44-03, DOC44-01, TE44-02

---

## Task 1: Remove FORMAT_TO_MM dict from cined.py — C44-01 (core) [COMPLETED]

**Finding:** C44-01 (7-agent consensus)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/cined.py`, lines 37-51

### Problem

After C43-01 removed `size = FORMAT_TO_MM.get(fmt.lower())`, the FORMAT_TO_MM dict is defined at module level but never referenced by any executable code. It's only mentioned in comments and the docstring. The module docstring claims "The FORMAT_TO_MM table is kept for the regex coverage test only" but no such test exists. This is dead code that could mislead future maintainers.

### Implementation

In `sources/cined.py`:

1. Remove the `FORMAT_TO_MM` dict (lines 37-51):

**Before:**
```python
FORMAT_TO_MM: dict[str, tuple[float, float]] = {
    "full frame": (36.0, 24.0),
    "super 35": (24.89, 18.66),
    ...
    "medium format": (43.8, 32.9),
}
```

**After:** Remove the entire dict.

2. Update the module docstring to remove references to FORMAT_TO_MM:

**Before (partial):**
```
The ``FORMAT_TO_MM`` table is kept for the regex coverage test only.
```

**After:** Remove this line. Update the surrounding docstring to reflect the current behavior (spec.size is None for format-class-only entries, the template shows "unknown").

---

## Task 2: Remove dead format extraction code from _parse_camera_page — C44-02 [COMPLETED]

**Finding:** C44-02 (5-agent consensus)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/cined.py`, _parse_camera_page function, lines 92-119

### Problem

After C43-01 removed `FORMAT_TO_MM.get(fmt.lower())`, the format extraction regex (fmt_m, lines 92-97) and fmt variable assignment are computed but never used. The `if size is None and fmt:` block (lines 106-119) contains only a `pass` statement with comments. The entire format detection code path is dead code.

### Implementation

In `sources/cined.py`, `_parse_camera_page` function:

1. Remove the format extraction regex and fmt assignment (lines 92-97):

**Before:**
```python
fmt_m = re.search(
    r"(Full Frame|Super[- ]?35(?:\s*mm)?|APS-C|Micro Four Thirds|Four Thirds|1\"|1[- ]inch|2/3\"|2/3[- ]inch|Medium Format)",
    body_text,
    re.IGNORECASE,
)
fmt = fmt_m.group(1) if fmt_m else ""
```

**After:** Remove these lines entirely.

2. Remove the `if size is None and fmt:` block with its comments (lines 106-119):

**Before:**
```python
if size is None and fmt:
    # Don't set spec.size from FORMAT_TO_MM lookup — the lookup
    # provides approximate dimensions from the format class name.
    # Setting spec.size from the lookup prevents merge_camera_data
    # from preserving more accurate measured values from Geizhals
    # (because the merge only preserves existing spec.size when
    # new spec.size is None). Leave spec.size = None; the template
    # will show "unknown" for sensor size when no Geizhals data
    # exists, which is more honest than showing an approximation
    # as if it were measured data.
    # Note: we also don't set spec.type because format class names
    # like "Super 35" or "APS-C" are not fractional-inch types that
    # TYPE_SIZE understands.
    pass
```

**After:** Remove the entire block (no replacement needed).

---

## Task 3: Update CineD module docstring — C44-01 cleanup [COMPLETED]

**Finding:** CRIT44-03, DOC44-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/cined.py`, module docstring

### Implementation

Update the module docstring to:
1. Remove the reference to FORMAT_TO_MM being "kept for the regex coverage test"
2. Clarify the current behavior: spec.size is set from explicit mm dimensions when available, otherwise left as None

**Before (partial docstring):**
```
Pixel pitch is derived (sensor_width_mm / sensor_pixels_w) when both
fields are present. If only the format class is given (no explicit mm
dimensions), we leave ``spec.size`` as None so that ``merge_camera_data``
can preserve more accurate measured values from Geizhals when available.
The template will show "unknown" for sensor size when no measured data
exists for the camera — this is more honest than presenting FORMAT_TO_MM
approximations as if they were measured. The ``FORMAT_TO_MM`` table is
kept for the regex coverage test only.
```

**After:**
```
Pixel pitch is derived (sensor_width_mm / sensor_pixels_w) when both
fields are present. If only the format class is given (no explicit mm
dimensions), we leave ``spec.size`` as None so that ``merge_camera_data``
can preserve more accurate measured values from Geizhals when available.
The template will show "unknown" for sensor size when no measured data
exists for the camera — this is more honest than presenting approximations
as if they were measured data.
```

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- CineD module imports without error
- FORMAT_TO_MM dict is no longer defined
- `_parse_camera_page` no longer references fmt or FORMAT_TO_MM
- Module docstring accurately describes current behavior
- All existing tests pass (no functional changes, only dead code removal)

---

## Deferred Findings

No new deferred findings. All findings from cycle 44 reviews are scheduled for implementation in this plan.
