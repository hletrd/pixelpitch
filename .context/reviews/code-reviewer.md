# Code Review (Cycle 22) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-21 fixes, focusing on NEW issues

## C22-01: `merge_camera_data` year-change log message is unreachable — `elif` attached to wrong `if`

**File:** `pixelpitch.py`, lines 428-437
**Severity:** MEDIUM | **Confidence:** HIGH (static analysis)

The C21-01 fix added SpecDerived field preservation after the Spec field preservation block. However, the `elif` branch that logs year changes (added in C20-03) is now structurally attached to the SpecDerived pitch preservation `if`:

```python
if new_spec.pitch is None and existing_spec.pitch is not None:
    new_spec.pitch = existing_spec.pitch
elif (                              # <-- this elif belongs to the pitch preservation
    new_spec.spec.year is not None  # <-- but compares YEAR, not pitch
    and existing_spec.spec.year is not None
    and new_spec.spec.year != existing_spec.spec.year
):
    print(
        f"  Year changed for {new_spec.spec.name[:40]}: "
        f"{existing_spec.spec.year} -> {new_spec.spec.year}"
    )
```

This `elif` is only reachable when `new_spec.pitch is NOT None` (i.e., the SpecDerived pitch is already set) AND the year conditions are met. If `new_spec.pitch is None` and `existing_spec.pitch is not None`, the `if` branch executes (preserving pitch), and the `elif` is skipped entirely — even if the year changed.

**Concrete failure scenario:**
1. Camera "Canon EOS R5" has `pitch=4.39, year=2020` in existing data
2. New source has `pitch=None, year=2021`
3. After merge: `new_spec.pitch = 4.39` (preserved from existing, correct)
4. But the year change log "Year changed for Canon EOS R5: 2020 -> 2021" is NEVER printed because the `elif` is attached to the pitch-preservation `if`
5. The year IS correctly set to 2021 (new data takes precedence), but the diagnostic log is silently suppressed

**Fix:** Move the year-change `elif` out of the SpecDerived block and make it a standalone `if`:

```python
if new_spec.pitch is None and existing_spec.pitch is not None:
    new_spec.pitch = existing_spec.pitch

if (
    new_spec.spec.year is not None
    and existing_spec.spec.year is not None
    and new_spec.spec.year != existing_spec.spec.year
):
    print(
        f"  Year changed for {new_spec.spec.name[:40]}: "
        f"{existing_spec.spec.year} -> {new_spec.spec.year}"
    )
```

---

## C22-02: `_parse_camera_name` Sony DSC hyphen is stripped to space — breaks "DSC-HX400" pattern

**File:** `sources/imaging_resource.py`, line 173
**Severity:** LOW | **Confidence:** MEDIUM

When the Sony name starts with "Sony" and falls through to URL-based parsing, the slug hyphens are replaced with spaces via `.replace("-", " ")`. For cameras like "Sony DSC-HX400", the URL slug is "sony-dsc-hx400". After `.replace("-", " ")` and `.title()`, it becomes "Sony Dsc Hx400". The DSC normalizer converts "Dsc" to "DSC" and the HX normalizer converts "Hx400" to "HX400", producing "Sony DSC HX400".

However, if the camera name is provided directly via "Model Name" (e.g., "Sony DSC-HX400"), the `.title()` call in the non-URL path does NOT apply, and the name flows through `normalise_name()` which only collapses whitespace. The hyphen in "DSC-HX400" is preserved, resulting in "Sony DSC-HX400" (with hyphen) vs. "Sony DSC HX400" (without hyphen from URL). This inconsistency means the same camera can have two different names depending on whether it came from the Model Name field or the URL.

The test expects "Sony DSC HX400" (space), which is the URL-derived form. But if a real IR page has Model Name = "Sony DSC-HX400", the name would have a hyphen, creating a potential dedup mismatch.

**Concrete failure scenario:** IR page has Model Name "Sony DSC-HX400". The name is not URL-derived, so hyphens are preserved. Apotelyt might produce "Sony DSC-HX400" too. But if GSMArena or another source produces "Sony DSC HX400" (without hyphen), the merge keys would differ, creating a duplicate.

**Fix:** Add a hyphen-to-space normalizer for "DSC-" patterns after the DSC uppercase normalizer:
```python
cleaned = re.sub(r'\bDSC-', 'DSC ', cleaned)
```

---

## Summary

- C22-01 (MEDIUM): Year-change log unreachable due to `elif` attached to wrong `if` — diagnostic regression from C21-01 fix
- C22-02 (LOW): Sony DSC hyphen inconsistency between Model Name and URL paths — potential dedup mismatch
