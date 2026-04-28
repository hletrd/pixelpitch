# Plan: Cycle 38 Findings — Zero-Pitch Template "unknown" Rendering

**Created:** 2026-04-28
**Status:** PENDING
**Source Reviews:** CR38-01, CRIT38-01, V38-02, TR38-01, ARCH38-01, DBG38-01, DES38-01, TE38-01

---

## Task 1: Update Jinja2 template to render "unknown" for `pitch=0.0` — C38-01 (core)

**Finding:** C38-01 (8-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `templates/pixelpitch.html`, lines 84-88

### Problem

The C37-02 fix added `pitch === 0` to JS `isInvalidData`, hiding zero-pitch rows by default. But the template still renders "0.0 µm" for `pitch=0.0`. This creates a UX contradiction: the template says "valid number" while JS says "invalid, hide it".

A 0.0 µm pixel pitch is physically impossible. The template should render "unknown" for `pitch=0.0`, consistent with how `None` pitch is rendered and consistent with JS treating 0.0 as invalid.

### Implementation

1. In `templates/pixelpitch.html`, line 84, change:
   ```jinja2
   {% if spec.pitch is not none %}
     {{ spec.pitch|round(1) }} µm
   {% else %}
     <span class="text-muted">unknown</span>
   {% endif %}
   ```
   To:
   ```jinja2
   {% if spec.pitch is not none and spec.pitch != 0.0 %}
     {{ spec.pitch|round(1) }} µm
   {% else %}
     <span class="text-muted">unknown</span>
   {% endif %}
   ```

2. Similarly, for mpix (line 76-80), change:
   ```jinja2
   {% if spec.spec.mpix is not none %}
     {{ spec.spec.mpix|round(1) }} MP
   {% else %}
     <span class="text-muted">unknown</span>
   {% endif %}
   ```
   To:
   ```jinja2
   {% if spec.spec.mpix is not none and spec.spec.mpix != 0.0 %}
     {{ spec.spec.mpix|round(1) }} MP
   {% else %}
     <span class="text-muted">unknown</span>
   {% endif %}
   ```

---

## Task 2: Update `test_template_zero_pitch_rendering` to expect "unknown" — TE38-01

**Finding:** TE38-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`, lines 496-523

### Problem

The test currently asserts that 0.0 pitch renders as "0.0 µm" and 0.0 mpix renders as "0.0 MP". After Task 1, these should render as "unknown".

### Implementation

1. In `test_template_zero_pitch_rendering()`, update assertions:
   - Change the mpix assertion to verify "unknown" appears for 0.0 mpix
   - Change the pitch assertion to verify "unknown" appears for 0.0 pitch
   - The test should verify that "0.0 µm" is NOT present and "unknown" IS present

---

## Task 3: Update `test_sorted_by_zero_values` to reflect "unknown" rendering for 0.0 pitch

**Finding:** Follow-up from Task 1
**Severity:** LOW | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`, lines 469-491

### Problem

`test_sorted_by_zero_values` creates a camera with `pitch_val=0.0`. After Task 1, the template would render "unknown" for this camera's pitch. The test itself still works correctly (it tests sorting, not rendering), but it should be verified to still pass after the template change.

### Implementation

1. Verify the test still passes after Task 1 changes. No code changes needed unless the test breaks.

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- Updated `test_template_zero_pitch_rendering` must pass
- `test_sorted_by_zero_values` must still pass
- Template with `pitch=0.0` must render "unknown" (not "0.0 µm")
- Template with `mpix=0.0` must render "unknown" (not "0.0 MP")
- JS `isInvalidData` still correctly catches `pitch === 0`

---

## Deferred Findings

### C38-02: `match_sensors` latent ZeroDivisionError risk

**File:** `pixelpitch.py`, line 243
**Original Severity:** LOW | **Original Confidence:** MEDIUM
**Reason for deferral:** Currently guarded by `megapixels > 0` check on line 242. No code path can trigger the division by zero. Only a theoretical risk if the guard is removed in the future.
**Exit criterion:** If `match_sensors` megapixel guard is ever changed from `> 0` to `>= 0`, a ZeroDivisionError guard must be added inside the `any()` expression.
