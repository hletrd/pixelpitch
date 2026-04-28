# Debugger Review (Cycle 18) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository latent bug review after cycles 1-17 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

- D16-01 (sensor_size_from_type crash): Fixed.
- D16-02 (merge dedup): Fixed.
- D16-03 (Pentax regex): Fixed (all letter-suffix models now covered).
- D16-04 (http_get OSError): Fixed.
- D17-01, D17-02, D17-03: All fixed.

## New Findings

### D18-01: Scatter plot includes hidden/invalid data — user-trust regression
**File:** `templates/pixelpitch.html`, lines 337-346
**Severity:** MEDIUM | **Confidence:** HIGH

The scatter plot data collection iterates over all table rows including hidden ones. This is a functional mismatch between the table view and the plot view. The user's explicit filter choice ("Hide possibly invalid data") is silently ignored by the scatter plot.

**Failure mode:** User enables "Hide possibly invalid data" (default ON), then creates scatter plot. The plot shows data points from hidden rows (e.g., a camera with 12 µm pixel pitch). The user sees an outlier in the plot that doesn't correspond to any visible table row. This erodes trust in the data display.

**Fix:** Add `if (!row.is(':visible')) return;` at the start of the `.each()` loop in `createPlot()`.

---

### D18-02: SENSOR_TYPE_RE in pixelpitch.py doesn't match Unicode quotes — silent data loss
**File:** `pixelpitch.py`, line 43
**Severity:** LOW | **Confidence:** MEDIUM

If Geizhals HTML ever uses Unicode quotes (U+2033) for sensor format in title attributes, `SENSOR_TYPE_RE` would fail to match, and `parse_sensor_field()` would return `type=None`. The sensor type would be silently lost, and the camera would appear with "unknown" sensor size.

This is the same class of issue as D17-03 (GSMArena Unicode quotes), which was fixed in C17-03. The fix missed `SENSOR_TYPE_RE` in pixelpitch.py.

**Fix:** Update `SENSOR_TYPE_RE` to match Unicode quotes: `re.compile(r'(1/[\d.]+)(?:\"|″)')`.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- D18-01: Scatter plot includes hidden data — MEDIUM
- D18-02: SENSOR_TYPE_RE doesn't match Unicode quotes — LOW
