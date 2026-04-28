# Designer Review (Cycle 18) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository UI/UX re-review after cycles 1-17 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

- UX16-01 (duplicate entries): Fixed — merge dedup working.
- UX16-02 (Pentax misclassification): Fixed — KP/KF/K-r/K-x and Nikon Df now DSLR.
- UX17-01, UX17-02: Fixed with C17-01, C17-02.

## New Findings

### UX18-01: Scatter plot includes hidden/invalid data points — inconsistent with visibility filter
**File:** `templates/pixelpitch.html`, lines 337-346 (JavaScript `createPlot` function)
**Severity:** MEDIUM | **Confidence:** HIGH

When the user enables "Hide possibly invalid data" and then clicks "Create Scatter Plot", the plot includes data from hidden rows. The `createPlot()` function iterates over all `#table_with_pitch tbody tr` elements without checking `row.is(':visible')`:

```javascript
$("#table_with_pitch tbody tr").each(function(i, el) {
    const row = $(el);
    const name = row.find('td:first-child a').text().trim();
    const pitch = parseFloat(row.attr('data-pitch'));
    const year = parseInt(row.attr('data-year'));
    // No visibility check!
    if (!isNaN(pitch) && pitch > 0 && !isNaN(year) && year >= 2000 ...) {
        data.push({ name, year, value: pitch });
    }
});
```

**Concrete failure scenario:**
1. User has "Hide possibly invalid data" checked (default)
2. A camera with pixel pitch > 10 µm (e.g., a misclassified camera) is hidden from the table
3. User clicks "Create Scatter Plot"
4. The hidden outlier appears as a data point in the scatter plot
5. User is confused — the plot shows data that isn't visible in the table

**Fix:** Add `if (!row.is(':visible')) return;` at the start of the `.each()` callback, or after extracting the row reference.

---

### UX18-02: Table sorter "Sensor Size" column sorts as text, not numerically
**File:** `templates/pixelpitch.html`, lines 218-233 (tablesorter header config)
**Severity:** LOW | **Confidence:** HIGH

The tablesorter configuration sets the Sensor Size column to `sorter: "text"`. Since sensor sizes are displayed as "35.9 x 23.9 mm" (text format), clicking the Sensor Size column header sorts alphabetically.

**Concrete failure:** "9.84 x 7.40 mm" sorts after "35.9 x 23.9 mm" alphabetically (because "9" > "3" in ASCII). The correct numeric sort would place the smaller sensor first.

The `<tr>` elements have `data-sensor-width` attributes that could be used for numeric sorting, but tablesorter doesn't use row data attributes by default.

**Fix:** Add a custom tablesorter parser that reads the `data-sensor-width` attribute for numeric sorting:
```javascript
$.tablesorter.addParser({
    id: 'sensor-width',
    is: function() { return false; },
    format: function(s, table, cell) {
        return $(cell).closest('tr').attr('data-sensor-width') || 0;
    },
    type: 'numeric'
});
```
Then set the Sensor Size column header to `{ sorter: "sensor-width" }`.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- UX18-01: Scatter plot includes hidden/invalid data — MEDIUM
- UX18-02: Sensor Size column sorts as text, not numerically — LOW
