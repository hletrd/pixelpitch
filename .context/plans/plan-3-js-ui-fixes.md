# Plan 3: JavaScript & UI Fixes

**Status:** completed
**Priority:** P1 (user-facing bug)
**Findings addressed:** F5, F16, F22

## Problem

Box-plot toggle button is broken (uses deprecated `arguments.callee.caller`). Theme flashes on page load. About page has Bootstrap grid nesting issue.

## Implementation Steps

### Step 1: Fix broken box-plot toggle (F5)
- [ ] Replace `arguments.callee.caller` pattern with a state variable
- [ ] Refactor the click handler to use `let plotVisible = false` toggle
- [ ] Test that Create Plot → Hide Plot → Create Plot cycle works

### Step 2: Fix theme flash on page load (F16)
- [ ] Move the theme detection/application inline script from the bottom of `<body>` to `<head>`, before the CSS `<link>` tags
- [ ] This prevents the brief flash of default theme before stored theme is applied
- [ ] Keep the event listeners (DOMContentLoaded, matchMedia) at the bottom

### Step 3: Fix about.html grid nesting (F22)
- [ ] Move the inner `<div class="row">` (line 23) to be a sibling of the first `<div class="col mx-auto">`
- [ ] Restructure so rows are direct children of the container or properly nested

## Exit Criteria
- Box-plot toggle works for multiple cycles without errors
- No visible theme flash on page load
- About page has valid Bootstrap grid structure
- Offline test gate passes
