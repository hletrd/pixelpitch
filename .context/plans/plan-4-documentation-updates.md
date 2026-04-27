# Plan 4: Documentation Updates

**Status:** completed
**Priority:** P2 (misleading but not broken)
**Findings addressed:** F8, F9, F28, F29

## Problem

Documentation is outdated — README references old URL and single source, template alert text is misleading, digicamdb.py comment is wrong, sources/__init__.py docstring is incomplete.

## Implementation Steps

### Step 1: Update template alert text (F8)
- [ ] `templates/pixelpitch.html` line 127: Change "All camera data was retrieved from geizhals.at." to something like "Camera data is aggregated from multiple sources including geizhals.eu, Imaging Resource, Apotelyt, GSMArena, and CineD."
- [ ] Keep the caveat about accuracy

### Step 2: Update README.md (F9)
- [ ] Change "Camera data is read from http://geizhals.at" to document the multi-source architecture
- [ ] Fix URL from `http://geizhals.at` to `https://geizhals.eu/`
- [ ] List all data sources with brief descriptions
- [ ] Add section about alternative sources (pixelpitch.py source command)

### Step 3: Fix digicamdb.py misleading comment (F28)
- [ ] Change comment "Re-tag category so downstream consumers can distinguish provenance" to reflect that it just inherits openMVG's category
- [ ] Or remove the module if it truly provides no additional value

### Step 4: Update sources/__init__.py docstring (F29)
- [ ] Add note that individual source modules may accept additional keyword arguments (e.g., `sleep_seconds`, `brands`)

## Exit Criteria
- Template alert text reflects multi-source architecture
- README documents all data sources with correct URLs
- No misleading comments in source code
- Offline test gate passes
