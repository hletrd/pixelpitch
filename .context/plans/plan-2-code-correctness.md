# Plan 2: Code Correctness Fixes

**Status:** completed
**Priority:** P0/P1 (correctness, security)
**Findings addressed:** F3, F4, F6, F10, F11, F12, F13, F14, F15, F17

## Problem

Multiple correctness issues across the codebase: assert used for runtime checks, sys.exit in library code, CWD-relative paths, missing deduplication, merge key collisions, sensor matching skips None mpix, debugging port exposed.

## Implementation Steps

### Step 1: Replace `assert` with proper exception (F3)
- [ ] `pixelpitch.py` line 399: `assert rows, "No entries found"` → `if not rows: raise RuntimeError("No entries found")`

### Step 2: Replace `sys.exit` with `ValueError` (F4)
- [ ] `pixelpitch.py` line 907: `sys.exit(1)` → `raise ValueError(f"Unknown source: {name}")`
- [ ] Update the caller in `main()` to catch `ValueError` if needed

### Step 3: Fix CWD-relative file paths (F6)
- [ ] Add `SCRIPT_DIR = Path(__file__).resolve().parent` at module level
- [ ] `_get_env()`: `FileSystemLoader(SCRIPT_DIR / "templates")` instead of `"templates"`
- [ ] `load_sensors_database()`: `SCRIPT_DIR / "sensors.json"` instead of `"sensors.json"`
- [ ] `render_html()`: `SCRIPT_DIR / "sitemap.xml"` instead of `"sitemap.xml"`
- [ ] Static SEO file copy: resolve relative to `SCRIPT_DIR`

### Step 4: Fix `match_sensors` to work with None megapixels (F10)
- [ ] When `megapixels is None`, still match on width+height only (skip megapixel check)
- [ ] Document that matches without megapixel verification are lower confidence

### Step 5: Add deduplication in `deduplicate_specs` (F11)
- [ ] After `remove_parens`, add a deduplication pass keyed on `(name, type, size, pitch, mpix)`
- [ ] Keep first occurrence when duplicates found

### Step 6: Fix `create_camera_key` collisions (F12)
- [ ] Include category in key: `f"{spec.name.lower().strip()}-{spec.category}-{spec.year or 'unknown'}"`
- [ ] This prevents same-name cameras in different categories from colliding

### Step 7: Fix `sensor_size_from_type` accuracy (F13)
- [ ] Always use lookup table when `typ in TYPE_SIZE`, regardless of `use_table` parameter
- [ ] Only fall back to computation when the type is not in the lookup table
- [ ] Document that computed values are approximations

### Step 8: Fix `open()` without context manager (F14)
- [ ] `pixelpitch.py` line 880: `open("sitemap.xml", ...) → Path(SCRIPT_DIR / "sitemap.xml").read_text(...)`
- [ ] Already addressed by Step 3 path fix

### Step 9: Bind debugging port to localhost (F15)
- [ ] Add `--remote-debugging-address=127.0.0.1` alongside `--remote-debugging-port=9222`

### Step 10: Error handling consistency (F17)
- [ ] Replace remaining `sys.exit` calls with appropriate exceptions
- [ ] Add docstrings documenting which exceptions can be raised

## Exit Criteria
- No `assert` used for runtime validation
- No `sys.exit` in non-CLI functions
- Script works from any CWD
- `match_sensors` returns matches for width+height even when mpix is None
- `deduplicate_specs` produces no exact duplicates
- Debugging port bound to localhost
- Offline test gate passes
