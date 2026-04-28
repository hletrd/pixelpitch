# Deferred Findings

These findings from the review are explicitly deferred. Each entry records:
- The original finding ID and description
- File+line citation
- Original severity/confidence (NOT downgraded)
- Concrete reason for deferral
- Exit criterion that would re-open it

---

## Items removed from deferred (now completed in Plans 12-13)
- C3-09 + S4-02 / C8-01: CDN resources without SRI hashes — COMPLETED (commit 447ee5a)
- C6-05 / C8-03: LD+JSON `temporalCoverage` starts at "2025" — COMPLETED (commit 447ee5a)
- C8-02: TYPE_SIZE incomplete, 25 phone sensor formats missing — COMPLETED (commit 2fba619)
- C8-04: No test for sensor_size_from_type with phone formats — COMPLETED (commit f2a863f)

---

## F18: `sorted_by` uses `-1` as default for numeric sort keys
- **File:** `pixelpitch.py`, lines 581–585
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** The default sort order is descending by pitch, so `-1` correctly pushes unknowns to the end. The issue only manifests in ascending sort, which is not exposed in the UI. Low user impact.
- **Re-open if:** Ascending sort is added to the UI, or users report unexpected ordering.

---

## F20: No `pyproject.toml` — no proper Python packaging
- **File:** Project root (missing file)
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** The project is a script-based static site generator, not a distributable package. Adding `pyproject.toml` is a quality-of-life improvement for developers but not needed for the site to function.
- **Re-open if:** The project needs to be pip-installable or published to PyPI.

---

## F21: No logging framework — all output via `print()`
- **File:** All source files
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** The script runs in CI with simple stdout capture. Adding `logging` would improve configurability but is not needed for the current use case. The existing `print()` statements are sufficient for CI logs.
- **Re-open if:** The project needs log levels (debug/info/warning), or runs as a long-lived service.

---

## F23: Unused `GSMARENA_MAX_PAGES_PER_BRAND` env var in CI
- **File:** `.github/workflows/github-pages.yml`, line 74
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** The env var is set but not used by the CLI. It's dead code in the CI config but doesn't cause any failure. Low priority cleanup.
- **Re-open if:** GSMArena fetch needs pagination control from CI.

---

## F24: No rate-limit handling in `http_get`
- **File:** `sources/__init__.py`, lines 44–57
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** The current retry logic handles connection errors. Adding 429 detection would be more robust, but the scrapers already use `time.sleep()` between requests and the sites haven't rate-limited the current UA. Low practical risk.
- **Re-open if:** A source starts returning 429 errors.

---

## F26: Global mutable `_env` for Jinja2
- **File:** `pixelpitch.py`, lines 616–630
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Not thread-safe, but the script is single-threaded. The lazy-init pattern works for the current use case. Refactoring would add complexity without benefit.
- **Re-open if:** The script needs concurrent template rendering.

---

## F27: `merge_camera_data` always re-matches sensors on preserved cameras
- **File:** `pixelpitch.py`, lines 331–338
- **Severity:** LOW | **Confidence:** LOW
- **Reason:** Re-matching ensures sensor matches stay current with the sensor database. Overwriting is actually desirable if the DB improves. The finding notes this as low confidence.
- **Re-open if:** Sensor matching becomes expensive or produces incorrect results.

---

## F30: CineD scraper hard limit of 200 cameras
- **File:** `sources/cined.py`, line 91
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** 200 cameras is more than enough for the current CineD database. Making it configurable would be nice but is not urgent.
- **Re-open if:** The CineD database exceeds 200 cameras.

---

## F31: No source Protocol/base class
- **File:** `sources/` modules
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Adding a `typing.Protocol` would improve type safety but is not needed for the 6 current source modules. The `SOURCE_REGISTRY` already serves as a contract.
- **Re-open if:** New sources are frequently added and need standardized interface.

---

## F32: `pixelpitch.py` is a 990-line monolith
- **File:** `pixelpitch.py`
- **Severity:** LOW | **Confidence:** HIGH (architectural, not a bug)
- **Reason:** Refactoring into multiple modules is a significant undertaking with high risk of introducing bugs. The current structure is understandable for a single-developer project. This is an architectural improvement, not a correctness fix.
- **Re-open if:** The file grows beyond 1500 lines or new developers join the project.

---

## F34: `importlib.import_module` with user-controllable input
- **File:** `pixelpitch.py`, lines 900–921
- **Severity:** LOW | **Confidence:** HIGH (mitigated by whitelist)
- **Reason:** The `SOURCE_REGISTRY` whitelist prevents arbitrary module import. The risk is theoretical — a developer would need to add a malicious entry to the hardcoded dictionary.
- **Re-open if:** Dynamic source registration is added.

---

## F35: Box plot hardcoded dimensions, not responsive
- **File:** `templates/pixelpitch.html`, lines 339–341
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** The D3 box plot has hardcoded height but responsive width (based on container). Adding a resize handler would improve UX on window resize, but this is a low-priority enhancement.
- **Re-open if:** Users report layout issues on resize.

---

## F36: No skip-to-content link for accessibility
- **File:** `templates/index.html`
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Adding a skip link would improve WCAG 2.2 compliance. However, the navbar already collapses on mobile, and the main audience is technical users who are likely mouse-oriented.
- **Re-open if:** Accessibility audit is required.

---

## F37: Filter dropdown doesn't show current state
- **File:** `templates/pixelpitch.html`, lines 164–173
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Visual enhancement only. The filter works correctly; it just doesn't show which option is active.
- **Re-open if:** Users report confusion about current filter state.

---

## F38: No loading indicator or pagination for large datasets
- **File:** `templates/pixelpitch.html`
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** The "All Cameras" page can be large, but the table is sortable and filterable. Adding pagination is a significant UX change. Current performance is acceptable.
- **Re-open if:** Page load time exceeds 5 seconds.

---

## F39: Navbar has 9 items, may be unwieldy on mobile
- **File:** `templates/index.html`, lines 198–248
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Bootstrap's hamburger menu handles mobile layout. The expanded list works fine in the collapse. Grouping into dropdowns would be nice but is not needed.
- **Re-open if:** Mobile users report navigation difficulty.

---

## F40: `openmvg.fetch` fetches entire CSV on every call with no caching
- **File:** `sources/openmvg.py`
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** The CSV is fetched once per CI run. Adding HTTP conditional requests would save bandwidth but the CSV is small (~500KB) and the CI cache is not a concern.
- **Re-open if:** The CSV grows significantly or fetch frequency increases.

---

## C9-07: jQuery SRI uses sha256 while all other CDN resources use sha384
- **File:** `templates/index.html`, line 287
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Both sha256 and sha384 are valid SRI algorithms. jQuery's CDN provides sha256 hashes by default. Recomputing as sha384 would be a style consistency improvement but provides no security benefit. The hash is still valid and verified against live CDN content.
- **Re-open if:** Project establishes a strict SRI algorithm policy requiring sha384-only.

---

## F41: `_body_category` heuristics can misclassify fixed-lens full-frame cameras
- **File:** `sources/apotelyt.py`, `sources/imaging_resource.py`
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Fixed-lens full-frame cameras (Leica Q3, Sony RX1) are rare. The heuristic correctly classifies 99%+ of cameras. Adding name-based checks would add complexity for marginal gain.
- **Re-open if:** Misclassified cameras are reported.

---

## C10-04: Medium format cameras classified as "mirrorless" in _body_category
- **File:** `sources/apotelyt.py`, line 89; `sources/imaging_resource.py`, line 126
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** The site has no "medium format" category page. Adding one would require a new page, nav entry, template, and render logic — a significant design decision. "mirrorless" is the closest approximation for modern medium-format digital cameras (Hasselblad, Fujifilm GFX). The current classification is intentional given the site's category structure.
- **Re-open if:** Medium-format camera count grows significantly or users report confusion.

---

## C10-07: HTTP redirect chain not validated — SSRF risk
- **File:** `sources/__init__.py`, lines 48-61
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** `urllib.request.urlopen` follows redirects without limit or same-origin validation. All source URLs are hardcoded trusted domains. The risk requires DNS poisoning or source compromise. Adding redirect validation would add complexity to the simple `http_get` helper for a theoretical threat in a CI-only tool.
- **Re-open if:** A source is compromised via redirect hijacking, or the fetcher runs in a cloud environment with sensitive metadata endpoints.

---

## C10-08: Remote debugging port on macOS browser
- **File:** `pixelpitch.py`, lines 383-384
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** The `--remote-debugging-port=9222` is only enabled on macOS with Google Chrome installed, and is bound to 127.0.0.1. This only runs on developer machines during data fetch. In CI (Linux), headless mode is used without the debugging port. The risk requires local access on the developer's machine.
- **Re-open if:** The browser session is used in a shared/multi-user environment.

---

## C11-05: No integration test for `render_html` output
- **File:** `pixelpitch.py`, lines 738-920; `tests/test_parsers_offline.py`
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Adding an integration test for the full rendering pipeline is a significant test infrastructure investment. The existing unit tests and the about.html rendering test provide reasonable coverage. The pixelpitch.html template rendering is exercised indirectly through the about.html test which validates template inheritance.
- **Re-open if:** A template rendering regression goes undetected by gate tests.

---

## C11-08: Scatter plot year axis label overlap with many years
- **File:** `templates/pixelpitch.html`, lines 376-394
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** The scatter plot is a secondary visualization feature. Label overlap only occurs with 20+ years of data. The current chart is readable for the typical 10-15 year range. Adding label rotation is a visual enhancement, not a correctness fix.
- **Re-open if:** Users report readability issues with the scatter plot axis.

---

## C11-09: CSV parser positional indexing — schema change fragility
- **File:** `pixelpitch.py`, lines 230-310
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Switching to DictReader is a significant refactor with risk of introducing bugs. The current code works correctly for the known schemas (has_id and no_id). `write_csv` and `parse_existing_csv` are maintained together, so schema drift is unlikely.
- **Re-open if:** The CSV schema changes and the parser silently breaks.

---

## C22-05: Field preservation logic is ad-hoc and fragile
- **File:** `pixelpitch.py`, `merge_camera_data()`, lines 408-440
- **Severity:** LOW | **Confidence:** HIGH (architectural, not a bug)
- **Reason:** Extracting a generic `_preserve_none_fields()` helper is a refactoring improvement, not a correctness fix. The current code works correctly after the C22-01 fix. Refactoring carries risk of introducing new bugs in the merge logic (the most critical path in the application). The benefit (declarative field list) does not outweigh the risk at this time.
- **Re-open if:** More fields are added to Spec/SpecDerived and the `if` chain grows beyond 12 statements, or another insertion bug occurs.

---
