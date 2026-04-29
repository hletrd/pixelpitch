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

## F49-02: `git pull --rebase || true` swallows rebase failures in CI — RE-OPENED in C50-01
- **File:** `.github/workflows/github-pages.yml`, line 108
- **Severity:** LOW | **Confidence:** HIGH (logic), LOW (impact)
- **Status:** Re-opened cycle 50 as F50-01 after consensus across code-reviewer / critic / verifier / tracer. Implementation plan: `C50-01-rebase-mask-and-matched-sensors-roundtrip.md`.

---

## F49-04: `merge_camera_data` re-runs `match_sensors` per existing-only camera
- **File:** `pixelpitch.py`, lines 532-547
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Linear sensor-DB scan per existing-only camera (~1000 cameras × ~200 sensors = ~200k comparisons). Acceptable at current scale; render pipeline still completes in seconds. An indexed lookup (`(width_rounded, height_rounded) -> sensors`) would cut this further but adds complexity in a file already flagged as monolithic (F32 deferred). Performance-only; not a correctness issue.
- **Re-open if:** Render time exceeds 30s on the merge step, or sensor DB grows past 5000 entries.

---

## F55-02: no boundary tolerance test for `match_sensors`
- **File:** `tests/test_parsers_offline.py` (gap)
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** `match_sensors` uses `<=` for both 2% size and 5% mpix tolerance. Boundary behavior is already implicitly correct from the operator and is exercised indirectly by C54-01 refresh tests. Adding an explicit boundary test is a nice-to-have but no bug has been observed and the boundary is unlikely to drift.
- **Re-open if:** Tolerance values become configurable, or a future refactor switches `<=` to `<`.

---

## F55-04: `merge_camera_data` mutates input `existing_specs` items in place when re-matching
- **File:** `pixelpitch.py`, lines 622-628
- **Severity:** LOW | **Confidence:** HIGH (latent contract issue)
- **Reason:** `existing_spec.matched_sensors = match_sensors(...)` writes back into the caller's list items. Today the only caller (`render_html`) does not reuse `existing_specs` after the merge call, so no observable bug. Refactoring to use `replace(existing_spec, matched_sensors=...)` is correct but introduces a new SpecDerived allocation per existing-only camera and changes a long-stable code path with no current consumer.
- **Re-open if:** A future caller reuses `existing_specs` after `merge_camera_data` and is surprised by the mutation.

---

## F55-05: `parse_existing_csv` `has_id` detection defeated by hand-edited blank-leading-cell
- **File:** `pixelpitch.py`, lines 371-372
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** A spreadsheet user inserting a blank column before id would defeat `header[0] == "id"`. The `dist/camera-data.csv` file is produced by `write_csv`, not edited by hand in normal operation. Hand-edits are explicitly out-of-scope per the current data flow (the F51/F52/F53 hand-edit hardening covers numeric-coercion edits which Excel applies silently; structural column edits are a different class).
- **Re-open if:** A user reports loss of data after editing the CSV in a spreadsheet, or the workflow starts accepting user-edited CSVs.

---

## F55-06: `_refresh_matched_sensors` helper extraction across `merge_camera_data` and `_load_per_source_csvs`
- **File:** `pixelpitch.py`, lines 615-628 and 1074-1086
- **Severity:** LOW | **Confidence:** HIGH (cleanup)
- **Reason:** After C55-01 the two paths share semantics enough to suggest a helper, but they still differ in subtle details: `merge_camera_data` only recomputes for existing-only (no new data) cameras and only when sensors_db is non-empty AND size is set; `_load_per_source_csvs` runs on every parsed row and falls back to None when size is unknown. Folding the two would obscure these differences. Refactor risk in a file already flagged monolithic (F32 deferred).
- **Re-open if:** A third refresh callsite is added, or the two paths converge to identical semantics.

---

## F56-CR-02: `merge_camera_data` size-mismatch warning has no rounding tolerance
- **File:** `pixelpitch.py`, lines 539-557
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Sub-percent rounding drift between source updates (e.g., `(23.6, 15.6)` vs `(23.7, 15.7)`) prints a warning. Adding a tolerance threshold (e.g., 0.5%) is subjective and would mask genuine source-data corrections. Current behavior is correct; the warning is informational only and goes to CI logs. No data is lost.
- **Re-open if:** CI logs become unreadable due to mismatch noise, or a real correction is hidden by tolerance choice.

---

## F56-CRIT-02: `tests/test_parsers_offline.py` is now 2456-line monolith
- **File:** `tests/test_parsers_offline.py`
- **Severity:** LOW | **Confidence:** HIGH (architectural, not a bug)
- **Reason:** Same class as deferred F32 (`pixelpitch.py` monolith) and F55-CRIT-03. Splitting into multiple test modules adds harness complexity and risk of regressions in the gate command (`python3 -m tests.test_parsers_offline`). The current file is structured by section markers and remains greppable. No correctness concern.
- **Re-open if:** Gate runtime exceeds 30s, or test discovery adopts pytest with proper module separation.

---

## F56-A-02: render_html category lists duplicated across multiple call sites
- **File:** `pixelpitch.py`, lines 1148-1164
- **Severity:** LOW | **Confidence:** HIGH (refactor opportunity)
- **Reason:** Geizhals categories and source-only categories are hardcoded in multiple spots inside `render_html`. Adding a new category requires edits in several places. Consolidating into a single source of truth is a refactor that touches the most critical render path; risk outweighs benefit at the current category count (~9). Carry-over of F55-A-02.
- **Re-open if:** A new category is added and the duplication causes a missed update, or the count grows past ~15.

---

## F56-DOC-03: `.context/plans/deferred.md` is growing; no periodic sweep
- **File:** `.context/plans/deferred.md`
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** 25+ entries from cycles 8 through 56 are documented in deferred.md. No entry has been re-opened or pruned even when the underlying rationale has shifted. A periodic sweep (annual) would be hygiene; not a correctness concern.
- **Re-open if:** A reviewer finds a deferred item that should now be actionable based on changed circumstances, or the file grows past ~50 entries.

---

## F58-04: `--out`/`--limit` consume value-arg without skipping loop counter (typo tolerance)
- **File:** `pixelpitch.py`, lines 1393-1401
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** The `for i, a in enumerate(args)` loop in `main()`'s `source` branch consumes `args[i+1]` for `--limit` and `--out`, but does not advance `i`. If the user typos `--out --limit 5`, the code sets `out_dir = Path("--limit")` without complaint. The happy path (`--limit 5 --out dist2`) works correctly. Repo policy defers this typo-tolerance because it is covered by F58-05 (argparse migration). Same root cause as the manual argparse drift.
- **Re-open if:** F58-05 argparse migration is accepted (this finding folds into that plan).

---

## F58-05: hand-coded argv parser drift between `html` and `source` branches
- **File:** `pixelpitch.py`, lines 1368-1431
- **Severity:** LOW | **Confidence:** HIGH (architectural, not a bug)
- **Reason:** `main()` rolls a hand-coded argv parser. The `html` branch uses while+counter; the `source` branch uses for+enumerate without counter advance. The two patterns drift, opening the door to F58-01 (silent `--limit -1`) and F58-04 (`--out --limit` typo). Migrating to `argparse` (stdlib, no new dependency) would consolidate the patterns, but the file is already flagged monolithic (F32 deferred). Same class as F32; refactor risk in a stable code path outweighs benefit for the current CLI surface (3 commands, 4 options).
- **Re-open if:** A new top-level CLI command is added (the third drift opportunity), or F32 monolith refactor is accepted (would naturally include the argparse migration).

---

## F58-06: no boundary tests for `_safe_year` at 1900/2100 and `_safe_int_id` at 0/1_000_000
- **File:** `tests/test_parsers_offline.py` (gap)
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Existing parse-tolerance tests cover well below / above the bounds (`"2099"`, `"2200"`, `"-1"`, `"1e308"`), but no test pins the exact boundary inclusively (`1900`, `2100`, `0`, `1_000_000`; `<=` operator). A future refactor to `<` would silently flip semantics. Indirect coverage via the existing parse-tolerance tests suffices for current scope. Same class as F55-02 (boundary tolerance defer).
- **Re-open if:** Bounds become configurable, or a future refactor switches `<=` to `<`.

---

## F59-04: `_load_per_source_csvs` "missing" log line wording is alarmingly worded on first build
- **File:** `pixelpitch.py`, line 1085
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** On a fresh `dist` directory (e.g., first CI run, after `rm -rf dist`), `_load_per_source_csvs` prints `"  source CSV missing: <name> (skipped)"` for every registered source. The message correctly signals "expected absence" but the word "missing" reads like a warning. Softer wording (e.g., "no cached CSV at <path> (skipped)") would convey the same diagnostic without the alarming framing. No behavior change; informational only. Same severity class as F58-DOC-* informational fixes deferred per repo policy.
- **Re-open if:** A user/operator reports confusion about the warning-style log line, or the wording is touched as part of a broader logging refactor.

---

## F60-CR-01: `_load_per_source_csvs` lacks per-source try/except wrapping `parse_existing_csv`
- **File:** `pixelpitch.py`, line 1132
- **Severity:** LOW | **Confidence:** LOW (theoretical at present)
- **Reason:** `parse_existing_csv(content)` is wrapped only by a per-row `try/except Exception` inside itself. If `csv.reader(io.StringIO(...))` ever raised a top-level exception on a corrupt cache file, `_load_per_source_csvs` would propagate it up, killing the build despite the docstring promise of "failure of one source must not block the build". `csv.reader` is permissive enough that no real failure mode is observed today — the contract is honored by happenstance, not structure. Defensive parity gap, not a live bug.
- **Re-open if:** A real per-source CSV load failure is observed in CI, or hand-edited cache files are accepted as a supported workflow.

---

## F60-PR-01: `match_sensors` recomputed twice for source-CSV cameras during full render
- **File:** `pixelpitch.py`, line 1139 (`_load_per_source_csvs`) and line 644 (`merge_camera_data` existing-only branch)
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Camera that exists only in a per-source CSV (not in fresh Geizhals data) gets `match_sensors` computed twice during one `render_html`: first in `_load_per_source_csvs` to refresh the cache, then in `merge_camera_data` existing-only branch. The computation is idempotent and cheap (~200 sensor comparisons), so this is wasteful-but-correct, not a bug. Same class as deferred F49-04.
- **Re-open if:** Render time exceeds 30s on the merge step, or sensor DB grows past 5000 entries.

---

## F60-SEC-01: `module.fetch(**kwargs)` uses dynamic kwargs without per-source schema validation
- **File:** `pixelpitch.py`, line 1395
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** `kwargs` is constructed from `--limit` and `GSMARENA_MAX_PAGES_PER_BRAND` env-var, then passed to `module.fetch(**kwargs)`. A future source whose `fetch()` signature lacks one of these kwargs would raise TypeError. Currently only `gsmarena` accepts `max_pages_per_brand`, and the code only sends it when `name == "gsmarena"` (line 1388) — so the safety check is manual but correct. A typed-dispatch (per-source kwargs whitelist) would be more robust but is over-engineering for the current 5-source registry. Not a security bug; a TypeError at startup is a fail-loud signal.
- **Re-open if:** Source registry expands to >10 sources, or per-source kwargs proliferate.

---

## F60-CRIT-01: `pixelpitch.py` line count is 1488 — close to F32 re-open threshold of 1500
- **File:** `pixelpitch.py` (1488 lines today, was ~990 at cycle 40)
- **Severity:** LOW | **Confidence:** HIGH (line count factual)
- **Reason:** F32 (monolith) deferred at "1500-line threshold" re-open trigger. Currently 12 lines below the threshold. Next defensive-parity hardening cycle will likely cross it. Pre-emptive flag so the orchestrator can plan a refactor track ahead of time.
- **Re-open if:** `pixelpitch.py` exceeds 1500 lines (the F32 re-open trigger).

---

## F60-A-01: no formal `fetch()` Protocol across sources
- **File:** `sources/*.py`
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Same as deferred F31. No new evidence to re-open. SOURCE_REGISTRY provides the implicit contract; adding a typing.Protocol would improve type safety but is not needed for the 5 current sources.
- **Re-open if:** New sources are frequently added and need standardized interface (same trigger as F31).

---

## F60-TE-01: no test pins `_load_per_source_csvs` behavior when `parse_existing_csv` raises
- **File:** `tests/test_parsers_offline.py` (gap)
- **Severity:** LOW | **Confidence:** LOW
- **Reason:** Pairs with F60-CR-01. Adding a regression test would require constructing pathological CSV input (e.g. binary garbage). Limited value since `csv.reader` is permissive enough that `parse_existing_csv` is unlikely to raise at the top level.
- **Re-open if:** F60-CR-01 is re-opened (paired finding).

---

## F60-D-01: `derive_spec` cleans size/area to None but leaves `spec.size` unchanged — Spec/SpecDerived asymmetry not explicit in docstring
- **File:** `pixelpitch.py`, line 902-904
- **Severity:** LOW | **Confidence:** MEDIUM (DOC-only)
- **Reason:** When input `spec.size` has `(inf, 24)`, `derive_spec` sets `derived.size = None` and `derived.area = None`, but `spec.size` (the underlying Spec field) is not modified. Spec is the raw input, SpecDerived is the cleaned output. A downstream caller that reads `spec.size` directly (rather than `derived.size`) would still see `(inf, 24)`. No such caller exists today — template, write_csv, and JSON-LD all read `derived.size`. Documenting the asymmetry more clearly in `derive_spec`'s docstring would aid future maintainers but is not a live bug.
- **Re-open if:** A new caller reads `spec.size` directly and surfaces the inf/nan values, or a docstring overhaul is scheduled.

---

## F60-DOC-01: repeats deferred F59-04 "missing" log wording
- **File:** `pixelpitch.py`, line 1125
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Identical to deferred F59-04. No change in disposition.
- **Re-open if:** Same as F59-04.

---

## F61-CR-01: CSV `matched_sensors` column cannot distinguish None vs [] — round-trip lossy
- **File:** `pixelpitch.py`, lines 462-466 (parse_existing_csv) and 1069-1081 (write_csv)
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** `derive_spec` documents a tri-valued sentinel for `matched_sensors` (None = "not checked", [] = "checked, found nothing", non-empty list = matches). The CSV format conflates the first two: `write_csv` emits `""` for both None and [], and `parse_existing_csv` reads `""` back as `[]`. After round-trip, the "not checked" sentinel is lost. Practical impact is nil — downstream consumers (template, write_csv) treat None and [] identically, and existing tests pin `[]` as the canonical post-parse value (test_parsers_offline.py:691, 701). Adding a third in-band token (e.g. `"-"` for None) would change a stable on-disk format and require a parallel migration; not justified for a doc-only asymmetry. Same class as F60-D-01.
- **Re-open if:** A future consumer needs to distinguish "never checked" from "checked, empty" after CSV round-trip, or the on-disk format is being revised for another reason.

---

## F61-TE-01: no test pins matched_sensors None-vs-[] CSV round-trip asymmetry
- **File:** `tests/test_parsers_offline.py` (gap)
- **Severity:** LOW | **Confidence:** LOW
- **Reason:** Pairs with F61-CR-01. Tests today (line 691, 701) pin `[]` as the canonical post-parse value for empty-cell or no-sensors-column rows, but no test explicitly pins the `derive_spec`-produced `None` -> `write_csv` -> `parse_existing_csv` -> `[]` asymmetry. The asymmetry is by design; documenting it as a regression test would help future maintainers but is not a behavior-change requirement.
- **Re-open if:** F61-CR-01 is re-opened (paired finding).

---

## F61-CRIT-01: `pixelpitch.py` line count is 1488 — close to F32 re-open threshold of 1500 (cycle 61)
- **File:** `pixelpitch.py` (1488 lines)
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Same finding as F60-CRIT-01 — no change since cycle 60. Currently 12 lines below the F32 re-open threshold. Pre-emptive flag so the orchestrator can plan a refactor track ahead of time.
- **Re-open if:** `pixelpitch.py` exceeds 1500 lines (the F32 re-open trigger).

---

## F61-DOC-01: repeats deferred F59-04 / F60-DOC-01 "missing" log wording
- **File:** `pixelpitch.py`, line 1125
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Identical to deferred F59-04 / F60-DOC-01. No change in disposition.
- **Re-open if:** Same as F59-04.

---

## F62-CRIT-01: `pixelpitch.py` line count is 1488 — close to F32 re-open threshold of 1500 (cycle 62)
- **File:** `pixelpitch.py` (1488 lines, unchanged from cycle 60-61)
- **Severity:** LOW | **Confidence:** HIGH
- **Reason:** Same finding as F60-CRIT-01 / F61-CRIT-01 — no change. Currently 12 lines below the F32 re-open threshold. Pre-emptive flag so the orchestrator can plan a refactor track ahead of time.
- **Re-open if:** `pixelpitch.py` exceeds 1500 lines (the F32 re-open trigger).

---

## F62-DOC-01: repeats deferred F59-04 / F60-DOC-01 / F61-DOC-01 "missing" log wording
- **File:** `pixelpitch.py`, line 1125
- **Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Identical to deferred F59-04 / F60-DOC-01 / F61-DOC-01. No change in disposition.
- **Re-open if:** Same as F59-04.

---
