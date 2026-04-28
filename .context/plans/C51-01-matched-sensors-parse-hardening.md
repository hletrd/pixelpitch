# Plan C51-01: Harden `parse_existing_csv` matched_sensors parsing

**Status:** completed
**Priority:** P3 (low impact, low risk; defense-in-depth)
**Findings addressed:** F51-01, F51-02

## Repo Rules Consulted

Read in order before planning:
- `CLAUDE.md` (none at repo root; user-global only)
- `AGENTS.md` (none)
- `.context/reviews/_aggregate.md` (cycle 51)
- `.context/plans/deferred.md`
- `setup.cfg` — flake8 max-line-length=160
- Cycle GATES: `flake8`, `python3 -m tests.test_parsers_offline`

User-global rules from CLAUDE.md govern commits:
- All commits must be GPG-signed (`git commit -S`).
- Conventional Commits + gitmoji.
- Fine-grained commits, one per logical change.
- `git pull --rebase` before `git push`.

No repo rule prohibits the changes below.

## Problems

### F51-01 — `parse_existing_csv` does not strip whitespace around `matched_sensors` tokens
- File: `pixelpitch.py:373`
- Hand-edited CSV with `IMX455; IMX571` (space after delimiter, common when editing in
  Excel/text-editor) parses as `["IMX455", " IMX571"]`. Phantom whitespace-prefixed token
  round-trips through the next CSV write.
- Latent: `write_csv` produces tokens without whitespace, so this is dormant unless the
  CSV is externally edited.

### F51-02 — `parse_existing_csv` does not deduplicate `matched_sensors`
- File: `pixelpitch.py:373`
- A CSV row with `IMX455;IMX455` produces two identical entries. `match_sensors` itself
  produces unique values, so the trigger is external-edit only.

## Implementation Steps

### Step 1: Strip and dedup matched_sensors in `parse_existing_csv`
- [x] Edit `pixelpitch.py:373`. Replace
  `matched_sensors = [s for s in sensors_str.split(";") if s] if sensors_str else []`
  with logic that:
  1. Splits on `;`.
  2. Strips each element.
  3. Filters empty after strip.
  4. Deduplicates while preserving first-seen order (`dict.fromkeys`).
- [x] Confirm flake8 still passes (max-line-length 160).
- [x] Commit: `fix(csv): 🐛 strip and dedup matched_sensors in parse_existing_csv`.

### Step 2: Add tolerance test in `tests/test_parsers_offline.py`
- [x] Add a new test function `test_matched_sensors_parse_tolerance()` that:
  - Constructs a raw CSV string by hand with the matched_sensors column containing
    `IMX455; IMX571 ;IMX455 ;IMX989` (mix of leading/trailing spaces and a duplicate).
  - Calls `parse_existing_csv` directly.
  - Asserts the parsed list equals `["IMX455", "IMX571", "IMX989"]` (whitespace
    stripped, duplicate removed, order preserved).
- [x] Wire the test into `main()` so it runs under the offline test gate.
- [x] Commit: `test(csv): ✅ add matched_sensors parse-tolerance test (whitespace + dedup)`.

### Step 3: Run gates
- [x] `python3 -m flake8 . --exclude=.git,__pycache__,dist,downloaded_files,.context,.omc,templates`
  — must report 0 errors.
- [x] `python3 -m tests.test_parsers_offline` — must PASS, including the new test.

### Step 4: Push commits
- [x] `git pull --rebase && git push`.

## Exit Criteria

- [x] `pixelpitch.py:parse_existing_csv` strips whitespace and deduplicates matched_sensors
  tokens.
- [x] `tests/test_parsers_offline.py` includes a tolerance test that runs in `main()`.
- [x] Both gates pass.
- [x] All commits GPG-signed, conventional + gitmoji.

## Risk Assessment

LOW. Each change is small and orthogonal:
- Step 1: pure parser hardening; cannot regress existing data because all current
  matched_sensors tokens come from `write_csv` which emits whitespace-free unique values.
- Step 2: pure test addition; no production code change.

## Out of Scope

- F51-03 (deferred.md audit) — process-only, not a code change.
- F51-04 (cycle-50 plan bundling) — process-only, plan already complete.
- All previously deferred items (F32, F31, C22-05, F49-04, etc.) — remain deferred per
  prior cycles' rationale.
