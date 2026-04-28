# Plan C50-01: Drop CI Rebase Mask and Add matched_sensors Round-Trip Defense

**Status:** completed
**Priority:** P2 (low impact, low risk; defense-in-depth)
**Findings addressed:** F50-01, F50-03, F50-04

## Repo Rules Consulted

Read in order before planning:
- `CLAUDE.md` (none at repo root; user-global only)
- `AGENTS.md` (none)
- `.context/reviews/_aggregate.md` (cycle 50)
- `.context/plans/deferred.md` — F49-02 was deferred LOW; F50-01 re-flags as consensus
- `setup.cfg` — flake8 max-line-length=160
- Cycle GATES: `flake8`, `python -m tests.test_parsers_offline`

No repo rule prohibits the changes below. Per the user-global rules:
- All commits must be GPG-signed.
- Conventional Commits + gitmoji.
- Fine-grained commits, one per logical change.
- `git pull --rebase` before `git push`.

## Problems

### F50-01 — `git pull --rebase || true` masks rebase failures
- File: `.github/workflows/github-pages.yml:108`
- The `|| true` suppresses rebase conflicts. Subsequent `git push` fails noisily on non-fast-forward, but the audit trail in workflow logs is muddled.

### F50-03 — matched_sensors round-trip uses unescaped `;` delimiter
- File: `pixelpitch.py:373` (split) and `pixelpitch.py:920-922` (join)
- No invariant declared or enforced that sensor names contain no `;`. Currently dormant.

### F50-04 — No round-trip test for matched_sensors
- File: `tests/test_parsers_offline.py`
- write_csv → parse_existing_csv parity is not asserted as a unit test.

## Implementation Steps

### Step 1: Drop `|| true` from CI rebase
- [x] Edit `.github/workflows/github-pages.yml:108`: replace `git pull --rebase || true` with `git pull --rebase`.
- [x] Confirm YAML still parses (`python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/github-pages.yml').read_text())"` returns no error).
- [x] Commit: `ci(workflow): 🐛 surface rebase failures instead of masking with || true`.

### Step 2: Add invariant assertion in write_csv for matched_sensors
- [x] In `pixelpitch.py:write_csv`, after the loop iterates each `derived`, assert that no element of `derived.matched_sensors` (when not None) contains `;`. Prefer a soft warning print (consistent with the codebase's style) rather than a hard assertion that would crash the entire render. The contract is: "matched_sensors elements must not contain `;`. Currently always true. If it ever changes, switch to a different delimiter or escape."
- [x] Form: when `derived.matched_sensors` is truthy and any element contains `;`, log a warning and skip the offending element from the joined string. This is a defensive measure: the broken element is dropped rather than fragmented.
- [x] Commit: `fix(csv): 🛡️ guard write_csv against matched_sensors entries containing the ';' delimiter`.

### Step 3: Add round-trip test for matched_sensors
- [x] In `tests/test_parsers_offline.py`, add a new test function `test_matched_sensors_roundtrip()` that:
  - Constructs a `SpecDerived` with `matched_sensors=["IMX455", "IMX571", "IMX989"]` (multi-element).
  - Writes via `write_csv` to a `tempfile.NamedTemporaryFile`.
  - Reads back with `parse_existing_csv`.
  - Asserts the parsed `matched_sensors` equals the original list.
- [x] Add the test to the `main()` runner so the gate covers it.
- [x] Commit: `test(csv): ✅ add matched_sensors write_csv → parse_existing_csv round-trip test`.

### Step 4: Run gates
- [x] `python3 -m flake8 . --exclude=.git,__pycache__,dist,downloaded_files,.context,.omc,templates` — must report 0 errors.
- [x] `python3 -m tests.test_parsers_offline` — must PASS, including the new round-trip test.

### Step 5: Push commits
- [x] `git pull --rebase && git push`.

## Exit Criteria

- [x] `.github/workflows/github-pages.yml` no longer contains `|| true` after `git pull --rebase`.
- [x] `pixelpitch.py:write_csv` defends against matched_sensors entries containing `;`.
- [x] `tests/test_parsers_offline.py` includes a matched_sensors round-trip test that runs in `main()`.
- [x] Both gates pass.
- [x] All commits GPG-signed, conventional + gitmoji.

## Risk Assessment

LOW. Each change is small and orthogonal:
- Step 1: pure CI workflow edit.
- Step 2: defensive print + filter in write_csv; cannot regress existing data because no current sensor name contains `;`.
- Step 3: pure test addition; no production code change.

## Out of Scope

- F32 / C22-05 / F31 architectural cleanups — remain deferred.
- F49-04 sensor-DB index — performance polish; remain deferred.
- F50-02 (review-file freshness) — addressed in cycle 50 by refreshing all per-agent files alongside the aggregate; no separate plan needed.
