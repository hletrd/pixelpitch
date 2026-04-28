# Document Specialist — Cycle 53

**Date:** 2026-04-29
**HEAD:** `1c968dd`

## Doc-vs-code check

### `_safe_int_id` docstring vs. code (pixelpitch.py:319-337)

Docstring (line 324) says:
> Same Excel-hand-edit class as `_safe_year`.

Reality:
- `_safe_year` has BOTH `isfinite` AND a 1900-2100 range guard.
- `_safe_int_id` has ONLY the `isfinite` guard.

Doc/code mismatch (F53-DOC-01). Fix lands with F53-01: add the range
guard to `_safe_int_id`, then the docstring is accurate. Severity LOW.

## Other doc/code checks

- `parse_existing_csv` docstring — accurate.
- `_safe_year` docstring — accurate.
- `_safe_float` docstring — accurate.
- `merge_camera_data` docstring — accurate.

## CLAUDE.md / AGENTS.md cross-check

Repo has no project-level CLAUDE.md. User-global CLAUDE.md governs:
GPG-sign, conventional commits + gitmoji, no `--no-verify`, no
Co-Authored-By, fine-grained commits, `git pull --rebase` before push.

Recent commit messages are compliant. No process drift.

## Verdict

| Finding     | Severity | Confidence |
|-------------|----------|------------|
| F53-DOC-01  | LOW      | HIGH       |

Bound to the F53-01 fix.
