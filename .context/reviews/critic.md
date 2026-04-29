# Critic Review (Cycle 58, orchestrator cycle 11)

**Date:** 2026-04-29
**HEAD:** `aef726b`

## Critique angles

1. **Correctness:** the C57-01 fix recomputes `area` from
   `width*height`. Verified by running the gate. No new
   correctness regression.
2. **CLI surface:** `python pixelpitch.py source <name>
   --limit <n>` accepts any integer including negatives /
   zero. Slicing-based consumers silently truncate or empty.
   No error is raised, which is user-hostile.
3. **Architecture:** F32 monolith carry-over still applies.
4. **Testing:** test file growing past 2595 LOC. Same class
   as F32; deferred.

## New findings

### F58-CRIT-01: `source` CLI silently accepts `--limit` <= 0 — LOW

- Same issue as `code-reviewer.F58-CR-01`. Cross-agent
  agreement.
- Multi-perspective angle: from the CLI-user perspective,
  the command appears to succeed (exit 0, file written) but
  the output is empty / wrong. A failing exit code with a
  clear message is the standard CLI contract.
- **Severity:** LOW. **Confidence:** HIGH.

### F58-CRIT-02: `--out`/`--limit` consume their value-arg without skipping the loop counter — LOW (deferred)

- **File:** `pixelpitch.py:1393-1401`
- **Detail:** The `for i, a in enumerate(args)` loop checks
  `a == "--limit"` and `a == "--out"`, consuming `args[i+1]`,
  but does not skip ahead. A subsequent iteration sees
  `a = "5"` (the value of `--limit`); since `"5"` does not
  match either flag string, no harm. But if a user typos
  `--out --limit` (a flag-string in the value position), the
  code would set `out_dir = Path("--limit")` without
  complaint.
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** defer (typo tolerance is nice-to-have but
  the happy path works).

## Carry-over critiques (deferred)

- F32 monolith: still 1437-line `pixelpitch.py`.
- F55-CRIT-03 / F56-CRIT-02 / F57-CRIT-01: test monolith.
- F35..F40: UI carry-overs.

## Confidence summary

- 1 LOW actionable (F58-CRIT-01, overlaps F58-CR-01).
- 1 LOW deferred (F58-CRIT-02, typo tolerance).
