# Findings log: unified harness smoke runs (2026-10-26)

Recorded by Task 8 (byte-driven harness). Triage is not done here: each entry is what the harness reported.
Violations are reported once per (lib, rule); the STATS counts include every case.

Round 2 (after fix round 1, commit 4ad7b4b) is the current result. Round 1 is kept below for history; its
instantiation findings were exceptions that round 1 fix A removed, and its crash label has changed (see round 2).


Command shape: `PYTHONPATH=ltn_torch:lnn/src python -m mr_unified.harness --lib <lib> --random <N>` (seed 0).
Each run wrote its findings to a separate folder (`MR_UNIFIED_FINDINGS`), which holds `input.bin` and `meta.json`.
Replay with `python -m mr_unified.replay <input.bin> --lib <lib>`. The three finding folders are in the task scratchpad:
`.../scratchpad/findings_ltn`, `.../scratchpad/findings_lnn`, `.../scratchpad/findings_all`.

## Round 1 (superseded)

### Run 1: `--lib ltn --random 20000`

Exit code 1.

| kind        | run  | skipped | violations |
|-------------|------|---------|------------|
| rewrite     | 3410 | 1568    | 0          |
| qrewrite    | 5067 | 9       | 0          |
| env_rule    | 4877 | 0       | 0          |
| property    | 5069 | 0       | 111        |
| cross       | 0    | 0       | 0          |

Violations (one per (lib, rule)):

1. `ltn instantiation`, margin inf, got `(nan, nan)`
   - formula: `forall y. not(imp(0, R(y, y)))`
   - minimised: `forall y. not(imp(0, R(y, y)))` (margin inf)
   - exception: `ValueError("formula has free variables: ['o0']")`, raised in `LTNAdapter.evaluate`
     while `instantiation_margin` evaluated the body with the object `o0` substituted for `y`.
   - The other `property` violations in this run are all further `instantiation` crashes; the CRASH lines
     printed were for `o0`, `o1` and `o2`. No other property law reported a violation.

### Run 2: `--lib lnn --random 5000`

Exit code 1.

| kind        | run | skipped | violations |
|-------------|-----|---------|------------|
| rewrite     | 265 | 960     | 0          |
| qrewrite    | 50  | 1205    | 0          |
| env_rule    | 577 | 649     | 0          |
| property    | 393 | 901     | 5          |
| cross       | 0   | 0       | 0          |

Violations:

1. `lnn instantiation`, margin inf, got `(nan, nan)`
   - formula: `exists x. Q(x)`
   - minimised: `exists x. Q(x)` (margin inf)
   - exception: `ValueError("formula has free variables: ('o2',)")` (and `('o0',)`), raised in `LNNAdapter._atom`
     for the instance `Q(o2)` created by substituting the object into the predicate argument.

### Run 3: `--lib all --random 3000`

Exit code 1.

| kind        | run  | skipped | violations |
|-------------|------|---------|------------|
| rewrite     | 927  | 481     | 0          |
| qrewrite    | 572  | 556     | 0          |
| env_rule    | 1522 | 329     | 0          |
| property    | 1357 | 398     | 11         |
| cross       | 277  | 317     | 1          |

Violations:

1. `ltn instantiation`, margin inf, got `(nan, nan)`
   - formula: `exists x. Q(x)`; minimised: `exists x. Q(x)`
   - same `ValueError` as run 1 (free variable `o0`, `o1`, `o2`).
2. `lnn instantiation`, margin inf, got `(nan, nan)`
   - formula: `exists x. Q(x)`; minimised: `exists x. Q(x)`
   - same exception as run 2.
3. `ltn vs lnn cross_check`, margin 0.05859375, got `(0.05859375, 0.0)`
   - formula: `forall y. Q(y)`; minimised: `forall y. Q(y)` (margin 0.05859375)
   - env: atoms `p0 = 0.11328125`, `p1 = 0.0`; domain `o0, o1`; `Q(o0) = 0.05859375`, `Q(o1) = 0.14453125`.
   - ltn computes min over the domain (0.05859375). lnn computes the strong fold (max(a + b - 1, 0) = 0).
     LNN declares fold quantifiers and LTN declares min, so this is a disagreement in the quantifier semantics the
     two adapters declare, and it was reported as is.

### Round 1 notes

- The instantiation crashes account for every `property` violation in all three runs. A substituted object name
  reaches LTN and LNN as a predicate argument and is treated as a free variable. The reference adapter accepts it.
  Whether this is an adapter or an interface question is not decided here.
- The `rewrite`, `qrewrite` and `env_rule` kinds reported no violation in any run.

## Round 2: smoke runs after fix round 1

Command shape: `PYTHONPATH=ltn_torch:lnn/src python -m mr_unified.harness --lib <lib> --random <N>` (seed 0).
Findings folders: `.../scratchpad/r2_findings_ltn`, `.../scratchpad/r2_findings_lnn`, `.../scratchpad/r2_findings_all`.
A margin of `inf` is an adapter exception (recorded as `<rule>:crash`). Crash findings carry the same formula as
their `minimised` field (crashes are not minimised).

### Run 1: `--lib ltn --random 20000`

Exit code 1.

| kind        | run  | skipped | violations |
|-------------|------|---------|------------|
| rewrite     | 3410 | 1568    | 0          |
| qrewrite    | 5067 | 9       | 0          |
| env_rule    | 4877 | 0       | 0          |
| property    | 5069 | 0       | 3          |
| cross       | 0    | 0       | 0          |

Violations:

1. `ltn instantiation:crash`, margin inf, got `(nan, nan)`
   - formula: `forall x. or_s(exists y. R(x, y), exists y. R(x, x))`
   - minimised: `forall x. or_s(exists y. R(x, y), exists y. R(x, x))` (margin inf; crashes are not minimised)
   - exception: `RuntimeError("shape '[3]' is invalid for input of size 9")`
   - Other printed exceptions: `RuntimeError("shape '[2]' is invalid for input of size 4")` (the third violation is
     this second crash; only the first crash of a (lib, rule) is written as a finding).

### Run 2: `--lib lnn --random 5000`

Exit code 0. No violations.

| kind        | run | skipped | violations |
|-------------|-----|---------|------------|
| rewrite     | 265 | 960     | 0          |
| qrewrite    | 50  | 1205    | 0          |
| env_rule    | 577 | 649     | 0          |
| property    | 393 | 901     | 0          |
| cross       | 0   | 0       | 0          |

### Run 3: `--lib all --random 3000`

Exit code 0. No violations.

| kind        | run  | skipped | violations |
|-------------|------|---------|------------|
| rewrite     | 927  | 481     | 0          |
| qrewrite    | 572  | 556     | 0          |
| env_rule    | 1522 | 329     | 0          |
| property    | 1357 | 398     | 0          |
| cross       | 312  | 282     | 0          |

The cross kind is now generated on quantifier-free formulas only, since the three adapters declare different
quantifier semantics (ltn and ref min/max, lnn fold). The round 1 cross disagreement (`ltn vs lnn cross_check`,
`forall y. Q(y)`) is therefore not generated in round 2.

## Round 3: LTN smoke run after the mixed-argument fix (commit 36d8861)

Command: `PYTHONPATH=ltn_torch:lnn/src python -m mr_unified.harness --lib ltn --random 20000` (seed 0), with
`MR_UNIFIED_FINDINGS` set to a fresh scratch folder. Exit code 0. Output verbatim:

```
rewrite     run=3410 skipped=1568 violations=0
qrewrite    run=5067 skipped=9 violations=0
env_rule    run=4877 skipped=0 violations=0
property    run=5069 skipped=0 violations=0
cross       run=0 skipped=0 violations=0
```

No findings were written. The round 2 crash `ltn instantiation:crash` (`forall x. or_s(exists y. R(x, y), exists y. R(x, x))`)
no longer occurs, and its saved input replays with no violation.
