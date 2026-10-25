# Findings log: unified harness smoke runs (2026-10-26)

Recorded by Task 8 (byte-driven harness). Triage is not done here: each entry is what the harness reported.
Violations are reported once per (lib, rule); the STATS counts include every case. Margin `inf` means the adapter
raised an exception that the oracle does not catch (the harness records it as a violation and prints `[CRASH]`).

Command shape: `PYTHONPATH=ltn_torch:lnn/src python -m mr_unified.harness --lib <lib> --random <N>` (seed 0).
Each run wrote its findings to a separate folder (`MR_UNIFIED_FINDINGS`), which holds `input.bin` and `meta.json`.
Replay with `python -m mr_unified.replay <input.bin> --lib <lib>`. The three finding folders are in the task scratchpad:
`.../scratchpad/findings_ltn`, `.../scratchpad/findings_lnn`, `.../scratchpad/findings_all`.

## Run 1: `--lib ltn --random 20000`

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

## Run 2: `--lib lnn --random 5000`

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

## Run 3: `--lib all --random 3000`

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

## Notes for triage

- The instantiation crashes account for every `property` violation in all three runs. A substituted object name
  reaches LTN and LNN as a predicate argument and is treated as a free variable. The reference adapter accepts it.
  Whether this is an adapter or an interface question is not decided here.
- The `rewrite`, `qrewrite` and `env_rule` kinds reported no violation in any run.
