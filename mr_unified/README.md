# mr_unified: one set of metamorphic tests for LTN, LNN and PSL

This package tests three neuro-symbolic libraries with the same set of checks. The libraries are LTNtorch (LTN),
IBM's LNN, and PSL, the reasoning core that NeuPSL is built on. All three compute "how true is this formula" with
fuzzy logic (truth values between 0 and 1). The checks are *metamorphic relations*: rules that two equal formulas must
satisfy, such as "a implies b has the same truth value as not b implies not a". Each library gets only a small adapter
that turns a formula into that library's objects. The rules come from Lukasiewicz logic, the fuzzy logic that all three
share, so the same rules apply to every library and no reference implementation is needed. The aim is to catch wrong
results that a library's own unit tests and simple closed-form checks miss.

## Files

| File | What it is |
|---|---|
| `formula.py` | The formula tree (`Node`), the truth-value environment (`Env`) and `ref_eval`, a plain float64 reference evaluator |
| `rewrites.py` | The rewrite rules (A1-A10, B2-B5, B1 and B7), the random formula generator and the rule that applies a rewrite at a random spot |
| `properties.py` | Checks that are not rewrites: monotonicity (A11), 0/1 behaviour (A12), piecewise slope (A13), separation (S1, S2), quantifier instantiation (B6) and domain split (B8) |
| `oracle.py` | The tolerance, the checks that compare two sides, the cross-library comparison and the minimiser that shrinks a failing formula |
| `cases.py` | Turns fuzz bytes into one test case (a rewrite, quantifier rewrite, property, environment rule or cross-library check) |
| `harness.py` | The driver: runs cases on the chosen libraries, counts results and saves each new failure |
| `replay.py` | Re-runs one saved failure (`input.bin`) and prints what it reproduces |
| `findings_io.py` | Writes each failure to `<lib>/<rule>/input.bin` and `meta.json` |
| `guidance.py` | Coverage hook for libFuzzer: bigger gaps count as new coverage, so those inputs are kept |
| `ltn_extras.py` | LTN-only checks from van Krieken et al.: gradients, duality, Godel invariance and mean relations |
| `mutation_study.py` | Plants one bug at a time in LTN or LNN and reports which oracle catches it |
| `adapters/base.py` | The adapter interface and the `Unsupported` error (a formula the library cannot express, not a wrong answer) |
| `adapters/ltn_adapter.py` | LTN adapter: Lukasiewicz connectives, `forall` as min, `exists` as max |
| `adapters/lnn_adapter.py` | LNN adapter: strong connectives and closed-world quantifiers over unary predicates |
| `adapters/psl_adapter.py` | PSL adapter: the PSL core through `pslpython`, solved by ADMM (an iterative solver) |
| `adapters/ref_adapter.py` | Reference adapter: `ref_eval` in float64, the trusted side in comparisons |
| `tests/` | pytest tests, one file per module. The PSL tests run only under `venv_psl` |

## The relations

Every rule holds in Lukasiewicz logic. "Side-condition" says when a rule is used. Rules marked "to confirm" come from the
design spec, which says those statements still need checking against the book text.

| ID | Law (in plain words) | Source | Side-condition |
|---|---|---|---|
| A1 | `a and b` = `b and a`, same for `or`, `min`, `max` | MV-algebra axioms | operator is commutative |
| A2 | `(a and b) and c` = `a and (b and c)`, same for `or` | MV-algebra axioms | same operator nested |
| A3 | `not not a` = `a` | MV-algebra axiom | none |
| A4 | `a and b` = `not(not a or not b)` (De Morgan) | De Morgan | none |
| A5 | `a -> b` = `not a or b` | Lukasiewicz implication | none |
| A6 | `a -> b` = `not b -> not a` (contraposition) | not named in spec | none |
| A7 | `max(a, b)` = `(a -> b) -> b` | MV-algebra | none |
| A8 | `(a -> b) or (b -> a)` is always 1 (prelinearity) | BL-algebra (to confirm) | none |
| A9 | `min(a, b)` = `a and (a -> b)` (divisibility) | BL-algebra (to confirm) | none |
| A10 | `a and not b` = `not(a -> b)` (negated implication, a building block of the MV distance; the full distance `\|a-b\|` is not in the grammar) | MV distance function | none |
| A11 | Raising an input whose effect is monotone never lowers the output the wrong way | residuated lattice | atom with a known sign |
| A12 | With only 0 and 1 as inputs, the output equals the classical truth table | all | none |
| A13 | The output is piecewise linear in each atom, with integer slopes | Handbook Ch. IX (to confirm) | none |
| S1 | `a or not a` is not always 1 (must fail at a = 0.5) | separating check | expected to fail |
| S2 | `a and a` is not always `a` (must fail at a = 0.5) | separating check | expected to fail |
| B1 | Permuting the domain objects leaves `forall` and `exists` unchanged | first-order logic | none |
| B2 | Renaming a bound variable changes nothing | first-order logic | none |
| B3 | `forall x forall y P` = `forall y forall x P` (nested quantifiers can swap) | first-order logic | none |
| B4 | `not forall x P` = `exists x not P` (and the reverse) | duality | `exists` is the dual of `forall` |
| B5 | `forall x P` = `P` when `x` does not occur in `P` | first-order logic | aggregators are min and max |
| B6 | `forall x P <= P(c) <= exists x P` for any object `c` | first-order logic | aggregators are min/max or fold |
| B7 | On a one-object domain, `forall` and `exists` both equal `P(c)` | first-order logic | domain has one object |
| B8 | A quantifier over `L + R` equals the combination of its two halves | aggregator property (van Krieken) | aggregator is a T-based fold |

B5 is deliberately strict. It shows which aggregator a library really uses, so a library that declares one aggregator and
computes another gets caught. For this reason the side-conditions come from each adapter's `declared_quantifiers()`.

## How it works

1. **Generate a formula.** A random generator builds a formula tree over a few atoms (`p0`, `p1`, predicates `R(x, y)`),
   constants, and, for quantifier cases, `forall` and `exists` over a small domain (`o0`, `o1`, `o2`).
2. **Rewrite it at a random position.** A rule from the table is applied at one random subformula. Two formulas come out:
   the original `phi` and the rewritten `phi'`. Both are equal in Lukasiewicz logic, so their truth values must match.
3. **Evaluate both on each library.** The adapter builds the library's objects for `phi` and for `phi'`, with the same
   atom values, and returns one number for each.
4. **Compare within a derived tolerance.** The allowed gap is `(size(phi) + size(phi')) * eps`, where `eps` is the
   machine precision of the library's arithmetic (about 2e-16 for float64, 1e-7 for float32). The sizes count nodes,
   because each node may add one rounding error. PSL solves by iteration, so its `eps` is set to 1e-5. A gap above the
   tolerance is a violation.
5. **Shrink and save.** The minimiser replaces parts of the formula while the violation remains, and the smallest
   failing formula is written to `findings/<lib>/<rule>/input.bin` with `meta.json`. `replay.py` re-runs it later.

If an adapter raises `Unsupported` for a formula, the case is counted as skipped, not as passed.

## Quick start

### Environment (once)

- Python 3.11 or newer, because LNN needs 3.11.
- LTNtorch is cloned at commit `d1bd981` by the repo's `setup.sh`, into `ltn_torch/`. This folder is git-ignored.
- LNN is fetched at commit `c88f513` by `lnn/setup_lnn.sh` into `lnn/src/`. Run it from `lnn/` with the venv active.
  Importing LNN writes `LNN_INFO.log` into the current folder. That file is git-ignored.
- Activate the venv (`. venv/bin/activate`, torch 2.14.1 used) and set the import paths:
  `export PYTHONPATH=ltn_torch:lnn/src`. Run every command from the repo root.
- PSL needs its own venv, `venv_psl`, with `pslpython` 2.4.1 and Java 11 installed. Only the PSL commands use it.
- Findings go to `mr_unified/findings/` by default. Set `MR_UNIFIED_FINDINGS=/some/folder` to change that.

### Commands

```bash
# All tests (PSL tests are skipped outside venv_psl)
PYTHONPATH=ltn_torch:lnn/src python -m pytest mr_unified/tests -q

# Random fuzz run on LTN, 2000 cases
python -m mr_unified.harness --lib ltn --random 2000

# Replay one saved failure on LTN (use the lib recorded in its meta.json)
python -m mr_unified.replay <input.bin> --lib ltn

# Mutation study on LTN and LNN, 2000 cases per mutant
python -m mr_unified.mutation_study --lib all --cases 2000

# PSL: tests and a fuzz run, from venv_psl
venv_psl/bin/python -m pytest mr_unified/tests/test_psl_adapter.py mr_unified/tests/test_psl_c_group.py -q
venv_psl/bin/python -m mr_unified.harness --lib psl --random 300
```

`--lib` accepts `ltn`, `lnn`, `ref`, `psl` or `all`. `all` means ltn, lnn and ref. PSL has to be asked for by name.

## What the results say so far

Numbers come from `docs/superpowers/plans/2026-10-26-mutation-table.md` and `docs/superpowers/plans/2026-10-26-findings-log.md`.

- **Planted bugs.** Twelve bugs were planted in LTN and LNN operators, one at a time, with 2000 cases each. The MR
  checks caught all twelve. The "per-operator" column is a stand-in for the existing closed-form checks (one operator
  over atoms), not the real old harness. It caught ten.
- **Only the two AggregMin rows separate the oracles.** "AggregMin ignores the last element" and "AggregMin uses the
  wrong axis with two free variables" were missed by per-operator checks. Both were caught by MR and by the full
  reference comparison.
- **MR-only count is 0.** No planted bug was caught only by the MR checks. Where a reference evaluator exists, the full
  reference comparison catches every value bug the MR checks catch. The MR checks do not need a reference, so their
  advantage would show on a library with no reference evaluator, such as NeuPSL. That case is not tested here.
- **Short runs miss some bugs.** A 100-case run (which I used to check the command) missed the wrong-axis bug and the
  LNN "And upward" bug. Quote the 2000-case table, not short runs.
- **Latest smoke runs found nothing.** LTN with 20000 random inputs, LNN with 5000, all libraries with 3000, and PSL
  with 300 and 100 all reported zero violations.
- **An earlier crash was an adapter bug.** Quantifier cases crashed on LTN with a shape error. The cause was in the
  adapter. It was fixed (commit `36d8861`), and the saved input no longer reproduces.
- **LNN quantifiers are open-world by default.** LNN's plain `Forall` and `Exists` add an unseen object, so `forall` can
  drop to 0 and `exists` can rise to 1. With `fully_grounded=True` they use only the declared objects, and they then
  equal the strong fold. The adapter uses the closed-world form.
- **LTN casts predicates to float32 by design.** Formulas with predicates are compared with float32 tolerance, not
  float64.

Limits of the PSL adapter: it covers the PSL core only, which means strong connectives and ground atoms. It has no
quantifiers and no weak `and`/`or`. Formulas deeper than 12 are skipped. NeuPSL's neural predicate layer is not tested.

## Known limits

- **NeuPSL's neural layer is not tested.** The PSL adapter covers the logic core only.
- **PSL is checked less tightly.** The solver stops at a set accuracy, so its tolerance is 1e-5 per node, far looser than
  LTN or LNN.
- **PSL has no quantifiers and no weak connectives.** Those formulas raise `Unsupported`.
- **LNN is checked only at default weights and bias.** The rules A1-A10 hold only there. LNN also takes only unary
  predicates under a quantifier. Nested quantifiers and quantifiers over a variable the body does not use are skipped.
- **Some relations are probably caught by simple checks.** The per-operator stand-in catches most A-group bugs, so the MR
  layer's extra value is mostly in the composition and quantifier rules.
- **Some source claims are unconfirmed.** A8, A9, A13 and van Krieken's implication tables still need checking against
  the text.
- **The LTN-only checks are not in the harness.** `ltn_extras.py` runs through its tests and `run_extras()`, not
  through the command line.
- **Guided fuzzing has not run for long.** The smoke runs used random inputs. A long libFuzzer (Atheris) run is still to
  do.
- **Small cleanups are pending.** `LNN_INFO.log` still needs cleaning up, LNN warnings are silenced for the whole process,
  crash findings are not minimised, and `rewrites.py` is about 670 lines and could be split.
