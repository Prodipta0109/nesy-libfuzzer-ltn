# Coverage-Guided Fuzzing of LNN with libFuzzer: Report

**Target:** IBM LNN 1.1.0 (Logical Neural Networks, commit `c88f513`), PyTorch 2.14, Python 3.13
**Engine:** libFuzzer via Atheris 3.1.0
**Guidance:** code coverage only
**Date:** 11 October 2026

---

## Summary

The method developed for LTNtorch and applied to DeepProbLog was applied to LNN, the third neurosymbolic library of the study. libFuzzer drives the library through a harness in which the bytes it mutates are decoded into the choices of seven fuzzing dimensions (L1–L7); the dimensions were derived automatically from LNN's source. Guidance is code coverage only. The twelve adaptations of the earlier studies were reused and two were added (Section 3).

A campaign of **52,589 executions** (8 chunks of about 170 s on two cores) found **five bugs**; a second round (Section 12, metamorphic oracles and binary predicates, nested quantifiers) found **three more**, and a ninth demonstration (L1b) shows the user-visible consequence of the first. All nine reproduce as plain LNN code without the fuzzer (`check_lnn_bugs.py`). Against the closed-form oracles, LNN's **propositional inference was exact**: upward bounds of And, Or, Not, Implies and Iff matched the Łukasiewicz formulas on every input, and classical data gave classical results. The bugs are in how long inference runs, in the identity of formulas, in an unfinished feature, and in one loss function.

| Objective | Status |
|---|---|
| Use libFuzzer on LNN | Done |
| Coverage-guided fuzzing only | Done |
| Adapt libFuzzer where needed | Done: A1–A12 reused, A13–A14 added |
| Fix the fuzzing dimensions | Done: L1–L7, derived automatically (173 parameters); L4 only partly (Section 5.1) |
| Identify adaptations and errors found | Done: 2 new adaptations, 8 bugs (5 in round 1, 3 in round 2/3) |
| Binary predicates, nested quantifiers | Covered in round 2/3 (Section 12); Gödel semantics not testable (L3) |

---

## 1. Setup

```bash
cd ~/nesy_libfuzzer_ltn && source venv/bin/activate
cd lnn && bash setup_lnn.sh        # clones IBM/LNN at c88f513 into ./src and installs matplotlib networkx pandas tabulate tqdm
```

LNN is used from the source clone (`src/`), not installed with pip; the scripts add `src` to `sys.path`.

| File | Contents |
|---|---|
| `lnn_fuzz.py` | Atheris harness: nine test families, independent oracles |
| `run_chunks.sh` | Campaign driver: chunks of libFuzzer on two cores, corpus kept between chunks |
| `classify_dims_lnn.py`, `lnn_dimensions.csv` | Static extraction of the dimensions L1–L7 |
| `check_lnn_bugs.py` | Every bug as a short piece of plain LNN code |
| `mutation_check.py` | Plants bugs in LNN to show that the oracles detect them |
| `findings/<id>/` | One folder per failure signature: `input.bin`, `meta.json` |
| `logs/` | libFuzzer output of the 16 chunks of the campaign |
| `phase1_hangs/` | The first, interrupted campaign: its libFuzzer timeout inputs and logs (Section 7) |

---

## 2. libFuzzer is the engine and LNN is the target

`lnn_fuzz.py` calls `atheris.instrument_imports(include=["lnn"])`, then `atheris.Setup` and `atheris.Fuzz()`. libFuzzer chooses and mutates the input bytes; `FuzzedDataProvider` decodes them into the formulas, truth bounds, neuron settings and API calls of one test. Replaying a saved input re-executes the same test:

```bash
python3 lnn_fuzz.py findings/7afeae3fbc/input.bin     # prints [FINDING 7afeae3fbc] ... true_hinge
```

---

## 3. Adaptations made to libFuzzer

A1–A8 are those of the LTN study, A9–A12 those added for DeepProbLog (A9 here instruments only `lnn`; A12 was not needed because LNN writes no files). Two were added:

| # | Adaptation | Reason |
|---|---|---|
| A13 | Every inference call in the harness has a step limit (`max_steps=60`), and reaching the limit is itself a finding (non-convergence oracle) | LNN's `infer()` loops until the bounds stop changing and has no default limit. In the first campaign (no limit in the training test) chunks ended with a libFuzzer timeout, and one worker kept running at 100 % CPU for an hour. A limit turns a hang into a reportable finding that does not stop the run |
| A14 | The step limit is also passed through `Model.train(..., max_steps=...)` | `train` forwards its keyword arguments to `infer`; this is the only way to bound it |

Throughput is about 19 executions per second in the campaign (DeepProbLog 22–27, LTN 350–500).

---

## 4. Guidance is code coverage only

libFuzzer receives no knowledge of LNN's semantics. Coverage (Atheris edge counters over the `lnn` package) rose from 669 and 672 after the first chunk to 673 in both workers and stayed there from the fifth chunk; 389 and 403 inputs were kept. The plateau is early because the harness reaches little of the package beyond the propositional and unary-predicate paths (Section 10); the numbers are not comparable with the other libraries.

---

## 5. Fuzzing dimensions

### 5.1 The seven dimensions

| Dim | Meaning | Parameters found by `classify_dims_lnn.py` |
|---|---|---|
| L1 | Formula structure: connectives And, Or, Not, Implies, Iff, XOr, arity, depth, repeated operands | 45 |
| L2 | Truth bounds and data: facts, intervals, points, UNKNOWN, contradictions, worlds | 21 |
| L3 | First-order structure: predicates, variables, groundings, quantifiers | 26 |
| L4 | Neuron configuration: activation type, weights, bias, alpha, learning flags | 0 found (see below) |
| L5 | Inference control: direction, source, step limit | 27 |
| L6 | Learning: losses, optimizer, epochs, labels | 9 |
| L7 | API boundary: knowledge, data and label dictionaries, printing, reset, flush, utilities | 33 |
| – | internal helpers | 12 |

The classifier reads 23 source files with Python's `ast` and never imports LNN (173 parameters in 143 functions and methods). It reports, for each parameter, whether the function validates it by one of LNN's `Assert*` classes; 160 of 173 (92 %) are unvalidated. **L4 shows 0 because LNN passes neuron settings through one `activation={...}` dictionary (`**kwds`)**, which a parameter-level analysis cannot split; its keys were listed by hand from `neural/parameters/neuron.py` (`type`, `alpha`, `weights`, `bias`, `weights_learning`, `bias_learning`, `w_max`, `b_max`, `negative_weights`) and generated in `t_weights` and `t_learn`.

### 5.2 Dimensions in the harness

| Test family | Dimensions | What is generated |
|---|---|---|
| `t_upward` | L1 L2 | random formula trees of depth 1–3 over 2–4 propositions, with bounds (TRUE, FALSE, UNKNOWN, intervals, points); upward inference; one run in eight uses the Gödel activation |
| `t_classical` | L1 L2 | the same trees with classical data; full inference |
| `t_downward` | L1 L2 L5 | a root axiom (true or false) and all but one leaf known; downward inference |
| `t_invariants` | L1 L2 | random bounds; inference; interval and NaN checks |
| `t_conv` | L2 L3 L5 | `Forall` / `Exists` over a unary predicate with 1–3 groundings, body `S(x)`, `Not S(x)` or `S(x) → S(x)`; inference with step limit |
| `t_fol` | L2 L3 | the same, checking the quantifier bounds against the data |
| `t_weights` | L1 L2 L4 | And / Or / Implies with weights from {0.5, 0.9, 1.0, 1.5, 2.0}, bias from {1.0, 0.9, 0.8}; upward bounds, then full inference with step limit |
| `t_learn` | L4 L6 | one And / Or / Implies / Iff with learnable weights and bias; SUPERVISED, CONTRADICTION or LOGICAL loss; 1–4 epochs |
| `t_api` | L7 | malformed bounds, `reset_bounds`, `flush`, `print`, round trips of the fact utilities, unusual proposition names |

---

## 6. Oracles

All oracles are independent of LNN's code.

| Oracle | Applied in |
|---|---|
| Closed-form Łukasiewicz interval for each connective (And lower = max(0, Σl − (n−1)), Or, Not, Implies, Iff as two Implies) | `t_upward` |
| Closed-form weighted Łukasiewicz interval (bias and weights from the paper) | `t_weights` |
| Classical truth value of every node when all leaves are classical; no contradiction may be reported | `t_classical` |
| Soundness of downward inference: after fixing the root, the hidden leaf's bounds must still contain every classical completion that satisfies the root | `t_downward` |
| Containment of quantifier bounds in the Łukasiewicz And/Or of the grounded bounds plus one implicit unknown grounding (open world), tolerance 0.05 | `t_fol` |
| Convergence: inference must stop before the step limit | `t_conv`, `t_weights` |
| Invariants: 0 ≤ lower ≤ upper ≤ 1 unless a contradiction is flagged; no NaN; parameters finite after training | all families |
| Any exception on valid input that is not one of LNN's own validation exceptions (raised in `_exceptions.py`) | all families |

The oracles were validated by planting three bugs in LNN in memory (`mutation_check.py`; 150 random inputs per family). The unmodified library gave one failure in `t_upward` and three in `t_classical`; all four are bug L5 (formulas built twice). The Gödel module error (L3) is excluded from the count.

| Planted bug | `t_upward` | `t_classical` | `t_weights` | `t_invariants` |
|---|---|---|---|---|
| And result lowered by 0.05 | 55 | 39 | 16 | 0 |
| Or result raised by 0.1 | 25 | 14 | 18 | 0 |
| Implies result scaled by 0.9 | 65 | 78 | 55 | 0 |

(`t_invariants` checks only the range; a small shift does not leave [0, 1].)

---

## 7. Results

| Item | Value |
|---|---|
| Executions | 52,589 (8 chunks of 170 s, two cores) |
| Coverage at the end | 673 and 673 edges |
| Inputs kept | 389 and 403 |
| Failure signatures | 18 (5 bugs: 8 signatures are L5, 5 are L2, 3 are L1, one each L3 and L4) |
| Wrong bounds against the oracles other than L5 | none |

**First campaign (interrupted).** The first campaign had no step limit in the training test. Most chunks ended early with a libFuzzer timeout and one worker spun for an hour; this is how L1b was found. The timeout inputs and logs are kept in `phase1_hangs/`. The harness was then changed (A13, A14) and the campaign repeated; only the second campaign is counted above. The weighted non-convergence (L1) was confirmed by hand after the first campaign and was found again by the fuzzer in the second (`t_weights`, three signatures: And, Or, Implies), once the harness generated points such as 0.4 as truth values.

---

## 8. Bugs

### 8.1 Catalogue

| ID | Severity | Bug | Where | Found by |
|---|---|---|---|---|
| L1 | high | `infer()` never terminates for a weighted And, Or or Implies whose weights differ, when an operand with the lighter weight has an exact value (e.g. weights (1.0, 0.5), `B = 0.4`). Each step moves the bounds by one float32 step, always above the stopping threshold of 1e-7, so the bounds creep towards 0 or 1 forever | `model.py:507` with the weighted activations | fuzzer, `t_weights`; phase 1 via `t_learn` |
| L1b | high | Consequence of L1: `Model.train()` never returns once training has made the weights unequal (the final inference has no step limit). Learnable weights are LNN's main feature | `model.py:619`, `model.py:635` | fuzzer, `t_learn` (timeouts of the first campaign) |
| L2 | high | `infer()` never terminates for a model with a quantifier over a predicate that has an informative grounding (`Exists(x, S(x))` with `S(a)` true; `Forall` with a false grounding; intermediate values). The quantifier rebuilds its neuron on every upward step, so every step re-learns the same bounds and reports them as new | `symbolic/logic/unary_operator.py:248` | fuzzer, `t_conv` |
| L5 | medium | A formula is identified by its printed name. When the same formula is built twice (for example an explicit `Implies(P, Q)` and the one inside `Iff(P, Q)`), the model keeps the first object and the other is never updated; the answer depends on the order of `add_knowledge`. `Or(P → Q, R)` with P true, Q false, R false is [0, 1] in one order and [0, 0] in the other | `model.py:275` | fuzzer, `t_upward`, `t_classical` |
| L3 | medium | `NeuralActivation.Godel` and `Frechet` are offered by the API, but `lnn/neural/methods` contains only the Łukasiewicz modules: constructing such a neuron raises `ModuleNotFoundError` | `symbolic/logic/neural_activation.py` | fuzzer, `t_upward` |
| L4 | medium | `Loss.LOGICAL` raises `UnboundLocalError: true_hinge` for Iff and XOr (the loss supports And, Or and Implies only) | `symbolic/logic/connective_neuron.py:170` | fuzzer, `t_learn` |

Severity is the author's judgement: L1, L1b and L2 make ordinary use (weights, training, quantifiers) hang, with no message; L5 gives incomplete answers silently; L3 and L4 are unfinished features that crash.

### 8.2 Reproduction without the fuzzer

```bash
python3 check_lnn_bugs.py list        # the bugs
python3 check_lnn_bugs.py L5          # one bug: its code, then run raw (real traceback or failed assertion)
python3 check_lnn_bugs.py             # all, with a summary table
```

Expected last line: `9 of 9 bug demonstrations fail exactly as described`. Each demonstration is written as a bug report. Calls that should return are guarded (a step limit, or an alarm of a few seconds) so the demonstration ends; the failing assertion states that the call did not return or did not converge. If LNN is fixed, the function returns normally and the bug is reported as NOT REPRODUCED.

### 8.3 Replay of saved fuzzer inputs

```bash
python3 lnn_fuzz.py findings/7afeae3fbc/input.bin     # L4
python3 lnn_fuzz.py findings/4656ce79a7/input.bin     # L3
python3 lnn_fuzz.py findings/4535b5ea66/input.bin     # L2
python3 lnn_fuzz.py findings/092b467c12/input.bin     # L1
python3 lnn_fuzz.py findings/8c751f7bc6/input.bin     # L5
```

Each prints the same `[FINDING <id>]` line as in the campaign.

### 8.4 Observations that are not counted as bugs

- `XOr` with three or more operands means "exactly one is true", not parity. This is a valid definition but differs from the usual reading.
- `reset_bounds()` restores the bounds from the data that was added, not to UNKNOWN; `flush()` sets everything to UNKNOWN. Both follow their documentation.
- A quantifier is evaluated as a Łukasiewicz And / Or over its groundings plus one implicit unknown grounding (open world), so `Forall` over a single grounding with bounds (0.2, 0.4) gives [0, 0.4]. This is consistent with the open-world assumption.
- Weights must be given as a tuple; a list raises a clear `TypeError` from LNN's validation.

---

## 9. Findings

1. LNN's propositional inference was exact: every connective matched the closed-form Łukasiewicz (and weighted) intervals, and classical data gave classical results in 52,589 executions. The bugs lie in termination, in formula identity and in unfinished parts.
2. The most serious problems are about **not stopping**. Termination is decided by a threshold of 1e-7 on the change of the bounds, which is at the resolution of float32, and by an update count that quantifiers report wrongly. A convergence oracle with a step limit is therefore as important here as a value oracle; the first campaign without it lost most of its time.
3. All five bugs were found by libFuzzer. L1 first showed as a hang (a libFuzzer timeout) in the first campaign and as a named finding in the second. It appeared only when exact truth values such as 0.4 became part of the data dimension: a dimension that is not generated cannot yield bugs.
4. Eight signatures were false alarms in the harness before they were understood as bug L5: expected values were read from the formula object the test had built, while the model kept another object with the same name. Reading the model's own node (`stored`) and the orphan check separated the two.
5. LNN validates 13 of 173 parameters; because neuron settings travel in a dictionary, unusual values inside it are checked, if at all, deep in the neuron classes.

---

## 10. Limitations

- Round 1 generated only propositional formulas and quantifiers over unary predicates; binary predicates and nested quantifiers were added in round 2/3 (Section 12). Predicates of arity 3, functions, and joins of three or more predicates are still not covered.
- The Gödel and Fréchet activations could not be tested (L3).
- The quantifier oracle checks containment with a tolerance, not exact values.
- The learning oracle checks exceptions, convergence and finite values only; it does not check gradients.
- Neuron settings (L4) were chosen by hand, not by the classifier.
- Visualisation (`plot_graph`, `plot_loss`, `plot_params`), `Model.print` options and the optimisation modules (`constraint_optim`, `primal_dual`) were barely touched.
- 52,589 executions is a small campaign. Exploratory runs and the first campaign are not counted in the execution total.
- LNN was taken from a source clone, version 1.1.0 at commit `c88f513`; the PyPI release, if different, was not examined.

## 11. Next steps

1. Report the bugs upstream, one issue per bug with `check_lnn_bugs.py <ID>` as its reproduction, starting with L1 and L2.
2. Extend the generator to binary predicates and nested quantifiers, and make the quantifier oracle exact.
3. Add a gradient oracle for `train` and fuzz the optimisation modules.
4. Apply the same method to NeurASP.

---

## 12. Round 2/3: metamorphic oracles and first-order structure

As in the LTN study, a second harness (`lnn_round3.py`, same engine, coverage-only guidance) adds oracles that compare a formula or a model with a transformed version of itself, and extends the first-order dimension L3 to binary predicates and nested quantifiers.

| Test family | What is generated | Oracle |
|---|---|---|
| `t_meta_prop` | three propositions; And/Or with permuted operands; De Morgan (both directions); material implication; double negation; Iff symmetry | the two sides must have identical bounds |
| `t_meta_model` | four formulas added in a random order; inference run twice; `flush` and data added again | same bounds and same contradiction flag |
| `t_contradiction` | a root asserted opposite to what classical data give (And/Or/Implies/Iff) | a contradiction must be reported |
| `t_fol2` | domain of 2–3 individuals, complete classical data for `R(x,y)` and `S(y)`; `∃y R∧S`, `∀x∃y R∧S`, `∀x∀y R→S`, `∀x,y R→S`, `∃x∃y R∧S` | no contradiction for consistent data; the classical truth value lies inside the bounds |
| `t_fol_shapes` | formulas with a quantifier over a bare predicate with another free variable, reflexive `R(x,x)` combined with other predicates, `R(x,y) → R(y,x)` | any raw exception |
| `t_learn_ranges` | learnable And/Or/Implies; SUPERVISED, CONTRADICTION, UNCERTAINTY; learning rate 0 to 100 | weights and bias stay non-negative and finite |

**Result.** 6,654 executions (11 chunks of 100–150 s on one core), 157 inputs kept, 700 Python edges covered (a different harness from Section 4; not comparable). Propositional metamorphic checks gave **no violation**: commutativity, De Morgan, double negation, material implication, order of `add_knowledge` and repeated inference all held. Three new bugs, all first-order:

| ID | Severity | Bug | Where | Found by |
|---|---|---|---|---|
| L6 | high | A consistent quantified sentence is reported as a contradiction. `Forall(x, S(x))` with `S(a)` true and `S(b)` false is simply false (and `Exists` with the same data is simply true), but downward inference from the quantifier is applied to every grounding: `S(a)` (resp. `S(b)`) becomes the impossible bounds (1, 0) and `has_contradiction()` is True. Seen with complete classical data over 3 individuals for `∀x∀y R→S` and `∃x∃y R∧S` | `symbolic/logic/unary_operator.py:267-272` (`_fully_quantified_downward`) | fuzzer, `t_fol2` |
| L7 | medium | A quantifier applied directly to a predicate that has another free variable (`Exists(y, R(x,y))`, hence the common pattern `Forall(x, Implies(S(x), Exists(y, R(x,y))))`) fails while the formula is built: `ValueError: tuple.index(x): x not in tuple`. LNN's own tests for this case are commented out | `symbolic/logic/unary_operator.py:83` | fuzzer, `t_fol_shapes` |
| L8 | medium | A predicate with a repeated variable (`R(x,x)`) combined with any other formula (`Or(S(x), R(x,x))`) fails in inference with a pandas `Length mismatch`; `R(x,x)` alone works | `symbolic/logic/` grounding tables (pandas `set_axis`) | fuzzer, `t_fol_shapes` |

`check_lnn_bugs.py L6` shows the code and runs it raw; the full script now ends with `9 of 9 bug demonstrations fail exactly as described`.

**Oracle errors, not library bugs.** Two signatures of the first round-3 run came from the harness: data supplied for a predicate that the formula does not use, and an expected upper bound for the bias (`b_max` has no default, so the bias is only bounded below). Both oracles were corrected before the counted run.

**Planted-bug check** (`mutation_check_round3.py`, 40 random inputs per family): see the table below.

| Planted bug | `t_meta_prop` | `t_meta_model` | `t_contradiction` |
|---|---|---|---|
| unmodified library | 0 | 0 | 0 |
| And result lowered by 0.05 | 29 | 1 | 0 |
| Or result raised by 0.1 | 40 | 2 | 0 |
| Implies result scaled by 0.9 | 16 | 8 | 0 |

The metamorphic family detects the planted arithmetic errors (a mutant that treats the two sides of De Morgan differently shows up at once); the order and flush family sees them rarely, and `t_contradiction` not at all, because it tests whether a contradiction is reported, which these mutants do not change. The first-order families (`t_fol2`, `t_fol_shapes`) were not part of this check: they already fail on the unmodified library (L6–L8).

**Findings of the round.**
1. The quantifier code is the weak part of LNN: three of the eight bugs (L2, L6, L7) and probably L8 live in its first-order handling, which round 1 only reached with one unary predicate. Propositional inference stayed exact under every metamorphic transformation.
2. L6 was found only after the data became complete classical models over several individuals and binary predicates: a dimension that is not generated cannot yield bugs.
3. `add_data` replaces earlier data for the same formula silently (`A` true, then false, gives false with no contradiction). This follows the documentation and is not counted.

**Replay.**

```bash
python3 lnn_round3.py findings_round3/a43a34cab0/input.bin     # L6
python3 lnn_round3.py findings_round3/c0faf71d86/input.bin     # L7
python3 lnn_round3.py findings_round3/e8a171202b/input.bin     # L8
```
