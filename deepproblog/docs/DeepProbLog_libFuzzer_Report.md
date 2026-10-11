# Coverage-Guided Fuzzing of DeepProbLog with libFuzzer: Report

**Target:** DeepProbLog 2.1.0 (ExactEngine, GraphSemiring) on ProbLog 2.3.0, PySDD 1.0.6, PyTorch 2.14
**Engine:** libFuzzer via Atheris 3.1.0
**Guidance:** code coverage only
**Date:** 11 October 2026

---

## Summary

The method developed for LTNtorch was applied to DeepProbLog, the second neurosymbolic library of the study. libFuzzer drives the library through a harness in which the bytes it mutates are decoded into the choices of seven fuzzing dimensions (P1–P7); the dimensions were derived automatically from DeepProbLog's source. Guidance is code coverage only. The eight adaptations made for LTN were reused and four were added for DeepProbLog (Section 3).

A campaign of **34,292 executions** (8 chunks of about 170 s on two cores) found **13 bugs**; a fourteenth was found by code reading and confirmed with a targeted check (D13). All 14 reproduce as plain DeepProbLog code without the fuzzer (`check_dpl_bugs.py`). A second round (Section 12: metamorphic and first-order oracles, 17,257 further executions) found no further bug. Against the possible-worlds oracle, DeepProbLog's inference produced **no wrong probability and no wrong gradient** on any generated program; the bugs are in the learning step, in the surrounding API and in file handling.

| Objective | Status |
|---|---|
| Use libFuzzer on DeepProbLog | Done |
| Coverage-guided fuzzing only | Done |
| Adapt libFuzzer where needed | Done: A1–A8 reused, A9–A12 added |
| Fix the fuzzing dimensions | Done: P1–P7, derived automatically (333 parameters) |
| Identify adaptations and errors found | Done: 4 new adaptations, 14 bugs |
| ApproximateEngine, image datasets, embeddings | Not covered (Section 9) |

---

## 1. Setup

```bash
cd ~/nesy_libfuzzer_ltn && source venv/bin/activate
cd deepproblog && bash setup_deepproblog.sh        # pip install deepproblog==2.1.0
```

| File | Contents |
|---|---|
| `dpl_fuzz.py` | Atheris harness: eight test families, independent oracles |
| `run_chunks.sh` | Campaign driver: chunks of libFuzzer on two cores, corpus kept between chunks |
| `classify_dims_dpl.py`, `deepproblog_dimensions.csv` | Static extraction of the dimensions P1–P7 |
| `check_dpl_bugs.py` | Every bug as a short piece of plain DeepProbLog code |
| `mutation_check.py` | Plants bugs in DeepProbLog to show that the oracles detect them |
| `findings/<id>/` | One folder per failure signature: `input.bin`, `meta.json` (program, settings, traceback) |
| `logs/` | libFuzzer output of the eight chunks |

---

## 2. libFuzzer is the engine and DeepProbLog is the target

`dpl_fuzz.py` calls `atheris.instrument_imports(include=["deepproblog", "problog"])`, then `atheris.Setup` and `atheris.Fuzz()`. libFuzzer chooses and mutates the input bytes; `FuzzedDataProvider` decodes them into the program, probabilities, network outputs and API calls of one test. Replaying a saved input re-executes the same test:

```bash
python3 dpl_fuzz.py findings/425f75f82c/input.bin     # prints [FINDING 425f75f82c] ... Fewer non-zero entries in p than size
```

---

## 3. Adaptations made to libFuzzer

A1–A8 are those of the LTN study (Python instrumentation through Atheris, structured input decoding, oracles other than crashes, deduplication of findings by signature, chunked runs against Atheris's memory growth, and the Round 3 extensions). Four were added:

| # | Adaptation | Reason |
|---|---|---|
| A9 | Instrument two packages: `deepproblog` and `problog` | DeepProbLog does its inference inside ProbLog; coverage of DeepProbLog alone would hide most of what executes |
| A10 | `-len_control=0 -max_len=200` | libFuzzer starts with inputs of 4 bytes and lengthens them gradually; one DeepProbLog test needs 60–150 bytes. Without this the first minutes were spent on all-zero decodings |
| A11 | `-timeout=25` and loop counters inside the harness | A hang is a finding. The first run stopped at 100 % CPU on one input; the timeout saved the input, and counters in the harness (loader, random generator) turn later hangs into reportable findings that do not stop the run |
| A12 | Tests that write files run in a temporary directory (`in_tmp`) and choose the file name form (`name`, `./name`, `sub/name`) | The fuzzer must not write into the project folder. The bare-name form is the dimension that exposed D4 |

Throughput is 22–27 executions per second against 350–500 for LTN, because every execution grounds a program, compiles a sentential decision diagram and evaluates it.

---

## 4. Guidance is code coverage only

libFuzzer receives no knowledge of DeepProbLog's semantics. Coverage (Atheris edge counters over `deepproblog` and `problog`) rose within each worker from 2,855 to 2,911 and from 2,866 to 2,916 over four chunks, the second and later chunks starting from the corpus of the previous one; 310 and 336 inputs were kept. The numbers are not comparable with those of the LTN study (different library, different Python). The PySDD compiler is native code and is not observed by Atheris.

---

## 5. Fuzzing dimensions

### 5.1 The seven dimensions

| Dim | Meaning | Parameters found by `classify_dims_dpl.py` |
|---|---|---|
| P1 | Logic program: facts, annotated disjunctions, rules, negation, recursion, predicate names | 7 |
| P2 | Probability values: 0, 1, 1e-9, 1−1e-9, random; saturated network outputs | 25 |
| P3 | Neural predicates: `nn/4`, `nn/2`, batching, `k` (top-k), tensor sources | 48 |
| P4 | Queries: `Query`, substitution, `output_ind`, batches of queries | 19 |
| P5 | Solver: cache, `cache_root`, `sdd_auto_gc`, repeated solves | 52 |
| P6 | Learning: loss, SGD on probabilistic parameters, AD groups, training loop, stop conditions | 54 |
| P7 | API boundary: datasets, folds, loaders, save/load, loggers, confusion matrix, file paths, foreign predicates | 123 |
| – | internal helpers | 5 |

The classifier reads 20 source files with Python's `ast` and never imports DeepProbLog (333 parameters in 211 public functions and methods). It reports, for each parameter, whether the function validates it; 317 of 333 (95 %) are unvalidated, a much higher share than in LTN, which is consistent with the kinds of bugs found (Section 8).

### 5.2 Dimensions in the harness

| Test family | Dimensions | What is generated |
|---|---|---|
| `t_prop` | P1 P2 P5 | random stratified programs of up to 6 atoms: facts, annotated disjunctions of 2–3 members, rules with negation, probabilistic rules; cache, `cache_root`, `sdd_auto_gc`, a batch with a repeated query |
| `t_graph` | P1 P2 | up to 6 probabilistic edges on 2–4 nodes (cycles, self-loops, parallel edges), recursive `path/2` |
| `t_neural` | P2 P3 P4 P5 | 7 program templates over `nn/4` and `nn/2` networks; domain 2–4; logits from the input, scaled to saturation; batching; top-k with k = 1..D in eval and train mode; the same input used twice (one ground atom) |
| `t_learn` | P2 P6 | `t(p)` parameters: plain fact, AD all learnable, AD with one fixed member; learning rate from −1 to 1e6; cross-entropy and MSE |
| `t_api` | P4 P7 | folds, subsets, extension, data loader, `Query` round trips and hashing, `save_state`/`load_state` with the file-name forms, foreign predicates |
| `t_train` | P3 P6 P7 | `train_model` with loader, stop conditions, negatives, snapshots; `get_confusion_matrix` |
| `t_data` | P7 | `QueryDataset` file round trip, `ground_dataset`, mutator, logger, confusion matrix, stop conditions |
| `t_names` | P1 | a user fact named like each of the 13 built-in tensor predicates, and ordinary names |

---

## 6. Oracles

All oracles are independent of DeepProbLog and of ProbLog.

| Oracle | Applied in |
|---|---|
| Exact probability by enumerating every possible world (facts, AD choices, rule coins), evaluating rules in order | `t_prop`, `t_graph` |
| Exact probability and gradient from a hand-written formula in torch (joint over network outputs, diagonal when two atoms are the same) | `t_neural`; top-k eval mode uses the masked distribution |
| Analytic gradient of the MSE loss with respect to a parameter | `t_learn` |
| Invariants: probability in [0, 1] and not NaN; parameters in [0, 1]; mass of a disjunction ≤ 1; caching and repetition leave answers unchanged; folds partition the dataset; round trips; the caller's objects are not modified | all families |
| Any exception on a valid input | all families |

The oracles were validated by planting three bugs in DeepProbLog in memory (`mutation_check.py`; 60 random inputs per family). The unmodified library gave 0 failures in `t_prop` and `t_graph`; the one in `t_neural` is bug D7.

| Planted bug | `t_prop` | `t_graph` | `t_neural` |
|---|---|---|---|
| `negate` returns 0.99 − a | 43 | 18 | 22 |
| `plus` drops its second argument when it is a float | 53 | 22 | 1 |
| `times` multiplies by 1.01 | 54 | 18 | 7 |

(`plus` on plain floats does not occur in the neural tests, whose values are tensors.)

---

## 7. Results

| Item | Value |
|---|---|
| Executions | 34,292 (8 chunks of 170 s, two cores) |
| Coverage at the end | 2,911 and 2,916 edges |
| Inputs kept | 310 and 336 |
| Failure signatures | 28 (13 bugs; the 10 `t_names` signatures and 5 round-trip signatures are one bug each) |
| Wrong probability or gradient against the oracles | none |

All bugs but D7 were first seen within the first 2,300 executions of a worker (about 100 s); D7 appeared after about 5,500. Bug D14 was found in an exploratory run of 145 executions before the harness restricted `subset` to valid arguments; the campaign could not find it again.

---

## 8. Bugs

### 8.1 Catalogue

| ID | Severity | Bug | Where | Found by |
|---|---|---|---|---|
| D7 | high | A network used with `k` (top-k) in training mode crashes with `ValueError: Fewer non-zero entries in p than size` when its softmax output has exact zeros, which happens as a network becomes confident | `engines/exact_engine.py:74` | fuzzer, `t_neural` |
| D5 | medium | An annotated disjunction with one learnable and one fixed member: the optimizer step rescales the learnable member to 1.0 and ignores the fixed one, so the total probability exceeds 1 (0.9 → 1.4, even with learning rate 0) | `optimizer.py:92-96` | fuzzer, `t_learn` |
| D6 | medium | `ZeroDivisionError` in the optimizer step when all learnable members of a disjunction reach 0 | `optimizer.py:95` | fuzzer, `t_learn` |
| D4 | medium | `save_state`, `Logger.write_to_file` and `train_model(snapshot_name=...)` fail with `FileNotFoundError: ''` for a file name without a directory part (`check_path` calls `os.makedirs('')`) | `utils/__init__.py:28` | fuzzer, `t_api`, `t_data`, `t_train` |
| D13 | medium | The persistent circuit cache (`cache_root`) names files `str(hash(term))`; Python salts string hashes per process, so a second run never finds the first run's files and adds one new file per query | `utils/cache.py:73` | code reading and targeted check (needs two processes) |
| D3 | medium | `Model.ground_dataset` crashes with `AttributeError ... 'from_cache'` unless the solver was created with `cache=True` (the default is `False`) | `model.py:127` | fuzzer, `t_data` |
| D11 | medium | A user predicate named like one of the 13 built-in tensor predicates (`embed rbf add mul dot max sigmoid mean stack cat one_hot list_to_tensor tensor_index`, with the built-in's arity) is routed to the built-in: ten of them crash inside torch when a plain fact is queried | `engines/builtins.py` | fuzzer, `t_names` |
| D1 | low | `Query.__hash__` always raises `TypeError: unhashable type: 'dict'` | `query.py:94` | fuzzer, `t_api` |
| D2 | low | `DataLoader(batch_size=0)` yields empty batches forever (and `len()` divides by zero) | `dataset.py:137` | fuzzer, `t_api` |
| D8 | low | `Threshold(duration=n)` raises `IndexError` when fewer than n values are logged and the first is above the limit | `utils/stop_condition.py:132` | fuzzer, `t_data` |
| D9 | low | `NumericIncorrectOutputMutator` loops forever if its domain contains only the current value | `dataset.py:326` | fuzzer, `t_data` |
| D10 | low | `ConfusionMatrix(classes)` appends new classes to the caller's list | `utils/confusion_matrix.py` | fuzzer, `t_data` |
| D14 | low | `Dataset.subset(i, j)` with `i` beyond the end gives a negative length (`ValueError`); a Python slice gives an empty result | `dataset.py` | fuzzer, `t_api` (exploratory run) |
| D12 | low | ProbLog rounds every float to 15 decimal places, so `1e-20::a.` has probability 0.0 and a query file with 8.6e-16 reads back as 1e-15. The cause is in ProbLog (`problog/logic.py:901`); it affects every DeepProbLog program and `QueryDataset` | ProbLog | fuzzer, `t_data` round trip |

Severity is the author's judgement: D7 stops training when it occurs and occurs naturally; D5 and D6 give wrong probabilities or crash during learning; D4, D13, D3, D11 break documented features on common inputs; the rest are edge cases of misuse.

### 8.2 Reproduction without the fuzzer

```bash
python3 check_dpl_bugs.py list        # the bugs
python3 check_dpl_bugs.py D7          # one bug: its code, then run raw (real traceback or failed assertion)
python3 check_dpl_bugs.py             # all, with a summary table
```

Expected last line: `14 of 14 bug demonstrations fail exactly as described`. Each demonstration is written as a bug report: it uses DeepProbLog as documented, states what should happen, and fails when the bug is present. If DeepProbLog is fixed, the function returns normally and the bug is reported as NOT REPRODUCED.

### 8.3 Replay of saved fuzzer inputs

```bash
python3 dpl_fuzz.py findings/425f75f82c/input.bin     # D7
python3 dpl_fuzz.py findings/d483dac29c/input.bin     # D5
python3 dpl_fuzz.py findings/ece78a03d0/input.bin     # D6
python3 dpl_fuzz.py findings/1e8c33b899/input.bin     # D4
```

Each prints the same `[FINDING <id>]` line as in the campaign.

### 8.4 Observations that are not counted as bugs

- `Network.__call__` calls `inputs.cuda(device=...)` and discards the result. On a GPU the stacked inputs would stay on the CPU. This was read in the code and could not be run here (no GPU).
- A query with no answer gives zero loss and is dropped from the batch (`get_loss`); a query whose target is 1 but whose probability is 0 therefore contributes nothing. This looks like a design decision.
- With an unbound variable, negation is evaluated left to right (Prolog semantics): `h(X) :- \\+ f(X), g(X).` means 'there is no f at all', not 'f(X) is false for this X'. Not a bug, but a trap for users.
- `AD` parameter groups are renormalised to sum to 1 at every step, also with learning rate 0. This is consistent with a learned distribution and is not counted.

---

## 9. Findings

1. DeepProbLog's inference was correct on every generated program: probabilities of programs with negation, annotated disjunctions, probabilistic rules, cycles and neural predicates matched the possible-worlds oracle, and the gradients matched torch autograd. The bugs lie around the inference.
2. Thirteen of the fourteen bugs were found by libFuzzer, seven of them in the first 1,000 executions of a worker. The one found by reading the code (D13) depends on two processes and cannot be seen from inside one fuzzing process.
3. A dimension that the harness does not generate cannot yield bugs: D4 appeared only after the file-name form became a dimension, and D14 was lost when the arguments of `subset` were restricted. Dimensions bound what libFuzzer can find; coverage decides how quickly.
4. The first false alarms came from the oracle, not from DeepProbLog: two uses of the same neural input are one ground atom and are perfectly correlated, and the first oracle treated them as independent. The mutation check and the ground-atom correction are what make the "no wrong probability" result trustworthy.
5. DeepProbLog validates almost nothing (317 of 333 parameters), so misuse produces exceptions deep inside torch or numpy rather than messages.

---

## 10. Limitations

- Only the ExactEngine and GraphSemiring were fuzzed. The ApproximateEngine needs SWI-Prolog and PySwip, which were not installed.
- Programs are small (up to 6 atoms or edges, domains of 2–4) so that the oracle can enumerate all worlds.
- Top-k networks in training mode sample their values; only validity of the probability can be checked there.
- The example applications (MNIST, CLUTRR, Poker, Forth), `TermEmbedder` and image datasets were not fuzzed.
- The compiler (PySDD) is native and unobserved by Atheris; coverage numbers refer to Python code only.
- 34,292 executions is a small campaign; coverage was still rising slowly at the end (2,855 → 2,911).
- Exploratory runs before the campaign are not counted in the execution total.

## 11. Next steps

1. Report the bugs upstream, one issue per bug with `check_dpl_bugs.py <ID>` as its reproduction, starting with D7, D5 and D6.
2. Longer campaigns and a larger domain (7+ atoms) with a faster oracle.
3. Fuzz `TermEmbedder`, image datasets and the ApproximateEngine.
4. Apply the same method to NeurASP and LNN.

---

## 12. Round 2/3: metamorphic and first-order oracles

As in the LTN study, a second harness (`dpl_round3.py`, separate from `dpl_fuzz.py`, same engine and the same coverage-only guidance) adds oracles that compare a program with a transformed version of itself, and extends the programs to variables and constants.

| Test family | What is generated | Oracle |
|---|---|---|
| `t_meta_prog` | the random propositional programs of `t_prop`, transformed: clauses shuffled; atoms renamed from `a<i>` to `p(<i>)`; unused probabilistic facts added; a deterministic clause duplicated; a trivially true extra rule | possible-worlds probability, which must not change under any transformation |
| `t_fo` | 2–3 constants, probabilistic `f(c)` and `g(c)`, one rule with a variable (`f(X), \+ g(X)`, `f(X), g(X)`, `f(X)`, `g(X), \+ f(X)`), optionally a second rule; existential query `some :- h(_)`; non-ground query `h(X)` | enumeration of all worlds; every answer of the non-ground query must equal the ground query, and no answer may be missing |
| `t_fo_ad` | annotated disjunctions with a variable (`p1::color(X,red); p2::color(X,green) :- obj(X).`), probabilistic rules with a variable, negation over derived atoms | enumeration of all colourings |
| `t_batch` | batches of 1–4 queries, repeated | batch answers equal single-query answers of a solver with the opposite `cache` setting; a repeated batch gives identical answers |
| `t_domain_perm` | a neural predicate whose domain order and network output columns are permuted together | the probability of one value must not change |

**Result.** 17,257 executions (11 chunks of 100–150 s on one core) and 601 kept inputs: **no failure in the real library**. The new families covered 2,193 Python edges of `deepproblog` and `problog` (a different harness from the 2,911 of Section 7, not comparable). The planted-bug check (`mutation_check_round3.py`, 40 random inputs per family):

| Planted bug | `t_meta_prog` | `t_fo` | `t_fo_ad` | `t_batch` |
|---|---|---|---|---|
| unmodified library | 0 | 0 | 0 | 0 |
| `negate` returns 0.99 − a | 30 | 39 | 35 | 0 |
| `plus` drops its second argument when it is a float | 35 | 40 | 40 | 10 |
| `times` multiplies by 1.01 | 30 | 38 | 21 | 0 |

(`t_batch` compares two runs of the same library, so a consistent arithmetic error is not visible to it; that is what the possible-worlds families are for.)

A further informal differential probe (`diff_probe.py`) compared DeepProbLog's ExactEngine with ProbLog on 60 random programs using arithmetic comparison, `is`, `findall/length`, negation and disjunction in bodies; no difference was found.

**One oracle error, not a library bug.** The first `t_fo` flagged a rule written as `\+ f(X), g(X)`: ProbLog evaluates negation left to right, so with `X` still unbound it means "no `f` at all", which is Prolog's meaning and gives a different number from the declarative reading. The rule was reordered to `g(X), \+ f(X)`. This is worth knowing for users (Section 8.4).

**Conclusion.** Together with Section 9, this strengthens the first finding: DeepProbLog's inference was correct on every generated program, including first-order programs with annotated disjunctions, negation, arithmetic and non-ground queries; the bugs of Section 8 lie around the inference, in learning, file handling and API validation. Limitations of Section 10 still apply (ExactEngine only, small programs, no image datasets).
