# Coverage-Guided Fuzzing of LTNtorch with libFuzzer: Final Report

**Target:** LTNtorch 1.0.2 (Logic Tensor Networks, PyTorch implementation), commit `d1bd981`
**Engine:** libFuzzer via Atheris 3.1.0
**Guidance:** code coverage only
**Date:** 11 October 2026

---

## Summary

The study applies libFuzzer's coverage-guided fuzzing to neurosymbolic AI libraries (LTN, DeepProbLog, NeurASP and LNN), starting with LTN. libFuzzer could not be applied to LTN unchanged: it relies on compile-time instrumentation of C/C++ and on sanitizers, while LTN is a Python library whose defects are mostly incorrect values rather than crashes. Eight adaptations were made around the unchanged libFuzzer engine, and seven fuzzing dimensions (D1–D7) were defined from LTN's API. The dimensions and test oracles of existing manual fuzzers were then integrated into the libFuzzer harness. A third round (Round 3) then added the dimension choices that the first two rounds could not generate, together with new oracles.

About one million executions in Rounds 1–2 found **three new bugs** (F1–F3) and re-found **four bugs** first reported by the manual fuzzers (F24–F27). An eighth bug (F28) was found by code inspection. Round 3 (about 0.7 million further executions) found **four more new bugs** (F29, F30, F31, F33), the most serious being that connectives modify their operands in place (F29). All twelve reproduce without the fuzzer.

| Objective | Status |
|---|---|
| Use libFuzzer, starting with LTN | Done |
| Coverage-guided fuzzing only | Done |
| Adapt libFuzzer where needed | Done: 8 adaptations |
| Fix the fuzzing dimensions; observe libFuzzer's behaviour | Done: D1–D7, derived automatically from LTN's code |
| Adapt the dimensions of the existing manual fuzzers | Done; 3 oracle ideas not yet ported |
| Identify adaptations and errors found | Done: 8 adaptations, 12 bugs |
| DeepProbLog, NeurASP, LNN | Not started |

---

## 1. Setup

All commands run in WSL (Ubuntu) from the project folder:

```bash
cd ~/nesy_libfuzzer_ltn
source venv/bin/activate
```

| Path | Contents |
|---|---|
| `ltn_torch/` | LTNtorch source at commit d1bd981 (system under test) |
| `ltn_fuzz/` | Round 1 harness: libFuzzer harness, dimensions D1–D7, basic oracle |
| `ltn_combined/` | Round 2 harness: Round 1 plus the manual fuzzers' oracles and seeds |
| `ltn_combined/logs/` | libFuzzer output of every campaign |
| `ltn_combined/findings_valid/`, `findings_misuse/` | one folder per failure signature: `input.bin`, `program.txt`, `meta.json` |
| `check_all_bugs.py` | standalone reproduction of every bug (F1–F3, F24–F31, F33) |
| `round3/ltn_round3.py` | Round 3 harness: new dimension choices and oracles (A8) |
| `classify_dims.py` | static extraction of fuzzing dimensions from LTN's source |

Source of the manual fuzzers: `github.com/PromaChow/neys_fuzz`, branch `fuzz-ltn-lnn-probfuzz-adaptation`, commit `9512567`.

---

## 2. libFuzzer is the engine and LTN is the target

| Command | What it checks | Result |
|---|---|---|
| `pip show atheris \| head -2` | Atheris, which ships the libFuzzer engine for Python, is installed | `atheris 3.1.0` |
| `head -12 ltn_combined/logs/valid.log` | the libFuzzer engine starts | `INFO: Using built-in libfuzzer`, `entropic power schedule`, `Seed: 101` |
| same log, first lines | only LTN is instrumented | `Instrumenting ltn`, `ltn.core`, `ltn.fuzzy_ops`; no torch modules |
| `grep -n "import atheris\|instrument_imports\|atheris.Setup\|atheris.Fuzz" ltn_combined/*.py` | the harness hands control to libFuzzer and restricts coverage to `ltn` | line 63 `instrument_imports(include=["ltn"])`; lines 563–564 `atheris.Setup` / `atheris.Fuzz()` |
| `git -C ltn_torch log -1 --oneline` | exact target version | `d1bd981 … version 1.0.2` |
| `grep -h "number_of_executed_units" ltn_combined/logs/*.chunk*.log \| awk '{s+=$2} END {print s}'` | total executions of the campaigns | ≈ 1.0 million (≈443k valid, ≈573k misuse) |

Campaigns: two modes (valid programs; API misuse), four chunks each, about 350–500 executions per second.

---

## 3. Adaptations made to libFuzzer

The libFuzzer engine (mutation, corpus management, coverage feedback) is unchanged. Everything around it was adapted.

| # | Adaptation | Reason | Implementation |
|---|---|---|---|
| A1 | Python support | libFuzzer obtains coverage from clang at compile time; Python is not compiled | Atheris inserts coverage counters into Python bytecode at import time; the engine reads them like clang's |
| A2 | Selective instrumentation | instrumenting PyTorch would be slow and coverage would follow PyTorch instead of LTN | `torch` imported first, uninstrumented; `ltn` imported inside `instrument_imports(include=["ltn"])` |
| A3 | Input decoding | libFuzzer produces raw bytes; LTN needs programs | `FuzzedDataProvider` reads bytes as a sequence of choices (46 choice points) that build an LTN program |
| A4 | Oracles | no sanitizer exists for Python, and many LTN defects are silent wrong values | value, formula, metamorphic, dtype, structure and gradient oracles; LTN's own validation errors are ignored |
| A5 | Keep-going mode | libFuzzer stops at the first crash, and the empty-domain crash appears at input 1 | failures de-duplicated by signature, saved, fuzzing continues (`LTNFUZZ_COLLECT=1`) |
| A6 | Harness-bug control | a failure in the harness itself looks like an LTN bug | two early harness false alarms fixed; every reported bug reproduced without the fuzzer |
| A7 | Chunked campaigns | Atheris grows memory by about 7–10 KB per execution, producing fake slow inputs | short chunks that restart the process and resume from the saved corpus (`run_chunks.sh`) |
| A8 | Extended dimensions and oracles (Round 3) | the first harness could not generate several API choices, and some defects need new checks | `ltn_round3.py`: eleven byte-driven tests covering per-call `p`/`stable`, `diag`/`undiag`, `add_batch_dim=False`, odd dtypes and shapes, `Predicate(model=/func=)`, vector-valued `Function`, trainable `Constant`, multi-variable and nested quantifiers, raw tensors to `SatAgg`, same-label variables and reuse of a sub-formula; oracles for caller-object mutation, shape vs free variables, Forall/Exists duality, all-true mask, quantifier variable order, `diag` vs paired mean and dtype |

### Verification

| Command | What it checks | Result |
|---|---|---|
| `grep "Instrumenting" ltn_combined/logs/valid.log` | A1, A2: coverage comes from LTN modules only | `ltn`, `ltn.core`, `ltn.fuzzy_ops` |
| `grep -n "FuzzedDataProvider\|ConsumeIntInRange\|ConsumeBool" ltn_combined/ltn_fuzz_combined.py` | A3: two decoders | line 136 (program mode), line 455 (seed mode), wrapped in choice helpers |
| `xxd ltn_combined/findings_valid/34be71331f/input.bin; cat ltn_combined/findings_valid/34be71331f/program.txt` | A3: one input as bytes and as the LTN program built from them | a few bytes → a program with a 0-individual variable and a Min quantifier |
| `grep -n "Violation" ltn_combined/ltn_fuzz_combined.py` | A4: oracle checks | lines 358 (structure), 364 (dtype), 414 (closed formula NaN / outside [0,1]), 422 (SatAgg), 435/439 (gradient), 511 (seed-mode range) |
| `sed -n 80,95p ltn_combined/ltn_fuzz_combined.py` | A4: LTN's own validation messages are not counted as bugs | prefixes such as `Expected inputs of connectives and quantifiers`, `Expected parameter` |
| `cat ltn_combined/findings_valid/5e774fb287/meta.json` | A4: a silent bug caught only by an oracle | `OracleViolation … nan … aggs=['Mean']` (F1) |
| `ls ltn_combined/findings_valid \| wc -l` | A5: many distinct failures from one campaign | 23 signatures; the first one (F1) was hit 134,864 times and saved once |
| `PYTHONPATH=ltn_torch python3 check_all_bugs.py` | A6: bugs belong to LTN, not the harness | all twelve reproduce as plain LTN code (Section 8.2) |
| `cd ltn_combined && PYTHONPATH=../ltn_torch python3 ltn_round3.py -max_total_time=240 -rss_limit_mb=3000` | A8: Round 3 harness runs under libFuzzer | about 1,800 executions/s alone (about 740/s with four parallel chunks), `cov` 268, `[FINDING …]` lines; results in `findings_r3/<signature>/meta.json` |
| `cat ltn_combined/run_chunks.sh`; `rss` in `logs/valid.log` | A7: chunking and memory growth | chunks resume from the corpus; memory 314 → 461 MB within one minute |

### Contrast: plain libFuzzer on C++

Google's tutorial target `fuzz_me.cc` was compiled with `clang++ -g -fsanitize=address,fuzzer` and fuzzed on the same machine.

| | Plain libFuzzer (`fuzz_me.cc`) | libFuzzer on LTN |
|---|---|---|
| Coverage counters | `7 inline 8-bit counters`, inserted by clang at compile time | inserted by Atheris at import time (A1, A2) |
| Input | raw bytes used directly (crash input `FUZ`) | bytes decoded into an LTN program (A3) |
| Bug detector | AddressSanitizer: `heap-buffer-overflow` | oracles (A4) |
| At first bug | stops, writes `crash-…` | saves, continues (A5) |
| Memory | flat (32 → 39 MB over ~92,000 executions) | grows (A7) |

The C++ run also shows libFuzzer's comparison tracing (`CMP- DE: "F\000\000\000"`): the engine observed a comparison with `'F'` and inserted that byte into the input.

---

## 4. Guidance is code coverage only

libFuzzer keeps an input in the corpus only if it increases coverage of `ltn` (`NEW`), or replaces a longer input with the same coverage (`REDUCE`). The oracles record failures but have no channel to libFuzzer.

| Command | What it checks | Expected |
|---|---|---|
| `grep -n "dict=\|use_value_profile\|custom_mutator\|CustomMutator\|custom_crossover" ltn_combined/*.sh ltn_combined/*.py` | no extra guidance (dictionary, value profile, custom mutator or crossover) | no output |
| `grep -n "atheris\." ltn_combined/*.py` | the harness uses only standard Atheris calls | only `instrument_imports`, `FuzzedDataProvider`, `Setup`, `Fuzz` |
| log check below | every kept input added coverage | `without coverage gain: 0`, `coverage changed: 0` in every log |

```bash
for f in ltn_combined/logs/*.chunk*.log; do
  printf "%-22s " "$(basename $f)"
  awk '/ (INITED|NEW|REDUCE|pulse|DONE) /{for(i=1;i<=NF;i++) if($i=="ft:") ft=$(i+1)
       if($2=="NEW"){n++; if(ft<=prev) bad++} if($2=="REDUCE"){r++; if(ft!=prev) rbad++} prev=ft}
       END{printf "NEW %d (without coverage gain: %d)  REDUCE %d (coverage changed: %d)\n", n, bad+0, r, rbad+0}' "$f"
done
```

Precise statement of the guidance: libFuzzer's default coverage feedback on the `ltn` package, that is, edge coverage with hit counts plus comparison tracing (`-use_cmp=1`, on by default). No dictionary, custom mutator, value profile or custom fitness function is used.

Some oracles (swap, permute, stable flip) call LTN a second time, so they can add coverage. The type of guidance is unchanged.

---

## 5. Fuzzing dimensions

### 5.1 The seven dimensions

The dimensions were derived from LTN's public API. Each is a group of choices the harness makes from libFuzzer's bytes. The table follows `ltn_fuzz/README.md`, lines 48–54.

| Dim | Name | Choices | LTN API |
|---|---|---|---|
| D1 | data values | 0, 1, 1e-4, NaN, ±inf, 1e30, 3.4e38, denormals | `Variable` / `Constant` contents |
| D2 | data shape | 1–3 variables, **0–5 individuals** each, feature width 1–3, 1-D vs 2-D, constants | `Variable` / `Constant` size |
| D3 | predicate model | sigmoid / crisp step / clamp, arity 1–2, weights; optional function term f(x) | `Predicate`, `Function` |
| D4 | formula structure | Not, And, Or, Implies, Equiv trees up to depth 3; every operator family; `stable` flag | `Connective`, `fuzzy_ops.*` |
| D5 | quantification | Forall / Exists; Min, Mean, pMean, pMeanError; **p ∈ {1, 2, 3, 6, 10, 0, −1, 0.5}**; diag; guarded | `Quantifier`, `diag`, `cond_vars` / `cond_fn` |
| D6 | satisfaction | SatAgg operator, 1–2 closed formulas, optional backward pass | `SatAgg`, `.backward()` |
| D7 | API boundary (misuse mode only) | quantifying a variable absent from the formula, `Quantifier([])`, SatAgg on an open formula | validation gaps in `Quantifier.__call__`, `SatAgg.__call__` |

### 5.2 Verification in the harness

| Command | What it checks | Result |
|---|---|---|
| `sed -n 1,40p ltn_fuzz/ltn_fuzz.py \| grep -n "D[1-7]"`; `grep -n "D[1-7]" ltn_fuzz/README.md` | dimensions documented in code and README | D1–D7 listed in both; Round 2 reuses them unchanged |
| `grep -c "self\.i(" ltn_combined/ltn_fuzz_combined.py` | number of byte-driven choice points | 46 |
| `grep -n "AggregMin\|AggregMean\|AggregPMean" ltn_combined/ltn_fuzz_combined.py` | D5 aggregators | all four (lines 290–297) |
| `grep -n "AndProd\|OrProbSum\|ImpliesGoguen\|ImpliesReichenbach\|Equiv" ltn_combined/ltn_fuzz_combined.py` | D4 operator families with the `stable` flag | And, Or, Implies, Equiv (lines 265–280) |
| `grep -n "diag\|cond_fn\|cond_vars" ltn_combined/ltn_fuzz_combined.py` | D5 diagonal and guarded quantification | `ltn.diag` (314), guard (336–337) |
| `grep -n "MISUSE" ltn_combined/ltn_fuzz_combined.py` | D7 enabled only in misuse mode | `LTNFUZZ_MISUSE=1` (77); vacuous quantification (321) and open SatAgg (409), each with probability 1/10 |

Two observations from the code:

- Inside `Equiv`, the inner operators use the default `stable` value, so `stable` is not varied for Equiv.
- In misuse mode an extra byte is read for D7 decisions, so the same `input.bin` decodes into different programs in the two modes. This is why the corpora are kept separate, and why findings must be replayed in their original mode.

### 5.3 Automatic derivation of the dimensions

`classify_dims.py` parses LTN's source with Python's `ast` module (LTN is not executed). It maps every public parameter to a dimension through three rule tables and flags parameters that LTN never validates.

```bash
python3 -W ignore classify_dims.py ltn_torch/ltn --csv ltn_dimensions.csv | tail -15
```

| What it checks | Result (commit d1bd981) |
|---|---|
| parameters per dimension | 104 parameters: D1 37, D2 11, D3 7, D4 17, D5 34, D6 3, D7 5 |
| validation gaps | `p` in `AggregPMean` / `AggregPMeanError` is unvalidated; empty `individuals` and absent quantified variables are not rejected |

The bugs lie exactly at these validation gaps: F2 at `p`, F1 at empty `individuals`, F3 and F25 in `Quantifier.__call__`. Only the rule tables are LTN-specific, so the same script applies to DeepProbLog, NeurASP and LNN.

### 5.4 Gaps in the Round 1–2 dimensions (addressed in Round 3, Section 7.2)

| Missing choice | Dimension | Relevance |
|---|---|---|
| `p` / `stable` passed per call (`Forall(x, φ, p=0)`) | D5 | a second code path for F2 |
| `undiag` | D5 | public API, never exercised |
| `add_batch_dim=False` | D2 | unvalidated; changes tensor shape |
| `Predicate(func=…)` | D3 | different constructor branch |
| `Constant(trainable=True)` | D6 | gradient flow into constants |
| raw non-scalar tensor passed to `SatAgg` | D7 | required for the fuzzer to reach F28 |

Round 3 implements all six choices. F28 was still not found by the fuzzer in Round 3; the reason has not been investigated.

### 5.5 Observed libFuzzer behaviour

| Observation | Source | Result |
|---|---|---|
| mutation loop | `logs/valid.log`, e.g. `REDUCE … MS: 3 CopyPart-InsertByte-EraseBytes-` | saved inputs are mutated at byte level; only inputs adding coverage (`NEW`) or shortening an input (`REDUCE`) are kept |
| plateau | 1-minute run: 24,527 executions, 402/s | `cov` constant (339 on this Python version), 13 corpus reductions, 0 new signatures |
| difficulty per bug | `first_seen_exec` in `meta.json` | F1: input 1; F26: 174–9,195; F2: ~13,800; F27: ~34,100; F24: ~42,900 |

Bugs needing one choice (an empty domain) appear immediately. Bugs needing several choices to coincide (p = 0 together with a pMean aggregator; the stable Goguen implication with inputs 0, 0) take thousands of inputs.

Raw `cov` values depend on the Python version, because Atheris instruments bytecode. Measured on the original environment: libFuzzer `cov` 279/285, and 82–83% line + branch coverage of `ltn`.

---

## 6. Integration of the manual fuzzers' dimensions

### 6.1 Manual fuzzer dimensions and their libFuzzer equivalents

The manual fuzzers (neys_fuzz) test one operator call at a time with random values and strong oracles, without coverage feedback. Each of their dimensions became a byte-driven choice, mostly in the harness's **seed mode**. Seed mode handles one input in four, chosen by the first byte. It mutates 105 operator calls recorded from LTN's own test suite.

| Manual dimension | libFuzzer equivalent | Location |
|---|---|---|
| M1 operator | seed mode picks one of 105 recorded calls; D4/D5 in program mode | `ltn_seeds.json` |
| M2 boundary values (0, 1) | seed mode boundary setting; D1 | seed mode |
| M3 perturbation | step from `DELTAS`, up or down | line 497 |
| M4 complement (1 − x) | seed mode | seed mode |
| M5 stable flag | seed-mode flip; D4/D5 | line 471 |
| M6 p value | seed mode; D5 | seed mode |
| M7 mask | random mask from bytes | line 502 |

### 6.2 Manual oracles ported

| Manual oracle | Ported form | Finding class |
|---|---|---|
| reference formula (`ltn_oracle.py`) | copied unchanged, float64 | `FormulaViolation` |
| Boolean law on 0/1 inputs | tolerance ±3e-4 | `FormulaViolation` |
| swap, permute, stable flip | tolerances 3e-6, 1e-5, 3e-4 | `MetamorphicViolation` |
| dtype | ported | `DtypeViolation` |
| structure (shape vs free variables) | ported | `StructureViolation` |
| NaN gradient with unstable operators | added, low severity | `GradientViolation` |

| Command | What it checks | Expected |
|---|---|---|
| `diff neys_fuzz/fuzz/ltn_oracle.py ltn_combined/ltn_oracle.py` | reference oracle copied unchanged | no differences |
| `python3 -c "import json; print(len(json.load(open('ltn_combined/ltn_seeds.json'))))"` | seed corpus size | 105 |
| `sed -n 1,30p ltn_combined/README.md` | origin of each component; two input modes | "What came from where" table; program mode 3/4, seed mode 1/4 |
| `ORACLES=basic ./run_chunks.sh basic 0 1 300` (in `ltn_combined/`) | without the ported oracles, F24 and F26 are not found | only crash / NaN signatures |

Round 2 added no code coverage over Round 1 (82% → 82–83%), yet it re-found four bugs. The gain came entirely from stronger oracles.

### 6.3 Not yet ported

| Idea | Purpose |
|---|---|
| brute-force quantifier evaluation (`ltn_quant.py`) | check quantifier values against an explicit loop over individuals |
| LTN vs LNN differential (`ltn_vs_lnn.py`) | detect disagreements between two implementations |
| finite-difference gradient check | detect wrong (not only NaN) gradients |

These are oracle extensions, not missing input dimensions.

---

## 7. Results

### 7.1 Rounds 1–2

| Measure | Value |
|---|---|
| executions | ≈ 1.0 million (valid ≈ 443k, misuse ≈ 573k) |
| throughput | 350–500 executions/s |
| coverage of `ltn` | 82–83% line + branch; plateau reached early |
| failure signatures (valid campaign) | 23, de-duplicating to 5 root causes (F1: 12, F2: 4, F24: 3, F26: 3, F27: 1) |
| bugs | 3 new (F1–F3), 4 re-found (F24–F27), 1 by code inspection (F28) |

The signature count over-states the number of bugs, because signatures include the aggregators present in the program. Root causes are the meaningful count.

### 7.2 Round 3

| Measure | Value |
|---|---|
| harness | `ltn_round3.py`, eleven byte-driven tests (A8) |
| executions | ≈ 712,000 in four parallel 4-minute chunks (seeds 101–104), after two tuning runs of about 110,000 and 287,000 |
| throughput | about 1,800/s single process; about 740/s per process with four in parallel |
| coverage of `ltn` | `cov` 268 (Atheris counter value for this harness, not comparable to the 339 of Rounds 1–2) |
| distinct signatures after de-duplication | 19 across the chunks |
| new bugs | 4 (F29, F30, F31, F33) |
| re-found | F1 (empty domain), F2 (p ≤ 0), F26 (mask changes dtype) |
| harness artefact, not an LTN bug | `IndexError` at `core.py:616`, caused by the harness's own model receiving 1-D input |

Known causes were filtered in the harness: NaN outputs on empty variables (F1) and on p ≤ 0 (F2) are not reported again, so every remaining signature is either new or a different failure. Coverage did not rise in Round 3 (268 at every chunk end); the new bugs came from new choices and new checks, as in Round 2.

---

## 8. Bugs

### 8.1 Catalogue

| ID | Description | Location | Dimension → oracle | Status | Severity |
|---|---|---|---|---|---|
| **F1** | "for all x" over **0 individuals**: Min → `IndexError`; Mean, pMean, pMeanError → `nan`; guarded path → `1.0`. Inconsistent; logically 1.0 | `fuzzy_ops.py:1220` (Min); mean-type aggregators | D2 + D5 → exception and value oracles | **new** | high: silent NaN during training |
| **F2** | `AggregPMean(p=0)` / `AggregPMeanError(p=0)` accepted by the constructor; use raises `ZeroDivisionError` | `fuzzy_ops.py` 1400, 1402, 1512, 1514 | D5 → exception oracle | **new** | medium |
| **F3** | quantifying a variable absent from the formula raises `ValueError` (`'y' is not in list` / `list.index(x): x not in list`, depending on the Python version); logically equals the unquantified formula | `core.py` ~1520 (`Quantifier.__call__`) | D7 → exception oracle | **new** | medium |
| F24 | `ImpliesGoguen(stable=True)(0, 0)` returns 0.0; false → false must be 1.0 (`stable=False` returns 1.0) | `ImpliesGoguen` | D4 + D1 → formula and metamorphic oracles | re-found | high: wrong truth value |
| F25 | `Quantifier([])` returns a scalar that still lists free variables | `Quantifier.__call__` | D7 → structure oracle | re-found | low |
| F26 | dtype changes: `NotGodel` float64 → float32; masked `AggregMin` and guarded quantifier float32 → float64 | `NotGodel`, `AggregMin`, `Quantifier` | D4/D5 → dtype oracle | re-found | low |
| F27 | NaN gradient with unstable operators, e.g. `AggregPMean(p=2, stable=False)` at inputs 0 | unstable aggregators / connectives | D4/D5 + D6 → gradient oracle | re-found | low: stable variants are the documented fix |
| F28 | `SatAgg` given a non-scalar plain tensor raises `TypeError: 'torch.Size' object is not callable` inside its own error message instead of the intended `ValueError` | `fuzzy_ops.py` ~1669 | found by code inspection; the current D7 cannot generate this input | **new** | low |
| **F29** | A `Connective` or a `Predicate` **modifies its operand in place** when the operand carries a gradient, such as the output of a model predicate or a `Function`. After `AndProd(px, qy)`, `px.free_vars` changes from `['x']` to `['x','y']` and its shape from (3,) to (12,). Reusing the sub-formula in a second formula then gives wrong shapes. A `Predicate` call on a `Function` output changes it from `['x']`, shape (3, 2) to `['x','y']`, shape (12, 2). Operands without a gradient are copied correctly, so behaviour is inconsistent. LTN's own comment states that reuse must be possible | `core.py` `process_ltn_objects` (copy only when `grad_fn is None` or the object is a `Variable` / `Constant`) | Round 3: reuse and mutation oracles | **new** | high: silent wrong shapes and values in training |
| **F30** | Two variables with one label but different sizes. 1 vs 2 individuals is accepted silently (result shape (2,)); 3 vs 4 gives a raw torch `RuntimeError` instead of an LTN message | `core.py` 307, 359 (`process_ltn_objects`) | D2 same-label → oracle | **new** | medium |
| **F31** | `SatAgg()` called with no formulas raises `RuntimeError: stack expects a non-empty TensorList` | `fuzzy_ops.py:1669` | D6/D7 → exception oracle | **new** | low |
| **F33** | `Variable(torch.tensor(0.5))` (0-d) is accepted by the constructor; every later use raises `IndexError: tuple index out of range` | `core.py:307` | D2 → exception oracle | **new** | low |

"New" means not found by the manual fuzzers. The upstream issue tracker of LTNtorch was checked on 11 October 2026: it contains two issues, both closed. Issue #1 (AggregPMean not compatible with GPU optimization) concerns a `torch.BoolTensor` mask check on CUDA. Issue #2 (a question about `process_ltn_objects`) asks why tensors of the same shape are repeated, and suspects that broadcasting follows variable labels instead of shapes; the reporter answered himself that pairing is chosen later with `diag`, and closed it with no maintainer reply. Neither issue reports any bug in this catalogue. Issue #2 concerns the same function as F29 and F30, but it asks about label-based broadcasting; it does not describe modification of the caller's objects (F29) or acceptance of same-label variables of different sizes (F30). Since a label identifies a variable in LTN, F30 is best read as a missing validation (two variables sharing a label with different sizes are not rejected), not as a fault in broadcasting. No match was found for the other bugs, so all twelve are new relative to the manual fuzzers and the tracker.

The manual fuzzers also tested p = 0, but counted any exception as a pass. The libFuzzer oracle accepts only LTN's documented validation messages, which is why it reported F2.

### 8.2 Reproduction without the fuzzer

Every bug is a short function of plain LTN code in `check_all_bugs.py`, written the way a bug report is: it uses LTN as documented and states what should happen. When the bug is present the function fails, either with LTN's own exception or with an assertion that names the wrong value.

```bash
PYTHONPATH=ltn_torch python3 check_all_bugs.py list      # the 14 demonstrations
PYTHONPATH=ltn_torch python3 check_all_bugs.py F29       # show the code of one bug and run it raw
PYTHONPATH=ltn_torch python3 check_all_bugs.py           # run all, then print a summary table
```

Running one bug by its identifier prints its source code and then runs it with nothing caught, so the output is LTN's own traceback or the failed assertion. Running without an argument prints each bug's code and failure, then a table. The exit code is 0 only if all 14 demonstrations fail as described (expected last line: `14 of 14 bug demonstrations fail exactly as described`). Bugs with two variants have two identifiers: F1 (Min raises `IndexError`) and F1b (Mean, pMean and pMeanError return `nan`), and F30 (sizes 3 and 4 raise a raw torch error) and F30b (sizes 1 and 2 are accepted silently).

| Bug | Expected (correct behaviour) | Observed (d1bd981) |
|---|---|---|
| F1 | one consistent value (1.0) | `IndexError` (Min); `nan` (Mean, pMean, pMeanError); `1.0` (guard) |
| F2 | `ValueError` from the constructor | `ZeroDivisionError: division by zero` (both aggregators) |
| F3 | same as P(x), or a clear error | `ValueError: 'y' is not in list` |
| F24 | 1.0 | `0.0` (stable=True); `1.0` (stable=False) |
| F25 | no free variables, or an error | shape `()`, free variables `['x']` |
| F26 | output dtype = input dtype | `float32` (NotGodel on float64); `float64` (masked AggregMin, guarded Forall on float32) |
| F27 | finite gradient | `[nan, nan, nan]` (stable=False); finite (stable=True) |
| F28 | LTN's `ValueError` about scalar formulas | `TypeError: 'torch.Size' object is not callable` |
| F29 | operand unchanged after the call | `px.free_vars ['x'] → ['x','y']`, shape `(3,) → (12,)`; a detached predicate stays `['x']` |
| F30 | LTN's `ValueError` about inconsistent labels | `RuntimeError: The size of tensor a (3) must match the size of tensor b (4)` (sizes 3 and 4); shape `(2,)` accepted (sizes 1 and 2) |
| F31 | LTN's `ValueError`, or 1.0 | `RuntimeError: stack expects a non-empty TensorList` |
| F33 | `ValueError` from the constructor | constructor accepts shape `()`; `IndexError: tuple index out of range` on use |

### 8.3 Replay of saved fuzzer inputs

```bash
cd ltn_combined
LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_valid/34be71331f/input.bin              # F1
LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_valid/221f71e246/input.bin              # F2
LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_valid/96b8e7b1f7/input.bin              # F24
LTNFUZZ_MISUSE=1 LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_misuse/<id>/input.bin  # F3, F25
```

Each replay runs libFuzzer's exact input with instrumentation off, confirming that the failure comes from the input and not from the fuzzing machinery.

### 8.4 Not counted as bugs

- Invalid escape sequences in docstrings cause `SyntaxWarning` on Python 3.12 and later. This is cosmetic.
- Connective operators called directly do not range-check their inputs; only `Connective(...)` does. This is a design choice.
- The empty-domain failures that Round 3 also meets with `diag` and with two empty variables are F1, not new bugs.
- Negative results: no reference-formula mismatch beyond F24, no NaN on non-empty domains, and no NaN gradients with stable operators.

---

## 9. Findings

1. **libFuzzer's engine transfers to a neurosymbolic Python library.** It requires an instrumentation front-end, a byte-to-program decoder, and oracles.
2. **Oracle strength dominates.** Coverage plateaus early because most of LTN is branch-free tensor arithmetic: branch coverage cannot distinguish p = 2 from p = 0.001. Stronger oracles found four more bugs with no change in coverage.
3. **Dimensions bound what can be found.** The three new bugs of Rounds 1–2 come from choices the manual fuzzers did not make (empty domains, misuse) or judged too leniently (p = 0).
4. **Bugs cluster at validation gaps.** Static extraction of unvalidated parameters points to the locations of F1, F2, F3 and F25.
5. **Adding dimensions and oracles found more.** Round 3 added choices and checks and found four further bugs with no gain in coverage. F29 needs two things at once: a model-based predicate (a result carrying a gradient) and a second use of that result; neither Round 1 nor Round 2 could generate that combination, and a value-only oracle could not have detected it. This confirms that the dimensions and the oracles, not the coverage, set what libFuzzer can find.
6. **The combination is complementary.** libFuzzer's dimensions found new bugs, and the manual fuzzers' oracles let libFuzzer re-find the known ones.

---

## 10. Limitations

- PyTorch is not instrumented, so branches inside its C++ core are invisible to coverage.
- Coverage counts depend on the Python version and are comparable only within one environment.
- Failure signatures are finer than root causes (23 signatures for 5 causes in the valid campaign).
- F28 was not found by the fuzzer in Rounds 1–3. Round 3 does generate non-scalar raw tensors for `SatAgg`, so the remaining cause has not been investigated.
- The three manual oracles in Section 6.3 are not yet implemented.
- Round 3 filters known causes (F1, F2) in its NaN check, so it cannot report a second, different NaN cause on empty variables or p ≤ 0.
- The Round 3 harness is a separate file from the Round 2 harness and does not share its seed corpus.
- Novelty of F1–F3, F28–F31 and F33 is relative to the manual fuzzers and to the two issues in the LTNtorch tracker on 11 October 2026 (no match); the tracker is small, so the absence of a match shows only that nobody reported these in the tracker, and does not show that the maintainers are unaware of them.

---

## 11. Next steps

1. Find out why Round 3 does not reach F28, and merge the Round 3 tests into the Round 2 harness so they share seeds and corpus.
2. Port brute-force quantifier evaluation, the LTN–LNN differential and finite-difference gradient checks.
3. Report the new bugs upstream, one issue per bug, each with its `check_all_bugs.py <ID>` demonstration. Start with F29 and F1/F1b.
4. Apply the same method to DeepProbLog, then NeurASP and LNN. DeepProbLog and NeurASP contain native components (PySDD, clingo), where clang instrumentation (`-fsanitize=fuzzer-no-link`) can be combined with Atheris.
