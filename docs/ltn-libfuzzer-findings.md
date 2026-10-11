# LTN (LTNtorch) coverage-guided fuzzing with libFuzzer — notes

Target: LTNtorch commit d1bd981. Engine: libFuzzer via Atheris 3.1, coverage guidance only.

## Round 1: libFuzzer harness, basic oracle (ltn_fuzz/)
Campaigns "valid" (D1–D6) and "misuse" (D1–D7), ~30 min each, ~250–1300 exec/s.
Coverage of ltn/: 82% line+branch; libFuzzer cov plateau 266 / 272.

### Adaptations libFuzzer needed
1. Language: LTN is Python, libFuzzer instruments C/C++ only -> Atheris (real libFuzzer engine + Python bytecode coverage).
2. Selective instrumentation: only `ltn` instrumented; torch imported first, uninstrumented.
3. Input decoding: raw bytes -> structured LTN program via FuzzedDataProvider (dimensions D1–D7).
4. Oracle: no sanitizer for Python logic bugs. Filter LTN's own validation errors; add value / gradient oracles.
5. Keep-going mode: de-duplicate failures by signature, save reproducers.
6. Harness-bug control: harness reshape on 0 individuals and harness overflow caused early false alarms; fixed.
7. Memory: Atheris instrumentation grows RSS ~10 KB/exec (not LTN); fake slow units. Mitigation: chunked campaigns resumed from corpus.
8. (Round 3) Extended dimensions and oracles: ltn_round3.py, see below.

### Dimensions
D1 values · D2 shape (0–5 individuals) · D3 predicate model · D4 formula (all connectives, depth 3, stable) · D5 quantifiers (aggregators, p, diag, guarded) · D6 SatAgg + backward · D7 API boundary (vacuous quantification, Quantifier([]), open formula to SatAgg).

## Round 2: combined with neys_fuzz oracles (ltn_combined/)
Source of manual fuzzers: github.com/PromaChow/neys_fuzz, branch fuzz-ltn-lnn-probfuzz-adaptation, commit 9512567
(fuzzing/ltn/fuzz_*.py, fuzz/ltn_fuzz.py, fuzz/ltn_oracle.py, results/seeds/ltn_seeds.json).
Guidance unchanged (coverage only). Added: reference formula (ltn_oracle.py), Boolean-boundary law, swap / permute / stable-flip metamorphic checks, dtype check, shape-vs-free-variables check, unstable-gradient category, and a seed mode that mutates the 105 operator calls recorded from LTN's own tests.
Campaigns: 4 x 5 min chunks each; valid 443k execs, misuse 573k execs; ~350–500 exec/s (checks cost ~30–40% speed).
Coverage 82–83% line+branch (libFuzzer cov 279 / 285): stronger oracle, not more code.

## Findings
New relative to neys_fuzz (found by libFuzzer harness):
- F1 Empty domain: Variable with 0 individuals. Min -> IndexError (fuzzy_ops.py:1220); Mean/pMean/pMeanError -> NaN. Guarded quantification with empty mask returns 1.0/0.0, so the paths disagree. With Quantifier([]) and Min: RuntimeError (same line).
- F2 p = 0 not validated in AggregPMean / AggregPMeanError -> ZeroDivisionError at fuzzy_ops.py 1400, 1402, 1512, 1514. neys_fuzz tested p=0 but its check counted any exception as a pass.
- F3 Vacuous quantification -> ValueError "'y' is not in list" at core.py 1520 / 1502; works if guarded on the quantified variable itself.
Rediscovered by the combined fuzzer (originally neys_fuzz):
- F24 ImpliesGoguen(stable=True) gives 0 for false->false (Boolean law + stable flip). First seen at ~18k–43k execs.
- F25 Quantifier([]) returns a scalar that still claims free variables (structure check, misuse mode).
- F26 dtype: guarded quantifier float64, masked AggregMin float64, NotGodel float32 for float64 input. First seen within ~50–9k execs.
- F27 NaN gradients with unstable operators (low severity).
Not ported yet: brute-force quantifier evaluation (ltn_quant.py), LTN vs LNN differential (ltn_vs_lnn.py), finite-difference gradient check.
Negative results: no reference-formula mismatch beyond F24; no float32 precision notes; no NaN on non-empty domains; no NaN gradients with stable operators.
Side observation (static): invalid escape sequences in docstrings -> SyntaxWarning on Python 3.12+.

## Round 3: extended dimensions and oracles (ltn_combined/ltn_round3.py), 2026-10-11
Separate Atheris harness (no seed corpus). Adds: per-call p / stable, diag / undiag, add_batch_dim=False, odd dtypes and shapes, Predicate(model=) and Predicate(func=), vector Function, trainable Constant, multi-variable and nested quantifiers, raw tensors to SatAgg, same-label variables, reuse of one sub-formula in two formulas. New oracles: caller-object mutation, shape vs free variables, Forall/Exists duality, all-true mask == no mask, quantifier variable order, diag == paired mean, dtype.
Campaign: 4 parallel 4-minute chunks, ~712k execs (~740/s per process), plus two tuning runs. User's own 4-minute run: 402,859 execs, 1,670/s, cov 328 (cov differs by Python version; compare only within one machine). cov plateaued by ~11k execs. Findings found within the first ~30k execs.
New bugs:
- F29 (high) Connective / Predicate modify their operand in place when it carries a gradient (model-based predicate or Function output): free_vars ['x'] -> ['x','y'], shape (3,) -> (12,). Detached predicates are copied correctly. core.py process_ltn_objects copies only when grad_fn is None or the object is a Variable/Constant.
- F30 (medium) Two variables with the same label but different sizes: sizes 1 and 2 accepted silently (shape (2,), F30b); sizes 3 and 4 give a raw torch error raised inside the user's predicate (F30). Best described as missing validation, since a label identifies a variable.
- F31 (low) SatAgg() with no formulas -> RuntimeError "stack expects a non-empty TensorList" (fuzzy_ops.py:1669).
- F33 (low) Variable(torch.tensor(0.5)) (0-d) accepted, every use raises IndexError (core.py:307).
Also re-found F1, F2, F26. F28 (SatAgg non-scalar tensor, TypeError inside its own error message, fuzzy_ops.py ~1668) still found by code reading only; Round 3 did not reach it, reason not investigated.

## Reproduction (check_all_bugs.py)
Each bug is a short function of plain LTN code that fails the way the bug describes: `PYTHONPATH=ltn_torch python3 check_all_bugs.py list | <ID> | (no argument = all + summary)`. Running one ID prints its source and runs it raw (real traceback or failed assertion). 14 demonstrations for 12 bugs (F1/F1b, F30/F30b are variants). Verified 14 of 14 on the user's machine (Python version changes the wording of F3's message, not the verdict).

## Upstream tracker check (11 Oct 2026)
LTNtorch has two issues, both closed: #1 AggregPMean mask type check on GPU (fixed in a later release), #2 question about process_ltn_objects broadcasting (answered by the reporter: pairing is chosen with diag; no maintainer reply). Neither reports any bug above. Pull requests and discussions not checked.

## Status
LTN work: 12 bugs (F1-F3, F24-F31, F33), dimensions derived automatically (classify_dims.py, 104 parameters), final report in docs/ (LTN_libFuzzer_Final_Report.md) with step1-step7 verification files. Open: push to GitHub (private repo), 3 manual oracles not ported, F28 not found by the fuzzer. Next library: DeepProbLog (installs in a clean venv: deepproblog 2.1.0, problog 2.3.0, pysdd 1.0.6), then NeurASP and LNN.
