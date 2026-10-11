# Step 6: Overall result: adaptation needed, what it was, and the errors found

**Requirement (supervisor):** overall, understand whether any adaptation was needed for libFuzzer, what it is, and what errors the fuzzer found.

**Short answer:**

1. **Was adaptation needed?** Yes. libFuzzer works on compiled C/C++ and finds crashes; LTN is Python and its bugs are mostly wrong values.
2. **What was it?** Seven adaptations around the unchanged libFuzzer engine (details and checks in `step3_libfuzzer_adaptations.md`).
3. **What errors were found?** **3 new bugs** (F1, F2, F3), **4 rediscovered** bugs first found by the manual fuzzers (F24–F27), and **1 extra bug found by code reading** (F28), all in LTNtorch 1.0.2 (commit d1bd981). All 8 reproduce without the fuzzer.

**Verified on:** ____ Oct 2026, WSL (Ubuntu), folder `~/nesy_libfuzzer_ltn`

---

## Part 1: Was adaptation needed? What was it?

| # | Adaptation | Why plain libFuzzer couldn't do it | Evidence |
|---|---|---|---|
| A1 | **Atheris** runs the libFuzzer engine on Python | libFuzzer's coverage comes from clang at compile time; Python isn't compiled | log: `INFO: Using built-in libfuzzer` |
| A2 | Only **`ltn`** is instrumented | instrumenting PyTorch too would be slow and coverage would follow PyTorch, not LTN | log: `Instrumenting ltn / ltn.core / ltn.fuzzy_ops` |
| A3 | Bytes **decoded into LTN programs** (dimensions D1–D7) | libFuzzer makes raw bytes; LTN needs variables, predicates, formulas | `input.bin` → `program.txt` in every finding |
| A4 | **Oracles** decide what is a bug | Python has no sanitizer, and many LTN bugs are silent wrong values (NaN, wrong result) | F1's NaN reported by `OracleViolation` (harness line 414) |
| A5 | **Keep-going**: failures saved, fuzzing continues | libFuzzer stops at the first crash; F1 appears at input 1 | 23 signatures from one campaign |
| A6 | **Harness-bug control** | a crash in the fuzzer's own code looks like an LTN bug | every bug reproduces without the fuzzer (Part 3) |
| A7 | **Chunked runs** | Atheris grows memory by ~7–10 KB per execution | `rss` 314 → 461 MB in one minute; `run_chunks.sh` |

What did **not** change: libFuzzer's engine (mutation, corpus, coverage feedback). Guidance is coverage only (`step2_coverage_guided_only.md`).

Contrast with plain libFuzzer on C++ (Google's `fuzz_me.cc`, run 10 Oct 2026): clang inserted 7 coverage counters at compile time, AddressSanitizer detected a `heap-buffer-overflow`, and libFuzzer **stopped** at the first crash. LTN has no compile step, no sanitizer, and needs the run to continue, which is exactly what A1, A4 and A5 replace.

---

## Part 2: The errors found

### Overview

| ID | Bug | Status | Severity | Found by |
|---|---|---|---|---|
| **F1** | empty domain gives a crash, NaN or 1.0 depending on the path | **new** | high (silent NaN in training) | libFuzzer harness |
| **F2** | p = 0 accepted, then `ZeroDivisionError` | **new** | medium | libFuzzer harness |
| **F3** | quantifying a variable not in the formula crashes | **new** | medium | libFuzzer harness (misuse mode) |
| F24 | `ImpliesGoguen(stable=True)` gives 0 for false → false | rediscovered | high (wrong logic value) | combined fuzzer, ported checks |
| F25 | `Quantifier([])` result still claims free variables | rediscovered | low | combined fuzzer, misuse mode |
| F26 | output dtype differs from input dtype | rediscovered | low | combined fuzzer, ported checks |
| F27 | NaN gradients with unstable operators | rediscovered | low (stable versions are the documented fix) | combined fuzzer |
| F28 | `SatAgg` crashes inside its own error message | **new** | low | **code reading, not the fuzzer** |

"New" means not found by the manual fuzzers (neys_fuzz). **Before calling them previously unreported, check the LTNtorch issue tracker** (github.com/tommasocarraro/LTNtorch/issues, open and closed) for `empty`, `p=0`, `division`, `not in list`, `nan`, `SatAgg`.

### Details

For each bug: what it is, where in LTN, which dimension and oracle found it, and how hard it was to reach (`first_seen_exec` = number of inputs before the first hit).

**F1: Empty domain.** "For all x, P(x)" when x has 0 individuals. Logically vacuously true (1.0).

| Path | Result |
|---|---|
| Min | `IndexError` at `fuzzy_ops.py:1220` |
| Mean, pMean, pMeanError | `nan` |
| same question through a guard selecting nobody | `1.0` |

- Dimension: D2 (0 individuals) with D5 (any aggregator). Oracle: unexpected exception, and value oracle (NaN).
- First hit: input **1**. Saved as 12 signatures (`34be71331f` = Min crash; 11 NaN variants, e.g. `5e774fb287`).
- Why it matters: NaN doesn't stop a program; during training it silently breaks learning.

**F2: p = 0.** `AggregPMean(p=0)` and `AggregPMeanError(p=0)` are accepted; using them raises `ZeroDivisionError` at `fuzzy_ops.py` lines 1400, 1402 (pMean) and 1512, 1514 (pMeanError).

- Dimension: D5 (p ∈ {…, 0, …} with a pMean aggregator). Oracle: unexpected exception.
- First hit: ~**13,800** inputs (needs two choices to coincide). 4 signatures (`ac9cd02be8`, `221f71e246`, `e3cfee9f73`, `0f0760d50f`).
- The manual fuzzers also tried p = 0, but counted any exception as a pass. libFuzzer's oracle only accepts LTN's own validation messages.

**F3: Unused quantified variable.** "For all y, P(x)" raises `ValueError` (`'y' is not in list`, or `list.index(x): x not in list` on newer Python) in `Quantifier.__call__` (core.py ~line 1520). Logically it equals P(x).

- Dimension: D7 (misuse mode, harness line 321). Oracle: unexpected exception.
- Found in the misuse campaign (`findings_misuse/`).
- The message text depends on the Python version; identify the bug by exception type and code location.

**F24: Goguen implication, stable version.** `ImpliesGoguen(stable=True)(0, 0)` returns **0.0**; the unstable version returns 1.0. "False implies false" must be true (1.0).

- Dimension: D4 (operator + stable flag) with D1 values 0, 0. Oracles: reference formula **and** stable-flip metamorphic check, so two independent checks agree.
- First hit: ~**42,900** inputs. Signatures `96b8e7b1f7`, `82cd98139b`, `1ae966e95d`.

**F25: `Quantifier([])`.** Quantifying over an empty list returns a scalar (shape `()`) whose free variables are still `['x']`. Shape and metadata disagree.

- Dimension: D7. Oracle: structure check. Found in the misuse campaign.

**F26: dtype changes.** `NotGodel` turns float64 input into float32; masked `AggregMin` and guarded quantification turn float32 into float64.

- Dimension: D4/D5. Oracle: dtype check. First hits after 174, 373 and 9,195 inputs (`5d17c074af`, `0ffd0ad9bb`, `ad91e9f6c0`).

**F27: NaN gradients, unstable operators.** `AggregPMean(p=2, stable=False)` at inputs 0, 0, 0 gives gradient `[nan, nan, nan]`; with `stable=True` it is finite. Low severity, because LTN documents the stable versions as the fix.

- Dimension: D4/D5 unstable operator + D6 backward pass. Oracle: gradient check. First hit ~34,100 inputs (`b624d8c3fd`).

**F28: SatAgg's error message crashes.** `SatAgg()(tensor(0.5), tensor([0.3, 0.7]))` should raise LTN's own `ValueError` (formulas must be scalars), but the code that builds the message calls `f.shape()` on a tensor, so a `TypeError: 'torch.Size' object is not callable` is raised instead (`fuzzy_ops.py` ~line 1669).

- **Not found by the fuzzer.** Found while checking `classify_dims.py`'s output, then confirmed by hand. The current D7 never passes a raw non-scalar tensor to SatAgg, so the fuzzer cannot reach it. Report it separately from the fuzzer's results.

### Not counted as bugs

- **Docstring escape sequences** (`\l`, `\o`, `\e`, `\g`, `\d`) cause `SyntaxWarning` on Python 3.12+. Cosmetic.
- **Connective operators called directly don't check their inputs are in [0, 1].** Only `Connective(...)` checks. Arguably a design choice.
- **Negative results** (from the findings notes): no reference-formula mismatch beyond F24, no NaN on non-empty domains, no NaN gradients with stable operators.

---

## Part 3: Verify every bug yourself (no fuzzer)

Put `check_all_bugs.py` in `~/nesy_libfuzzer_ltn`, then:

```bash
cd ~/nesy_libfuzzer_ltn
source venv/bin/activate
PYTHONPATH=ltn_torch python3 check_all_bugs.py
```

**Expected (verified 10 Oct 2026 on LTNtorch d1bd981):**

```
F1  Forall, Min                         -> IndexError: amin(): Expected reduction dim 0 to have non-zero size.
    Forall, Mean / pMean / pMeanError   -> nan
    Forall, Min, guard selects nobody   -> 1.0
F2  AggregPMean(p=0) in a Forall        -> ZeroDivisionError: division by zero
    AggregPMeanError(p=0) in a Forall   -> ZeroDivisionError: division by zero
F3  Forall y, P(x)                      -> ValueError: 'y' is not in list   (wording varies by Python version)
F24 ImpliesGoguen(stable=True)(0, 0)    -> 0.0      (stable=False gives 1.0)
F25 Forall [], P(x)                     -> shape (), free variables ['x']
F26 NotGodel(float64 input).dtype       -> torch.float32
    AggregMin(float32 input, mask)      -> torch.float64
    guarded Forall (float32 data)       -> torch.float64
F27 AggregPMean(p=2, stable=False)      -> grad [nan, nan, nan]   (stable=True: finite)
F28 SatAgg(tensor(0.5), tensor([0.3, 0.7])) -> TypeError: 'torch.Size' object is not callable
```

**Observed:** ____

### Replay the fuzzer's own saved inputs

Each finding folder holds the exact input libFuzzer generated. Replay it without instrumentation:

```bash
cd ~/nesy_libfuzzer_ltn/ltn_combined
LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_valid/34be71331f/input.bin   # F1
LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_valid/221f71e246/input.bin   # F2
LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_valid/96b8e7b1f7/input.bin   # F24
```

Misuse findings (F3, F25) must be replayed in misuse mode, because the same bytes decode into a different program in each mode:

```bash
LTNFUZZ_MISUSE=1 LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_misuse/<folder>/input.bin
```

**Observed:** ____

### Count the evidence

```bash
ls findings_valid  | wc -l     # 23 signatures, 5 root causes (F1, F2, F24, F26, F27)
ls findings_misuse | wc -l     # includes F3 and F25
```

**Observed:** ____

---

## Part 4: Main lessons

1. **libFuzzer's engine works unchanged on a neurosymbolic Python library**, but only with an instrumentation front-end (Atheris), a byte-to-program decoder, and oracles.
2. **Oracles matter more than coverage here.** Coverage plateaued quickly (82–83% of LTN, `cov` flat in later runs); Round 2 added no new code coverage but found 4 more bugs, purely through stronger oracles. Most of LTN is branch-free tensor arithmetic, so coverage cannot distinguish, for example, p = 2 from p = 0.001.
3. **Dimensions decide what can be found.** The three new bugs come from dimension choices the manual fuzzers never made (0 individuals, misuse) or judged too leniently (p = 0).
4. **Bugs sit at validation gaps.** `classify_dims.py` shows F1, F2, F3 and F25 lie exactly where LTN does not validate its parameters.
5. **Combining both fuzzers works in both directions:** libFuzzer's dimensions found 3 new bugs, and the manual fuzzers' checks let libFuzzer re-find 4 known ones.

---

## Part 5: Open items

| Item | Status |
|---|---|
| Check the LTNtorch issue tracker before calling F1–F3, F28 "previously unreported" | to do |
| Confirm with the supervisor that neys_fuzz is the "fuzzing directory" | to do |
| Round 3: add the six missing dimension choices (Step 4, Check 5); goal: the fuzzer finds F28 itself | planned |
| Port the remaining manual oracles: brute-force quantifier, LTN vs LNN, finite-difference gradients | planned |
| Next library: DeepProbLog, then NeurASP and LNN | not started |

---

## Text for the report

> libFuzzer could not be applied to LTN as is. LTN is a Python library and most of its bugs are wrong values rather than crashes, while libFuzzer relies on compile-time instrumentation of C/C++ and on sanitizers. Keeping the libFuzzer engine and coverage-only guidance unchanged, we made seven adaptations: Atheris for Python instrumentation, instrumentation limited to the `ltn` package, decoding of libFuzzer's bytes into LTN programs along seven dimensions, value/formula/metamorphic/dtype/structure/gradient oracles, keep-going de-duplication of failures, separation of harness bugs from LTN bugs, and chunked campaigns to contain Atheris' memory growth. On LTNtorch 1.0.2, about one million executions found three new bugs (inconsistent results on empty domains, including silent NaN; an unvalidated p = 0 causing division by zero; a crash when quantifying a variable absent from the formula) and re-found the four bugs previously found by manual fuzzers (a wrong truth value in the stable Goguen implication, inconsistent metadata for empty quantification, dtype changes, and NaN gradients in unstable operators). An eighth bug, a crash in SatAgg's own error message, was found by code inspection. All eight reproduce in standalone scripts without the fuzzer.
