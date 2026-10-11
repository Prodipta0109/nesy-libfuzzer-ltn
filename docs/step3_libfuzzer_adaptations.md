# Step 3 verification: the seven adaptations libFuzzer needed for LTN

**Requirement (supervisor):** ask Claude to adapt the code of libFuzzer if needed for these types of libraries.

**Answer:** yes, adaptation was needed. libFuzzer was built for C/C++ programs and finds memory-safety crashes. LTN is a Python library whose bugs are mostly wrong numbers, not crashes. Seven adaptations were made. The libFuzzer **engine** (mutate, run, keep inputs by coverage) was never changed; everything around it was.

**Verified on:** ____ Oct 2026, WSL (Ubuntu), folder `~/nesy_libfuzzer_ltn`

---

## Overview

| # | Adaptation | Problem with plain libFuzzer | What was done |
|---|---|---|---|
| A1 | Language | Only measures coverage of compiled C/C++ | Atheris: libFuzzer engine + Python coverage |
| A2 | Selective instrumentation | Would measure PyTorch too (huge, slow, not the target) | Coverage only for the `ltn` package |
| A3 | Input decoding | Produces raw bytes, meaningless to LTN | Bytes are read as choices that build an LTN program |
| A4 | Oracle | Only detects crashes (via sanitizers); Python has none, and LTN's bugs are often silent | Value, formula, metamorphic, dtype, structure and gradient checks |
| A5 | Keep-going mode | Stops at the first failure | Failures de-duplicated by signature, saved, fuzzing continues |
| A6 | Harness-bug control | A failure in the fuzzer's own code looks like an LTN bug | Harness false alarms identified and fixed; every bug confirmed outside the fuzzer |
| A7 | Memory | Atheris grows memory per execution, causing fake slow inputs | Campaigns run in short chunks that resume from the corpus |

```
plain libFuzzer:   C/C++ program  ── clang instrumentation ──> engine ── sanitizer ──> crash report, stop
adapted:           LTN (Python)   ── Atheris, ltn only (A1,A2) ──> engine (unchanged)
                     ↑ bytes decoded into an LTN program (A3)
                     ↓ oracles judge the result (A4), failures saved, run continues (A5)
                     harness bugs filtered (A6), runs split into chunks (A7)
```

Setup for all commands:

```bash
cd ~/nesy_libfuzzer_ltn
source venv/bin/activate
```

---

## A1. Language: Atheris runs libFuzzer on Python

**Problem.** libFuzzer gets its coverage from the compiler. You build a C/C++ program with `clang -fsanitize=fuzzer`, and clang inserts a counter at every branch. LTN is Python: there is nothing to compile, so no counters, so libFuzzer would be blind.

**Example (optional, for contrast).** In Google's libFuzzer tutorial, a C++ target is compiled with clang before it can be fuzzed. This runs in the tutorial folder, **not** in `nesy_libfuzzer_ltn`:

```bash
find ~ -name fuzz_me.cc 2>/dev/null          # tutorial file from your earlier libFuzzer setup
# if nothing is found, get the tutorial:
git clone --depth 1 https://github.com/google/fuzzing.git ~/fuzzing
cd ~/fuzzing/tutorial/libFuzzer
clang++ -g -fsanitize=address,fuzzer fuzz_me.cc -o fuzz_me
./fuzz_me
cd ~/nesy_libfuzzer_ltn
```

**Expected:** within a second, AddressSanitizer reports `heap-buffer-overflow` and libFuzzer stops and writes a `crash-…` file. Note the two things LTN lacks: a **compile step** (where clang inserts the coverage counters) and a **sanitizer** (which detects the bug). A1 replaces the first; A4 replaces the second.

There is no equivalent compile step for `ltn/core.py`.

**What was done.** Atheris contains the real libFuzzer engine. Instead of clang, it inserts the counters into Python **bytecode** when the module is imported. The engine reads those counters exactly as it would read clang's.

**Verify:**

```bash
pip show atheris | head -2
grep -n "import atheris\|atheris.Setup\|atheris.Fuzz" ltn_combined/ltn_fuzz_combined.py
head -4 ltn_combined/logs/valid.log
```

**Expected:**
```
Name: atheris
Version: 3.1.0
46:import atheris
563:    atheris.Setup(sys.argv, TestOneInput)
564:    atheris.Fuzz()
INFO: Instrumenting ltn
INFO: Instrumenting ltn.core
INFO: Instrumenting ltn.fuzzy_ops
INFO: Using built-in libfuzzer
```

The fuzzer is started with `python3`, not compiled, and the log shows the libFuzzer engine running.

**Observed:** ____

---

## A2. Selective instrumentation: only LTN is measured

**Problem.** If everything imported were instrumented, PyTorch's thousands of Python files would get counters too. Three consequences: very slow startup and execution; coverage dominated by PyTorch code, so libFuzzer would chase new PyTorch paths instead of LTN paths; and PyTorch is not the system under test.

**What was done.** PyTorch is imported **first**, normally. Then LTN is imported inside `atheris.instrument_imports(include=["ltn"])`, so only modules whose names start with `ltn` get counters.

**Verify:**

```bash
sed -n 40,70p ltn_combined/ltn_fuzz_combined.py
```

**Expected:** `import torch` appears **above** the line

```python
with atheris.instrument_imports(include=["ltn"]):
```

and `import ltn` appears **inside** that block.

Confirm in the log that only LTN modules were instrumented:

```bash
grep "Instrumenting" ltn_combined/logs/valid.log
```

**Expected:**
```
INFO: Instrumenting ltn
INFO: Instrumenting ltn.core
INFO: Instrumenting ltn.fuzzy_ops
```

No `torch` lines.

**Observed:** ____

---

## A3. Input decoding: bytes become an LTN program

**Problem.** libFuzzer generates byte strings such as `03 00 ff 41 …`. LTN cannot run bytes; it needs variables, predicates, connectives and quantifiers.

**What was done.** The harness reads the bytes **as a sequence of choices** with Atheris' `FuzzedDataProvider`. Each choice belongs to one dimension:

| Bytes read | Choice | Dimension |
|---|---|---|
| next byte | how many individuals in x (0–5) | D2 shape |
| next 4 bytes | a feature value | D1 values |
| next byte | which predicate model | D3 |
| next byte | which connective (And, Or, Implies, …) | D4 formula |
| next byte | which aggregator, which p | D5 quantifiers |
| … | SatAgg, backward pass | D6 |
| (misuse mode) | quantify an unused variable, `Quantifier([])` | D7 |

Because every byte string decodes into **some valid program**, libFuzzer's blind byte mutations become program mutations: changing one byte changes one choice.

**Verify: where the decoding happens**

```bash
grep -n "FuzzedDataProvider\|ConsumeIntInRange\|ConsumeBool\|ConsumeProbability\|PickValueInList" \
    ltn_combined/ltn_fuzz_combined.py | head -30
```

**Observed (10 Oct 2026):**
```
136:        self.fdp = atheris.FuzzedDataProvider(data)
151:        return self.fdp.ConsumeIntInRange(lo, hi)
154:        return self.fdp.ConsumeBool()
455:        self.fdp = atheris.FuzzedDataProvider(data)
459:        return self.fdp.ConsumeIntInRange(lo, hi)
471:        if "stable" in params and self.fdp.ConsumeBool():
497:                d = DELTAS[self.i(0, len(DELTAS) - 1)] * (1 if self.fdp.ConsumeBool() else -1)
502:            m = torch.tensor([self.fdp.ConsumeBool() for _ in range(args[0].numel())],
```

Two decoders, one per mode: **program mode** (136, building whole LTN programs) and **seed mode** (455, mutating one of the 105 operator calls recorded from LTN's tests). In each, the raw `Consume…` calls are wrapped in short helper methods (lines 151 and 154 are inside them; 459 for seed mode), and every choice in the builder goes through those helpers. That's why only a few `Consume…` lines appear. To see the helpers and count the choices made through them:

```bash
sed -n 136,160p ltn_combined/ltn_fuzz_combined.py
grep -c "self\.i(" ltn_combined/ltn_fuzz_combined.py
```

Seed-mode examples visible above: line 471 decides whether to flip the `stable` flag, 497 perturbs a value up or down by a chosen step, and 502 builds a random mask (which individuals a guard selects).

**Verify: an actual example, bytes and the program built from them**

Each saved finding keeps both. Use the F1 empty-domain finding:

```bash
xxd ltn_combined/findings_valid/34be71331f/input.bin | head
cat ltn_combined/findings_valid/34be71331f/program.txt
```

**Expected:** the first command shows a few raw bytes; the second shows the LTN program those bytes decoded into, containing a variable with **0 individuals** and a `Min` quantifier. That program is what triggered `IndexError@fuzzy_ops.py:1220`.

**Observed:** ____

---

## A4. Oracle: deciding what counts as a bug

**Problem.** libFuzzer reports a bug only when the program **crashes**, usually detected by AddressSanitizer. Two problems for LTN:

1. Python has no sanitizer. The log warns about it:
   ```
   WARNING: Failed to find function "__sanitizer_acquire_crash_state".
   ```
2. Many LTN bugs **don't crash**: they return a wrong number. Example, F1 with Mean: "for all x, P(x)" over an empty domain returns `nan`. Without an oracle, libFuzzer sees a normal run and moves on.

On top of that, some exceptions are **correct behaviour**: LTN raising `ValueError: Expected inputs … in the range [0., 1.]` for a bad input is validation working, not a bug.

**What was done.** The harness judges every result:

| Oracle | Detects | Example finding |
|---|---|---|
| Unexpected exception | any exception **except** LTN's own documented validation errors | F2 `ZeroDivisionError`, F3 `ValueError: … not in list` |
| Value | final truth value NaN or outside [0, 1] with finite inputs | F1 `nan` for Mean / pMean / pMeanError |
| Formula | LTN's result differs from the documented formula | F24 `ImpliesGoguen(stable=True)` |
| Metamorphic | swapping operands, permuting individuals, or flipping `stable` changes the result | F24 |
| Dtype | output type differs from input type | F26 |
| Structure | result shape doesn't match the free variables | F25 |
| Gradient | NaN/inf gradient with finite inputs | F27 |

**Verify: the oracle classes exist**

```bash
grep -n "class .*Violation\|OracleViolation\|FormulaViolation\|MetamorphicViolation\|DtypeViolation\|StructureViolation\|GradientViolation" \
    ltn_combined/ltn_fuzz_combined.py | head -20
```

**Verify: LTN's own validation errors are filtered out**

```bash
grep -n "Expected inputs of connectives\|Expected parameter\|VALIDATION\|validation" \
    ltn_combined/ltn_fuzz_combined.py | head -20
```

**Expected:** a list of LTN's validation message prefixes; exceptions starting with these are not reported.

**Observed (10 Oct 2026):**
```
96:class OracleViolation(Exception):
100:class GradientViolation(Exception):
358:            raise K.StructureViolation(f"{what}: value shape {shape} but free variables {fv} "
364:            raise K.DtypeViolation(f"Quantifier({'guarded' if guarded else 'unguarded'}): returns "
414:                raise OracleViolation(f"closed formula has truth value {v.tolist()} outside [0,1] "
422:            raise OracleViolation(f"SatAgg returned {sat.item()} with finite inputs")
435:                    raise GradientViolation(
439:                    raise GradientViolation(
511:            raise OracleViolation(f"{op!r} returned values outside [0,1]: {o.tolist()}")
```

| Line | Check | Finding it produced |
|---|---|---|
| 358 | formula's shape doesn't match its free variables | F25 (misuse mode) |
| 364 | quantifier returns the wrong dtype | F26 `DtypeViolation: Quantifier(guarded)` |
| 414 | closed formula is NaN or outside [0, 1] | F1 (11 NaN signatures) |
| 422 | SatAgg result not finite although inputs were | none observed |
| 435, 439 | NaN/inf gradient (stable vs unstable operators) | F27 |
| 511 | seed mode: an operator returns values outside [0, 1] | none observed |

`OracleViolation` and `GradientViolation` are defined in the harness itself. `StructureViolation` and `DtypeViolation` come from a module imported as `K`: the checks ported from neys_fuzz in Round 2, where `FormulaViolation` and `MetamorphicViolation` also live. To list all of them:

```bash
grep -n "class .*Violation\|^import\|^from\| as K" ltn_combined/*.py
```

**Observed (validation filter, 10 Oct 2026):**
```
25:  - any exception that is NOT one of LTN's own documented input-validation errors
82:# Messages raised by LTN's own validation code. Exceptions with these prefixes
86:    "Expected inputs of connectives and quantifiers",
88:    "Expected parameter", "Expected the given LTN variables",
```

To see the full list of accepted LTN messages:

```bash
sed -n 80,95p ltn_combined/ltn_fuzz_combined.py
```

**Verify: a bug only the oracle could catch**

```bash
cat ltn_combined/findings_valid/5e774fb287/meta.json
```

**Expected:**
```
"signature": "OracleViolation@ltn_fuzz_combined.py:414: closed formula has truth value nan outside [N,N] [empty_domain=True aggs=['Mean']]"
```

LTN didn't crash here; it returned `nan`. Plain libFuzzer would have counted this input as a normal run.

**Observed:** ____

---

## A5. Keep-going mode: don't stop at the first failure

**Problem.** By default libFuzzer stops at the first crash and writes one `crash-…` file. On LTN, the F1 empty-domain crash appears within the **first input**, so a default run would stop immediately and never reach F2, F24 or anything else.

**What was done.** With `LTNFUZZ_COLLECT=1`, the harness catches each failure, computes a **signature** (failure type + code location, plus details), saves the first example per signature under `findings_*/<id>/`, counts repeats, and lets libFuzzer continue.

**Verify: how it is switched on**

```bash
grep -n "COLLECT" ltn_combined/ltn_fuzz_combined.py ltn_combined/run_campaign.sh
```

**Expected:** `LTNFUZZ_COLLECT` read in the harness and set to `1` in `run_campaign.sh`.

**Verify: the result, many distinct failures from one campaign**

```bash
ls ltn_combined/findings_valid | wc -l
grep -h '"count"' ltn_combined/findings_valid/*/meta.json | sort -t: -k2 -n -r | head -3
```

**Expected:** 23 signatures, and repeat counts in the thousands. For example, the F1 Min crash was hit 134,864 times but saved once. A default run would have ended at the first of those.

**Optional demonstration: default behaviour without keep-going**

```bash
cd ~/nesy_libfuzzer_ltn/ltn_combined
mkdir -p /tmp/demo_corpus
LTNFUZZ_COLLECT=0 timeout 60 python3 ltn_fuzz_combined.py /tmp/demo_corpus
ls crash-* 2>/dev/null
```

**Expected:** the run stops after a few inputs with an uncaught Python exception, and a `crash-…` file is written. Delete it afterwards with `rm crash-*`.

**Observed:** ____

---

## A6. Harness-bug control: is it LTN's bug or the fuzzer's?

**Problem.** The harness is itself code that can have bugs. A crash in the harness looks exactly like a crash in LTN. Two early alarms were harness bugs:

1. the harness **reshaped** a tensor with 0 individuals incorrectly, so the "crash" was in the harness, not LTN
2. a **test predicate overflowed** for huge inputs, so the "NaN gradient" came from the harness's own function

Both were fixed before the reported campaigns.

**What was done.** Two rules for every reported bug:

1. the failure location must be **inside LTN** (`ltn/core.py`, `ltn/fuzzy_ops.py`), or an oracle reporting a wrong LTN result
2. the bug must **reproduce without the fuzzer**, in a small standalone script

**Verify rule 1: where each failure happened**

```bash
grep -h '"signature"' ltn_combined/findings_*/*/meta.json | sed 's/.*"signature": "//' | cut -c1-60 | sort | uniq
```

**Expected:** exceptions located in `fuzzy_ops.py` or `core.py` (LTN's files), and oracle violations (`OracleViolation@ltn_fuzz_combined.py:414`, `FormulaViolation`, …). An oracle violation's location is the harness line where the **check** fired, not a harness crash. No unexpected exception located in the harness.

**Verify rule 2: reproduction without the fuzzer**

```bash
PYTHONPATH=ltn_torch python3 check_f1_f2_f3.py
```

**Expected (verified 10 Oct 2026):** F1 gives `IndexError` for Min, `nan` for Mean/pMean/pMeanError and `1.0` through a guard; F2 gives `ZeroDivisionError` with p = 0; F3 gives `ValueError: … not in list`. No fuzzer code is involved.

**Also: replay a saved finding without instrumentation**

```bash
cd ltn_combined
LTNFUZZ_NO_INSTRUMENT=1 python3 repro.py findings_valid/221f71e246/input.bin
```

**Expected:** the F2 `ZeroDivisionError` again.

**Observed:** ____

---

## A7. Memory: run in chunks

**Problem.** With Atheris' Python instrumentation active, memory grows by several KB **per execution** (Atheris, not LTN). Over a long run this reached about 3 GB in 20 minutes and caused **fake slow inputs**: libFuzzer reported inputs taking 25–34 s that replay in 0.01 s.

**Evidence from your own run (10 Oct 2026, 1 minute):**

```
#4096   pulse  … rss: 314Mb
#24527  DONE   … rss: 461Mb
```

That is +147 MB over about 20,400 executions, roughly 7 KB per execution.

**What was done.** Long campaigns are split into short chunks. Each chunk restarts the process (memory back to normal) and libFuzzer **resumes from the saved corpus**, so no progress is lost.

**Verify: the chunk script**

```bash
cat ltn_combined/run_chunks.sh
```

**Expected:**
```bash
# usage: run_chunks.sh <name> <misuse 0|1> <chunks> <secs-per-chunk>
# libFuzzer resumes from the corpus each chunk; restarting avoids Atheris' memory growth.
for c in $(seq 1 $3); do
  ORACLES=${ORACLES:-all} $D/run_campaign.sh $1 $2 $4 $((100+c))
  ...
```

**Verify: memory per chunk stays bounded**

```bash
for f in ltn_combined/logs/valid.chunk*.log; do
  printf "%-20s " "$(basename $f)"
  grep -o "rss: [0-9]*Mb" "$f" | sed -n '1p;$p' | tr '\n' ' '; echo
done
```

**Expected:** each chunk starts low again and ends at a few hundred MB, instead of one run climbing to gigabytes.

**Verify: memory growth comes from instrumentation, not LTN**

```bash
cd ltn_combined
/usr/bin/time -v python3 repro.py findings_valid/221f71e246/input.bin 2>&1 | grep "Maximum resident"
LTNFUZZ_NO_INSTRUMENT=1 /usr/bin/time -v python3 repro.py findings_valid/221f71e246/input.bin 2>&1 | grep "Maximum resident"
```

**Expected:** similar values for a single replay. The growth only builds up over thousands of instrumented executions in one process.

**Observed:** ____

---

## Side-by-side: plain libFuzzer (C++) vs adapted libFuzzer (LTN)

Observed on 10 Oct 2026, same machine. Left: `./fuzz_me` from Google's tutorial. Right: `ltn_combined/logs/valid.log`.

| What to compare | Plain libFuzzer on `fuzz_me.cc` | libFuzzer on LTN | Adaptation |
|---|---|---|---|
| Where coverage counters come from | `Loaded 1 modules (7 inline 8-bit counters)`: inserted by clang at **compile time** | `Instrumenting ltn / ltn.core / ltn.fuzzy_ops`: inserted by Atheris at **import time** | A1, A2 |
| What the input is | raw bytes used directly: the crash input is `0x46,0x55,0x5a` = `"FUZ"` | raw bytes **decoded** into an LTN program (`input.bin` → `program.txt`) | A3 |
| Who detects the bug | AddressSanitizer: `ERROR: AddressSanitizer: heap-buffer-overflow … fuzz_me.cc:9:7` | no sanitizer (`WARNING: Failed to find function "__sanitizer_…"`); the harness oracles detect it | A4 |
| What happens at the first bug | `==ABORTING`, `Test unit written to ./crash-0eb8e4…`: **stops** | failure saved under `findings_valid/<id>/`, **continues** (23 signatures in one campaign) | A5 |
| Memory | `rss: 32Mb` → `39Mb` over ~92,000 executions (flat) | `rss: 314Mb` → `461Mb` over ~20,000 executions (grows) | A7 |
| Coverage growth | `cov: 3 → 4 → 5 → 6 → 7`; every `NEW` adds coverage | same rule; plateaued at `cov: 339` | (Step 2) |

Two more details visible in the C++ run:

- **The bug:** `fuzz_me.cc` reads `Data[3]` when the input starts with `"FUZ"` but is only 3 bytes long, so it reads 1 byte past the end of the buffer. AddressSanitizer catches this invalid read. Without ASan the program might just continue, and libFuzzer would never know. The same blindness is why LTN needs oracles.
- **Comparison tracing:** `MS: … CMP- DE: "F\000\000\000"` means libFuzzer saw the program compare a byte with `'F'` and inserted `'F'` into the input. That's how it guessed `F`, `U`, `Z` in about 92,000 tries instead of millions. This is the default comparison feedback mentioned in Step 2; it is part of coverage guidance, not an extra signal.

---

## Summary

| # | Adaptation | Evidence | Result |
|---|---|---|---|
| A1 | Atheris runs libFuzzer on Python | `Using built-in libfuzzer`, atheris 3.1.0 | ____ |
| A2 | Only `ltn` instrumented | `Instrumenting ltn / ltn.core / ltn.fuzzy_ops`, torch imported before | ____ |
| A3 | Bytes decoded into LTN programs | `FuzzedDataProvider`, `input.bin` vs `program.txt` | ____ |
| A4 | Oracles for silent bugs | F1 `nan` reported by `OracleViolation` | ____ |
| A5 | Keep-going mode | 23 signatures from one campaign; counts in the thousands | ____ |
| A6 | Harness bugs separated from LTN bugs | failures in LTN files; F1–F3 reproduce without the fuzzer | ____ |
| A7 | Chunked campaigns | memory +~7 KB/exec; chunks resume from corpus | ____ |

### Text for the report

> libFuzzer could not be applied to LTN directly. We made seven adaptations: (1) Atheris runs the unmodified libFuzzer engine on Python by instrumenting bytecode instead of compiled code; (2) only the `ltn` package is instrumented, so coverage reflects LTN and not PyTorch; (3) libFuzzer's raw bytes are decoded into structured LTN programs along seven dimensions; (4) since Python has no sanitizer and many LTN bugs are silent wrong values, the harness provides value, formula, metamorphic, dtype, structure and gradient oracles, and ignores LTN's own validation errors; (5) failures are de-duplicated by signature and fuzzing continues instead of stopping at the first crash; (6) harness false alarms were identified and fixed, and every reported bug was reproduced in a standalone script; (7) because Atheris' instrumentation grows memory per execution, campaigns run in short chunks that resume from the saved corpus.
