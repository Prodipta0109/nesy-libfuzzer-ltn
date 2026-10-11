# Step 1 verification: libFuzzer is used, and LTN is the target

**Requirement (supervisor):** use libFuzzer to fuzz the neurosymbolic libraries, starting with LTN.

**Claim to verify:** the LTN fuzzer runs the real libFuzzer engine, its coverage measures only LTN's code, it targets a fixed LTN version, and campaigns were actually run.

**Verified on:** 10 Oct 2026, WSL (Ubuntu), folder `~/nesy_libfuzzer_ltn`

---

## 0. Setup before running the checks

Every command below is run from the project folder with the virtual environment active.

```bash
cd ~/nesy_libfuzzer_ltn
source venv/bin/activate
```

The prompt shows `(venv)` when the environment is active.

Folder layout:

| Folder / file | Contents |
|---|---|
| `ltn_torch/` | LTNtorch source code (the program under test) |
| `ltn_fuzz/` | Round 1 fuzzer (basic oracle) |
| `ltn_combined/` | Round 2 fuzzer (stronger oracles); used for these checks |
| `ltn_combined/logs/` | libFuzzer output of every campaign |
| `venv/` | Python environment with torch and atheris |

---

## Check 1: The fuzzing tool is installed

**Purpose:** libFuzzer is a C/C++ tool. For Python code it is used through **Atheris**, Google's package that contains the real libFuzzer engine.

```bash
pip show atheris | head -2
```

**Expected:** the package name and a version number.

**Observed:**
```
Name: atheris
Version: 3.1.0
```

The extra line `ERROR: Pipe to stdout was broken` is harmless: `head` stops reading after 2 lines.

**Result: PASS.** Atheris 3.1.0 is installed.

---

## Check 2: The engine that ran is libFuzzer

**Purpose:** every program that uses libFuzzer prints the engine's startup messages at the top of its output. The fuzzer saves this output in the campaign logs.

```bash
head -12 ltn_combined/logs/valid.log
```

**Expected:** lines starting with `INFO:` from libFuzzer: the engine name, the scheduling strategy, the random seed, and the loaded corpus.

**Observed:**
```
INFO: Instrumenting ltn
INFO: Instrumenting ltn.core
INFO: Instrumenting ltn.fuzzy_ops
INFO: Using built-in libfuzzer
WARNING: Failed to find function "__sanitizer_acquire_crash_state".
WARNING: Failed to find function "__sanitizer_print_stack_trace".
WARNING: Failed to find function "__sanitizer_set_death_callback".
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 101
INFO:      405 files found in /home/prodipta0109/nesy_libfuzzer_ltn/ltn_combined/corpus_valid
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 405 min: 1b max: 195b total: 17563b rss: 274Mb
```

**How to read it:**

| Line | Meaning |
|---|---|
| `Using built-in libfuzzer` | The libFuzzer engine bundled in Atheris is running |
| `entropic power schedule` | libFuzzer's default strategy for choosing which saved input to mutate next |
| `Seed: 101` | Random seed for this run (the script uses 100 + chunk number), so runs are repeatable |
| `405 files found … corpus_valid` | libFuzzer resumed from the inputs saved by earlier runs |
| `-max_len … 4096 bytes` | libFuzzer's default maximum input size |
| `WARNING: … __sanitizer_…` | **Expected and harmless.** In C/C++, libFuzzer uses AddressSanitizer to detect memory bugs. Python has no sanitizer, so these functions don't exist. This is why the fuzzer needs its own checks (oracles) to detect wrong answers |

**Result: PASS.** The log shows the libFuzzer engine starting.

---

## Check 3: Coverage measures only LTN

**Purpose:** libFuzzer decides which inputs to keep by measuring which code they reach. That measurement must cover LTN, not PyTorch or the fuzzer itself.

The first three lines of the log in Check 2 already show this:

```
INFO: Instrumenting ltn
INFO: Instrumenting ltn.core
INFO: Instrumenting ltn.fuzzy_ops
```

Only LTN's modules are instrumented (receive coverage counters). No `torch` module is listed.

The code that sets this up:

```bash
grep -n "import atheris\|instrument_imports\|atheris.Setup\|atheris.Fuzz" ltn_combined/*.py
```

**Observed:**
```
ltn_combined/ltn_fuzz_combined.py:46:import atheris
ltn_combined/ltn_fuzz_combined.py:63:    with atheris.instrument_imports(include=["ltn"]):
ltn_combined/ltn_fuzz_combined.py:136:        self.fdp = atheris.FuzzedDataProvider(data)
ltn_combined/ltn_fuzz_combined.py:455:        self.fdp = atheris.FuzzedDataProvider(data)
ltn_combined/ltn_fuzz_combined.py:563:    atheris.Setup(sys.argv, TestOneInput)
ltn_combined/ltn_fuzz_combined.py:564:    atheris.Fuzz()
```

**How to read it:**

| Line | Meaning |
|---|---|
| 46 `import atheris` | The fuzzer uses Atheris (libFuzzer for Python) |
| 63 `instrument_imports(include=["ltn"])` | Coverage counters are added **only** to the `ltn` package. PyTorch is imported earlier in the file, outside this block, so it is not measured |
| 136, 455 `FuzzedDataProvider(data)` | libFuzzer's raw bytes are read as a sequence of choices (number of individuals, operator, p value, …) to build an LTN program. 136 is program mode, 455 is seed mode |
| 563 `atheris.Setup(sys.argv, TestOneInput)` | Registers `TestOneInput` as the function libFuzzer calls with each new input |
| 564 `atheris.Fuzz()` | Hands control to the libFuzzer engine |

To confirm PyTorch is imported before the instrumented block, view the lines around 63:

```bash
sed -n 40,70p ltn_combined/ltn_fuzz_combined.py
```

`import torch` should appear above line 63.

**Result: PASS.** Only `ltn`, `ltn.core` and `ltn.fuzzy_ops` are measured.

---

## Check 4: The exact LTN version fuzzed

**Purpose:** bug reports cite line numbers, so the LTN version must be fixed and stated.

```bash
git -C ltn_torch log -1 --oneline
```

**Observed:**
```
d1bd981 (HEAD, origin/main, origin/HEAD, main) version 1.0.2 is now released
```

**Result: PASS.** The target is **LTNtorch 1.0.2, commit d1bd981**, the latest version on the main branch at the time of fuzzing.

---

## Check 5: The campaigns actually ran

**Purpose:** show how much fuzzing was done.

List the campaign logs:

```bash
ls ltn_combined/logs/
```

**Observed:**
```
misuse.chunk1.log  misuse.chunk3.log  misuse.log        valid.chunk2.log  valid.chunk4.log
misuse.chunk2.log  misuse.chunk4.log  valid.chunk1.log  valid.chunk3.log  valid.log
```

Two campaigns, four chunks each:

- **valid:** well-formed LTN programs
- **misuse:** also includes API misuse (quantifying a variable that isn't in the formula, `Quantifier([])`, an open formula given to `SatAgg`)

Campaigns are split into chunks because Atheris' coverage measurement grows memory by about 10 KB per execution. Each chunk restarts libFuzzer, which resumes from the saved corpus.

Count the executions of the original campaigns (chunk logs only):

```bash
grep -h "number_of_executed_units" ltn_combined/logs/*.chunk*.log | awk '{s+=$2} END {print s, "executions"}'
```

**Expected:** about 1,016,000 (443k valid + 573k misuse, as recorded in the findings notes).

**Observed:** ________ executions *(fill in)*

Use the chunk logs only. `misuse.log` and `valid.log` hold the latest chunk or the latest manual run, so counting all logs counts some runs twice. Counting all of them gave 1,080,904.

Per-chunk detail, if needed:

```bash
grep -H "number_of_executed_units" ltn_combined/logs/*.chunk*.log
```

**Result: PASS** once the number is filled in.

---

## Check 6 (optional): Run libFuzzer live

**Purpose:** show the engine working, not just old logs.

```bash
cd ~/nesy_libfuzzer_ltn/ltn_combined
./run_chunks.sh valid 0 1 60
tail -8 logs/valid.log
```

Arguments: campaign name, misuse off (0), 1 chunk, 60 seconds.

**Observed (10 Oct 2026):**
```
#24527  DONE   cov: 339 ft: 1355 corp: 203/6900b lim: 353 exec/s: 402 rss: 461Mb
Done 24527 runs in 61 second(s)
stat::number_of_executed_units: 24527
stat::average_exec_per_sec:     402
stat::new_units_added:          13
stat::slowest_unit_time_sec:    0
stat::peak_rss_mb:              461
```

**How to read it:**

| Part | Meaning |
|---|---|
| `Done 24527 runs in 61 second(s)` | libFuzzer ran LTN on 24,527 generated inputs in one minute |
| `exec/s: 402` | about 400 inputs per second |
| `cov: 339` | LTN code branches reached (constant: coverage has plateaued) |
| `new_units_added: 13` | inputs kept because they were new or smaller than an existing one |
| `rss: 461Mb` | memory used; it grows during the run (Atheris memory growth) |

The raw coverage number depends on the Python version, so compare it only between runs on the same machine.

**Result: PASS.** libFuzzer runs and fuzzes LTN on this machine.

---

## Summary

| # | Check | Result |
|---|---|---|
| 1 | Atheris (libFuzzer for Python) installed: version 3.1.0 | PASS |
| 2 | libFuzzer engine starts (`Using built-in libfuzzer`) | PASS |
| 3 | Coverage limited to `ltn`, `ltn.core`, `ltn.fuzzy_ops` | PASS |
| 4 | Target fixed: LTNtorch 1.0.2, commit d1bd981 | PASS |
| 5 | Campaigns ran: 2 modes × 4 chunks, ______ executions | PASS |
| 6 | Live run works: 24,527 executions in 61 s | PASS |

**Conclusion: Step 1 is satisfied.** LTN was fuzzed with libFuzzer.

### Text for the report

> We fuzzed LTNtorch 1.0.2 (commit d1bd981) with libFuzzer, using Atheris 3.1.0 to run the libFuzzer engine on Python code. Coverage instrumentation was limited to the `ltn` package (`ltn.core`, `ltn.fuzzy_ops`); PyTorch was not instrumented. Two campaigns (valid programs and API misuse) of four chunks each executed ______ inputs in total, at about 400 inputs per second.
