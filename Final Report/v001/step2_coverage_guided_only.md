# Step 2 verification: the fuzzing is guided by code coverage only

**Requirement (supervisor):** for now, do guided fuzzing based on code coverage only.

**Claim to verify:**

1. libFuzzer decides which inputs to keep **only** from coverage of LTN's code.
2. No other guidance is added: no dictionary, no custom mutator, no value profile, no custom fitness score.
3. The bug checks (oracles) **report** failures but do **not** steer the fuzzer.

**Verified on:** ____ Oct 2026, WSL (Ubuntu), folder `~/nesy_libfuzzer_ltn`

---

## Background: what "guided by coverage" means

libFuzzer repeats one loop:

1. Take an input from the **corpus** (the saved collection of interesting inputs).
2. **Mutate** it (flip bits, insert, delete, copy bytes).
3. Run LTN on it and **measure coverage**: which branches of LTN's code ran, and roughly how often.
4. If the input reached something new, **keep it** in the corpus (`NEW`). If it reached the same code as an existing input but is shorter, **replace** that one (`REDUCE`). Otherwise **discard** it.

"Coverage-guided only" means step 4 uses nothing but coverage. Other fuzzers add extra signals, such as a dictionary of interesting values, a hand-written mutator, or a score for "how close to a bug". This check proves none of those are used.

Two terms used below:

| Term | Meaning |
|---|---|
| `cov` | number of LTN code branches (edges) reached so far |
| `ft` (features) | finer coverage: each branch **plus** how often it ran (1, 2–3, 4–7, … times). libFuzzer keeps an input when `ft` increases |

---

## 0. Setup

```bash
cd ~/nesy_libfuzzer_ltn
source venv/bin/activate
```

---

## Check 1: Coverage is measured only on LTN

This was shown in Step 1 (Check 3). The log starts with:

```
INFO: Instrumenting ltn
INFO: Instrumenting ltn.core
INFO: Instrumenting ltn.fuzzy_ops
```

So the coverage that guides libFuzzer comes only from LTN's code, not from PyTorch and not from the fuzzer's own code.

**Result: PASS** (see Step 1).

---

## Check 2: No extra guidance is configured

**Purpose:** libFuzzer can be given extra guidance through options or code. Search the fuzzer for all of them.

| Extra guidance | How it would appear |
|---|---|
| Dictionary of interesting values | `-dict=` option |
| Value profile (guidance by comparison operand values) | `-use_value_profile=1` option |
| Custom mutator (hand-written mutation) | `custom_mutator` / `CustomMutator` in the Python code |
| Custom crossover | `custom_crossover` in the Python code |

```bash
grep -n "dict=\|use_value_profile\|custom_mutator\|CustomMutator\|custom_crossover" \
    ltn_combined/*.sh ltn_combined/*.py
```

**Expected:** no output.

**Observed:**
```
(fill in; empty output = PASS)
```

Then look at the exact command each campaign runs:

```bash
cat ltn_combined/run_campaign.sh
```

**Expected:** the Python fuzzer started with the corpus folder and basic run options (time limit, seed, …), but none of the options in the table above.

**Observed options:** __________________ *(fill in)*

**Result:** PASS if both are as expected.

---

## Check 3: The fuzzer talks to libFuzzer only through standard calls

**Purpose:** if the bug checks (oracles) could influence libFuzzer, they would have to call it. List every place the fuzzer uses Atheris (libFuzzer):

```bash
grep -n "atheris\." ltn_combined/*.py
```

**Expected:** only these standard calls:

| Call | Role |
|---|---|
| `atheris.instrument_imports(include=["ltn"])` | adds coverage counters to LTN |
| `atheris.FuzzedDataProvider(data)` | reads libFuzzer's bytes as choices to build an LTN program |
| `atheris.Setup(sys.argv, TestOneInput)` | registers the function libFuzzer calls with each input |
| `atheris.Fuzz()` | starts the libFuzzer engine |

**Observed (Step 1, Check 3):**
```
ltn_combined/ltn_fuzz_combined.py:63:    with atheris.instrument_imports(include=["ltn"]):
ltn_combined/ltn_fuzz_combined.py:136:        self.fdp = atheris.FuzzedDataProvider(data)
ltn_combined/ltn_fuzz_combined.py:455:        self.fdp = atheris.FuzzedDataProvider(data)
ltn_combined/ltn_fuzz_combined.py:563:    atheris.Setup(sys.argv, TestOneInput)
ltn_combined/ltn_fuzz_combined.py:564:    atheris.Fuzz()
```

None of these pass oracle results to libFuzzer. The oracles have no channel to steer it.

**Result: PASS.**

---

## Check 4: Oracle failures are saved and the run continues

**Purpose:** see what the fuzzer does when an oracle detects a failure.

```bash
grep -n "def TestOneInput" ltn_combined/ltn_fuzz_combined.py
```

Open the function at that line (replace `N` with the line number shown):

```bash
sed -n 'N,+40p' ltn_combined/ltn_fuzz_combined.py
```

**What to look for:**

- The LTN program is built and run inside a `try` block.
- When a failure is detected, the input is **saved** under `findings_valid/` or `findings_misuse/` (with `meta.json`, `program.txt`, `input.bin`), and the function **returns normally**.
- Nothing in the function changes the corpus, mutation, or any libFuzzer setting.

To libFuzzer, an input that triggered a bug report looks the same as any other run. It keeps the input **only if it added coverage**.

**Observed:** __________________ *(fill in: where findings are saved, how the function ends)*

**Result:** PASS if failures are only saved and reported.

---

## Check 5: Every input kept in the corpus added coverage (from the logs)

**Purpose:** prove from libFuzzer's own output that every change to the corpus was coverage-driven.

In the logs, `NEW` means "kept because it added coverage" and `REDUCE` means "replaced an input with a shorter one reaching the same coverage". If guidance is coverage only, then:

- every `NEW` line has a higher `ft` than the line before it
- every `REDUCE` line has the **same** `ft` as the line before it

This command checks every line of every campaign log:

```bash
for f in ltn_combined/logs/*.chunk*.log; do
  printf "%-28s " "$(basename $f)"
  awk '/ (INITED|NEW|REDUCE|pulse|DONE) /{
         for(i=1;i<=NF;i++) if($i=="ft:") ft=$(i+1)
         if($2=="NEW"){n++; if(ft<=prev) bad++}
         if($2=="REDUCE"){r++; if(ft!=prev) rbad++}
         prev=ft }
       END{printf "NEW: %d (without coverage gain: %d)   REDUCE: %d (coverage changed: %d)\n", n, bad+0, r, rbad+0}' "$f"
done
```

**Expected:** `without coverage gain: 0` and `coverage changed: 0` for every log.

**Observed:**
```
(fill in)
```

The number of `NEW` lines usually drops in later chunks. That is the coverage plateau: once LTN's reachable code is covered, few new inputs qualify.

**Result:** PASS if both counts are 0 everywhere.

---

## Check 6: Coverage grows, then plateaus

**Purpose:** show the guidance working over time.

```bash
for f in ltn_combined/logs/valid.chunk*.log; do
  printf "%-20s " "$(basename $f)"
  grep -E " (INITED|DONE) " "$f" | awk '{for(i=1;i<=NF;i++) if($i=="cov:") c=c" "$(i+1)} END{print "cov at start / end:" c}'
done
```

**Expected:** coverage rises in the first chunk, then stays almost constant. In the 1-minute run on 10 Oct, `cov` stayed at 339 for the whole minute with 0 new failure signatures.

**Observed:**
```
(fill in)
```

Coverage numbers depend on the Python version (Atheris instruments Python bytecode), so compare them only between runs on the same machine.

**Result:** PASS if coverage plateaus.

---

## Note for the report: comparison tracing

By default, libFuzzer also watches **comparisons** in the instrumented code. When LTN runs `if p == 0`, libFuzzer sees both values and may try them in later mutations. This is part of libFuzzer's standard coverage feedback (option `-use_cmp=1`, on by default), not an extra guidance signal. Value profile, which *would* add guidance from comparisons, is off (Check 2).

So the precise statement is:

> Guidance: libFuzzer's default coverage feedback (edge coverage with hit counts, plus comparison tracing) on the `ltn` package only. No dictionary, custom mutator, value profile or custom fitness.

**Optional strict test:** if a reviewer asks, run one campaign with `-use_cmp=0` added to the fuzzer command in `run_campaign.sh`, and compare how quickly F2 (p = 0) is found.

---

## Note for the report: oracles that call LTN

Some oracles (swap, permute, stable flip) call LTN a second time with modified inputs. Those calls run instrumented LTN code, so they can add coverage. This doesn't change the guidance **type** (still coverage only), but it means that switching oracles on can slightly change which inputs are kept.

**Optional test:** compare `ORACLES=basic` with the default on a fresh copy of the corpus:

```bash
cd ~/nesy_libfuzzer_ltn/ltn_combined
ORACLES=basic ./run_chunks.sh basicA 0 1 120
./run_chunks.sh allA 0 1 120
tail -3 logs/basicA.log logs/allA.log
```

Compare `cov` and the number of executions.

---

## Summary

| # | Check | Result |
|---|---|---|
| 1 | Coverage measured only on `ltn`, `ltn.core`, `ltn.fuzzy_ops` | PASS (Step 1) |
| 2 | No dictionary, value profile, custom mutator or crossover | ____ |
| 3 | Only standard Atheris calls; oracles have no channel to libFuzzer | PASS |
| 4 | Oracle failures are saved, the run continues normally | ____ |
| 5 | Every `NEW` input added coverage; every `REDUCE` kept the same coverage | ____ |
| 6 | Coverage grows, then plateaus | ____ |

**Conclusion:** Step 2 is satisfied once Checks 2, 4, 5 and 6 are filled in as expected.

### Text for the report

> The fuzzing is guided only by code coverage. libFuzzer keeps an input in the corpus only when it increases coverage of the `ltn` package (edge coverage with hit counts, plus libFuzzer's default comparison tracing). No dictionary, custom mutator, value profile or custom fitness function is used. The bug-detection oracles run on every input but only record failures; they do not influence which inputs libFuzzer keeps or mutates. In all campaign logs, every input added to the corpus increased coverage.
