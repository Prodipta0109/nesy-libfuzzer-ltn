# Step 4 verification: the fuzzing dimensions, and what libFuzzer does with them

**Requirement (supervisor):** fix the dimensions the fuzzer will work in; try to see what libFuzzer actually does.

**Answer:** seven dimensions, D1–D7, were fixed by reading LTN's source code. Each is a group of choices the fuzzer makes when it builds an LTN program from libFuzzer's bytes. A script now derives them from LTN's code automatically, and libFuzzer's behaviour was observed directly in its logs and saved inputs.

**Verified on:** ____ Oct 2026, WSL (Ubuntu), folder `~/nesy_libfuzzer_ltn`

---

## Background: what a dimension is

A **dimension** is one group of related choices the fuzzer can vary. Example: "how many individuals does a variable have?" is a choice in **D2 (shape)**, with values 0–5.

This is the **category-partition method** from software testing. Read the specification (here, LTN's code), list the categories (dimensions), split each into choices (normal values, boundaries, special values). The difference from classic category-partition: libFuzzer, guided by coverage, picks which combinations to try, not a human.

Why the dimensions matter: **libFuzzer can only explore what the dimensions allow.** If no dimension can produce "0 individuals", the fuzzer can never find F1, however long it runs.

---

## 0. Setup

```bash
cd ~/nesy_libfuzzer_ltn
source venv/bin/activate
```

---

## Part A: The seven dimensions

Choices as documented in `ltn_fuzz/README.md` (lines 48–54):

| Dim | Name | Choices | Comes from (LTN API) |
|---|---|---|---|
| D1 | data values | feature values, incl. 0, 1, 1e-4, NaN, ±inf, 1e30, 3.4e38, denormals | contents of `Variable` / `Constant` tensors |
| D2 | data shape | 1–3 variables, 0–5 individuals each (0 allowed), feature width 1–3, 1-D vs 2-D, constants | size of `Variable` / `Constant` tensors |
| D3 | predicate model | sigmoid / crisp step / clamp, arity 1–2, weights; optional function term f(x) | `Predicate(model=…)`, `Function` |
| D4 | formula structure | Not, And, Or, Implies, Equiv trees up to depth 3; every operator family; `stable` flag | `Connective(...)`, `fuzzy_ops.*` |
| D5 | quantification | Forall / Exists; Min, Mean, pMean, pMeanError; p ∈ {1, 2, 3, 6, 10, 0, −1, 0.5}; diag; guarded (masked) | `Quantifier(...)`, `diag`, `cond_vars` / `cond_fn` |
| D6 | satisfaction | SatAgg operator, 1–2 closed formulas, optional backward pass | `SatAgg(...)`, `.backward()` |
| D7 | API boundary (opt-in, misuse mode) | vacuous quantification (a variable not in the formula), SatAgg on an open formula; the combined fuzzer also tries `Quantifier([])` | gaps in validation of `Quantifier.__call__`, `SatAgg.__call__` |

The p values in D5 are a boundary-value choice: normal values (1, 2, 3), large (6, 10), and invalid or edge values (0, −1, 0.5). The value **0** is what produced F2.

### Check 1: the dimensions are written in the fuzzer

```bash
sed -n 1,40p ltn_fuzz/ltn_fuzz.py | grep -n "D[1-7]"
grep -n "D[1-7]" ltn_fuzz/README.md ltn_combined/README.md | head -30
```

**Expected:** the harness description and the READMEs list D1–D7 with the choices in the table above.

**Observed (10 Oct 2026):**
```
12:  D1  data values        individuals' feature values incl. 0, 1, eps, NaN, +-inf, huge
13:  D2  data shape         number of individuals per variable (0 allowed), feature width,
15:  D3  predicate model    sigmoid / crisp step / clamp, arity 1 or 2, weights
16:  D4  formula structure  Not / And / Or / Implies / Equiv trees, operator family, depth,
18:  D5  quantification     Forall / Exists, aggregator (Min, Mean, pMean, pMeanError),
20:  D6  satisfaction       SatAgg operator, number of closed formulas, backward pass
21:  D7  API boundary use   (only with LTNFUZZ_MISUSE=1) vacuous quantification,

ltn_fuzz/README.md:48:| D1 | Data values | feature values, incl. 0, 1, 1e-4, NaN, ±inf, 1e30, 3.4e38, denormals |
ltn_fuzz/README.md:49:| D2 | Data shape | 1–3 variables, 0–5 individuals each (0 allowed), feature width 1–3, 1-D vs 2-D, constants |
ltn_fuzz/README.md:50:| D3 | Predicate model | sigmoid / crisp step / clamp, arity 1–2, weights; optional function term f(x) |
ltn_fuzz/README.md:51:| D4 | Formula structure | Not, And, Or, Implies, Equiv trees up to depth 3; every operator family; stable flag |
ltn_fuzz/README.md:52:| D5 | Quantification | Forall/Exists; Min, Mean, pMean, pMeanError; p ∈ {1,2,3,6,10,0,−1,0.5}; diag; guarded (masked) |
ltn_fuzz/README.md:53:| D6 | Satisfaction | SatAgg operator, 1–2 closed formulas, optional backward pass |
ltn_fuzz/README.md:54:| D7 | API boundary (opt-in) | vacuous quantification, SatAgg on an open formula |
ltn_combined/README.md:13:| Program builder, dimensions D1–D7, crash / NaN / range / gradient oracle | `ltn_fuzz/` (libFuzzer harness) |
ltn_combined/README.md:23:   SatAgg, backward pass). See `../ltn_fuzz/README.md` for D1–D7.
ltn_combined/README.md:45:./run_chunks.sh misuse 1 4 300     # also D7 API-boundary inputs
```

The dimensions are defined once, in the Round 1 harness (`ltn_fuzz`), and reused unchanged by the Round 2 fuzzer (`ltn_combined/README.md` line 13 and 23). D7 is switched on only with `LTNFUZZ_MISUSE=1`, which `./run_chunks.sh misuse 1 …` sets.

**Result: PASS.**

### Check 2: each dimension is a choice in the code

The harness builds programs in a fixed order: values and shapes, then predicates, formula, quantifiers, SatAgg. Each choice reads bytes through a small helper (`self.i(lo, hi)` for a number in a range, a boolean helper for yes/no; see Step 3, A3).

```bash
grep -c "self\.i(" ltn_combined/ltn_fuzz_combined.py
grep -n "AggregMin\|AggregMean\|AggregPMean\|AggregPMeanError" ltn_combined/ltn_fuzz_combined.py | head
grep -n "AndProd\|OrProbSum\|ImpliesGoguen\|ImpliesReichenbach\|Equiv" ltn_combined/ltn_fuzz_combined.py | head
grep -n "diag\|cond_fn\|cond_vars" ltn_combined/ltn_fuzz_combined.py | head
grep -n "MISUSE" ltn_combined/ltn_fuzz_combined.py | head
```

**Expected:** many `self.i(` choice points; the four aggregators (D5); the operators (D4); `diag` and guard arguments (D5); and `MISUSE` gating the D7 choices.

**Observed:** ____

---

## Part B: Dimensions derived automatically from LTN's code

The dimensions were first chosen by reading LTN by hand. `classify_dims.py` now does this automatically: it reads LTN's source (without running it), lists every public class, method and parameter, assigns each parameter to a dimension, and marks which parameters LTN never validates.

### Check 3: run the script

Put `classify_dims.py` (sent earlier in the chat) in `~/nesy_libfuzzer_ltn`, then:

```bash
python3 -W ignore classify_dims.py ltn_torch/ltn --csv ltn_dimensions.csv | tail -15
```

`-W ignore` hides LTN's docstring SyntaxWarnings.

**Expected (LTNtorch d1bd981):**
```
Parameters per dimension:
  D1               37
  D2               11
  D3               5
  D3 (arity)       2
  D4               17
  D5               34
  D6               3
  D7               5
  internal         1
  validator        1
```

104 parameter rows in total, saved to `ltn_dimensions.csv` (opens in Excel or LibreOffice).

**Observed:** ____

### Check 4: unvalidated parameters predict the bugs

```bash
grep "UNVALIDATED" ltn_dimensions.csv | grep -E ",p,|individuals|vars|closed_formulas"
grep -E "AggregPMean|AggregPMeanError" ltn_dimensions.csv | grep ",p,"
```

**What to look for:** `p` in both pMean aggregators is **UNVALIDATED**, which is exactly where F2 (p = 0) occurs. The bugs sit where LTN's input validation has gaps:

| Parameter | Dimension | Validated by LTN? | Bug |
|---|---|---|---|
| `AggregPMean(p)`, `AggregPMeanError(p)` | D5 | **no** | F2 |
| `Variable(individuals)`: type is checked, **size 0 is not** | D2 | partly | F1 |
| `Quantifier.__call__(vars)`: type checked, **variable absent from formula** not | D5 + D7 | partly | F3 |
| `Quantifier([])` | D7 | partly | F25 |

**How it works:** three small rule tables at the top of the script (class name → dimension, parameter name → dimension, LTN's validator functions). To apply the same method to DeepProbLog, NeurASP or LNN, only those tables change.

**Observed:** ____

### Check 5: gaps found in the current dimensions

Comparing the script's output with the harness shows six choices the fuzzer does **not** make yet:

| Missing choice | Dimension | Why it matters |
|---|---|---|
| `p` and `stable` passed per call: `Forall(x, φ, p=0)` | D5 | second code path; F2 may exist there too |
| `undiag(...)` | D5 | public API, never exercised |
| `Variable(..., add_batch_dim=False)` | D2 | unvalidated, changes tensor shape |
| `Predicate(func=...)` instead of `model=` | D3 | different constructor branch |
| `Constant(..., trainable=True)` | D6 | decides whether gradients flow |
| raw non-scalar tensor given to `SatAgg` | D7 | reaches F28 (SatAgg's broken error message), which the fuzzer currently cannot generate |

These are recorded as known limitations of the current dimensions and the plan for Round 3.

---

## Part C: What libFuzzer actually does

### Check 6: watch it run

```bash
cd ~/nesy_libfuzzer_ltn/ltn_combined
./run_chunks.sh valid 0 1 60 &
tail -f logs/valid.log          # Ctrl+C stops watching; the run continues
```

**What libFuzzer does, line by line:**

| Log part | Meaning |
|---|---|
| `INFO: … files found in corpus_valid` | loads the saved interesting inputs from earlier runs |
| `#4425 REDUCE … MS: 3 CopyPart-InsertByte-EraseBytes-` | took a saved input, applied 3 byte mutations (copy part, insert byte, erase bytes), ran LTN, got the same coverage with a shorter input, so replaced the old one |
| `NEW` | the mutated input reached new LTN code, so it is kept and mutated further later |
| `cov: 339` | LTN branches reached so far |
| `ft: 1355` | finer coverage (branches + how often they ran) |
| `exec/s: 402` | inputs per second |

**Observed (10 Oct 2026, 1 minute):** 24,527 inputs, `cov` constant at 339, 13 corpus updates (all `REDUCE`), 0 new failure signatures. Coverage had plateaued from earlier campaigns.

### Check 7: one input, three views (bytes → choices → LTN program)

libFuzzer only ever sees bytes. The harness turns them into dimension choices. Each saved finding keeps both:

```bash
xxd findings_valid/34be71331f/input.bin | head
cat findings_valid/34be71331f/program.txt
cat findings_valid/34be71331f/meta.json
```

**Expected:**
- `input.bin`: a few bytes, the only thing libFuzzer knows
- `program.txt`: the LTN program they decoded into; look for a variable with **0 individuals** (D2) and a **Min** quantifier (D5)
- `meta.json`: `IndexError@fuzzy_ops.py:1220`, F1, `first_seen_exec: 1`

**Observed:** ____

### Check 8: which dimension found which bug

| Bug | Dimension choice that triggers it | Saved example |
|---|---|---|
| F1 empty domain | D2 = 0 individuals, with D5 = any aggregator | `findings_valid/34be71331f` (Min crash), `5e774fb287` (Mean NaN) |
| F2 p = 0 | D5: pMean or pMeanError with p = 0 | `findings_valid/221f71e246` |
| F3 unused variable | D7: quantify a variable not in the formula | `findings_misuse/…` |
| F24 Goguen implication | D4: `ImpliesGoguen(stable=True)` with D1 values 0, 0 | `findings_valid/96b8e7b1f7` |
| F25 empty quantifier | D7: `Quantifier([])` | `findings_misuse/…` |
| F26 dtype | D4/D5: `NotGodel`, masked `AggregMin`, guarded quantifier | `findings_valid/ad91e9f6c0` |
| F27 NaN gradient | D4 unstable operator + D6 backward pass | `findings_valid/b624d8c3fd` |

```bash
for d in 221f71e246 96b8e7b1f7 ad91e9f6c0 b624d8c3fd; do
  echo "== $d"; grep signature findings_valid/$d/meta.json; head -15 findings_valid/$d/program.txt
done
```

To fill the misuse rows, run a short misuse campaign and list its findings:

```bash
./run_chunks.sh misuse 1 1 120
for d in findings_misuse/*/; do echo "== $d"; grep signature "$d/meta.json"; done
```

**Observed:** ____

### Check 9: how hard each dimension choice was to reach

From `meta.json`, `first_seen_exec` is the number of inputs libFuzzer needed before the first hit:

| Bug | First hit after | Why |
|---|---|---|
| F1 Min crash | 1 input | D2 = 0 is one byte value; very easy |
| F26 dtype | 174–9,195 inputs | common operators |
| F2 p = 0 | ~13,800 inputs | needs **two** choices at once: a pMean aggregator **and** p = 0 |
| F27 gradient | ~34,000 inputs | needs an unstable operator + backward pass + specific values |
| F24 Goguen | ~43,000 inputs | needs a specific operator, `stable=True` and values exactly 0, 0 |

Bugs that need several dimensions to line up take longer to find.

---

## Summary

| # | Check | Result |
|---|---|---|
| 1 | D1–D7 documented in the fuzzer | PASS (10 Oct) |
| 2 | each dimension is a choice in the harness code | ____ |
| 3 | dimensions derived automatically: 104 parameters, counts per dimension | ____ |
| 4 | unvalidated parameters match where F1, F2, F3, F25 occur | ____ |
| 5 | six missing choices identified (Round 3 plan) | recorded |
| 6 | libFuzzer's loop observed: mutate → run → keep if new coverage | PASS (10 Oct) |
| 7 | one input seen as bytes, program and failure | ____ |
| 8 | each bug traced to the dimension choice that triggers it | ____ |
| 9 | difficulty per bug from `first_seen_exec` | PASS (10 Oct) |

### Text for the report

> We fixed seven fuzzing dimensions by reading LTN's API: data values (D1), data shape (D2), predicate model (D3), formula structure (D4), quantification (D5), satisfaction and gradients (D6), and API-boundary misuse (D7). The harness decodes libFuzzer's bytes into one choice per dimension and builds an LTN program from them. A static analysis script derives the dimensions automatically from LTN's source, mapping its 104 public parameters to D1–D7 and flagging parameters LTN never validates; the bugs found (F1, F2, F3, F25) lie exactly at those validation gaps. The script also revealed six choices the current dimensions do not cover, planned for the next round. Observing libFuzzer directly, it mutates saved inputs at the byte level and keeps only those reaching new LTN code; bugs needing a single choice (an empty domain) appear within the first input, while bugs needing several choices to coincide (p = 0 together with a pMean aggregator, or a specific stable operator with boundary values) take 14,000–43,000 inputs.
