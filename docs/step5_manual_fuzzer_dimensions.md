# Step 5 verification: adapting the manual fuzzers' dimensions to libFuzzer

**Requirement (supervisor):** if no dimensions, check the code inside the fuzzing directory, see which dimensions the manual fuzzers explored, and adapt the idea to libFuzzer.

**Answer:** done in **Round 2** (`ltn_combined/`). The manual fuzzers' dimensions, checks (oracles), metamorphic tests and seed inputs were ported into the libFuzzer harness. Guidance stayed coverage-only. Three manual-fuzzer ideas are **not yet** ported (listed in Part D).

**Verified on:** ____ Oct 2026, WSL (Ubuntu), folder `~/nesy_libfuzzer_ltn`

---

## Background

| | Manual fuzzers (neys_fuzz) | libFuzzer harness (Round 1) |
|---|---|---|
| How inputs are chosen | random values from hand-written rules, no feedback | mutated bytes, kept only if they reach new LTN code |
| Unit tested | one operator call at a time (e.g. `AndProd(x, y)`) | whole LTN programs (variables → predicate → formula → quantifier → SatAgg) |
| Strength | **strong oracles**: exact formulas, metamorphic laws | **broad inputs**: empty domains, misuse, nesting |
| Weakness | narrow inputs | Round 1 oracle only checked crashes, NaN, range, gradients |

Round 2 combines the two: libFuzzer's broad, coverage-guided inputs with the manual fuzzers' strong oracles.

**Source of the manual fuzzers:** `github.com/PromaChow/neys_fuzz`, branch `fuzz-ltn-lnn-probfuzz-adaptation`, commit `9512567`.

> **To confirm with the supervisor:** this repository is the "fuzzing directory" referred to. It was used because your computer was offline when the comparison was made. If the supervisor meant a different folder, compare it with this one first.

---

## 0. Setup

```bash
cd ~/nesy_libfuzzer_ltn
source venv/bin/activate
# get the manual fuzzers at the exact commit used (skip if already present)
git clone -b fuzz-ltn-lnn-probfuzz-adaptation https://github.com/PromaChow/neys_fuzz.git
git -C neys_fuzz checkout 9512567
```

---

## Part A: What the manual fuzzers explore

### Check 1: list the manual fuzzers

```bash
ls neys_fuzz/fuzzing/ltn/
ls neys_fuzz/fuzz/
ls neys_fuzz/results/seeds/
```

**Expected:** per-operator / per-property fuzzers under `fuzzing/ltn/` (`fuzz_*.py`, including swap, permute and stable variants), the shared oracle `fuzz/ltn_oracle.py`, the driver `fuzz/ltn_fuzz.py`, the not-yet-ported `fuzz/ltn_quant.py` and `fuzz/ltn_vs_lnn.py`, and the seed file `results/seeds/ltn_seeds.json`.

**Observed:** ____

### Check 2: the manual fuzzers' dimensions

Read the manual fuzzers' inputs (the values they vary):

```bash
grep -n "def \|random\.\|np\.random\|torch\.rand\|stable\|p=\|mask" neys_fuzz/fuzzing/ltn/fuzz_*.py | head -60
```

**Expected:** the manual fuzzers vary these dimensions, each on **one operator call**:

| Manual fuzzer dimension | Example |
|---|---|
| M1 operator | which connective or aggregator is called |
| M2 boundary values | inputs exactly 0 or 1 |
| M3 perturbation | inputs moved by a small step |
| M4 complement | input x replaced by 1 − x |
| M5 stable flag | `stable=True` vs `stable=False` |
| M6 p value | p for pMean / pMeanError, including 0 |
| M7 mask | which individuals a guard selects |

And these **checks** (oracles):

| Manual fuzzer check | Example |
|---|---|
| reference formula | LTN's `AndProd(x, y)` must equal x·y |
| Boolean law | on 0/1 inputs, connectives must match Boolean logic |
| swap | And/Or must give the same result with operands swapped |
| permute | an aggregator must give the same result when individuals are reordered |
| stable flip | stable and unstable versions must agree on ordinary values |
| dtype | output type must match input type |
| structure | result shape must match the formula's free variables |

**Observed:** ____

---

## Part B: How each manual dimension was adapted to libFuzzer

The manual fuzzers choose values with Python's random generator. In libFuzzer, every choice must come from **libFuzzer's bytes**, so coverage can guide it. The adaptation: each manual dimension became a byte-driven choice in the harness's **seed mode** (one of four inputs), or was already covered by D1–D7.

| Manual dimension | In the libFuzzer harness | Where |
|---|---|---|
| M1 operator | seed mode picks one of **105 operator calls recorded from LTN's own tests**; program mode picks operators via D4/D5 | `ltn_seeds.json`; D4, D5 |
| M2 boundary values | seed mode sets chosen values to 0 or 1; D1 includes 0, 1 | seed mode; D1 |
| M3 perturbation | seed mode moves a value by a step from `DELTAS`, up or down | `ltn_fuzz_combined.py` line 497 |
| M4 complement | seed mode replaces chosen values with 1 − x | seed mode |
| M5 stable flag | seed mode flips `stable`; D4/D5 choose it in programs | line 471; D4, D5 |
| M6 p value | seed mode chooses p; D5 uses p ∈ {1, 2, 3, 6, 10, 0, −1, 0.5} | seed mode; D5 |
| M7 mask | seed mode builds a random mask from bytes | line 502 |

### Check 3: seed mode exists and is chosen by the first byte

```bash
grep -n "seed\|SEED\|ltn_seeds" ltn_combined/ltn_fuzz_combined.py | head -20
sed -n 18,30p ltn_combined/README.md
```

**Expected:** the README describes two input modes chosen by the first byte: **program mode (3 of 4 inputs)** and **seed mode (1 of 4 inputs)**. Seed mode picks a recorded operator call, then chooses the stable flag, p, an optional mask, and which values to set to boundaries, perturb or complement.

**Observed:** ____

### Check 4: the 105 seed operator calls

```bash
python3 -c "import json; d=json.load(open('ltn_combined/ltn_seeds.json')); print(len(d), 'seed calls'); print(sorted({s['op'] for s in d}))"
cmp ltn_combined/ltn_seeds.json neys_fuzz/results/seeds/ltn_seeds.json && echo "identical to neys_fuzz"
```

**Expected:** `105 seed calls`, the list of LTN operators they cover, and `identical to neys_fuzz`.

If the first command fails because the file is elsewhere or uses different keys, find it with `ls ltn_combined/*.json` and print one entry with `python3 -c "import json; print(json.load(open('ltn_combined/ltn_seeds.json'))[0])"`.

**Observed:** ____

---

## Part C: How each manual check was adapted to libFuzzer

| Manual check | In the libFuzzer harness | Name in findings |
|---|---|---|
| reference formula (`ltn_oracle.py`) | copied **unchanged**, evaluated in float64 | `FormulaViolation` |
| Boolean law on 0/1 inputs | ported, tolerance ±3e-4 | `FormulaViolation` |
| swap, permute, stable flip | ported with the manual fuzzers' tolerances 3e-6, 1e-5, 3e-4 | `MetamorphicViolation` |
| dtype | ported | `DtypeViolation` |
| structure (shape vs free variables) | ported | `StructureViolation` |
| NaN gradient, unstable operators | added as a low-severity category | `GradientViolation` |

### Check 5: the reference oracle is copied unchanged

```bash
ls ltn_combined/
diff neys_fuzz/fuzz/ltn_oracle.py ltn_combined/ltn_oracle.py && echo "identical"
```

**Expected:** `ltn_oracle.py` in `ltn_combined/`, and `identical`.

**Observed:** ____

### Check 6: origin of every part is documented

```bash
sed -n 1,20p ltn_combined/README.md
```

**Expected:** a table "What came from where" listing which parts come from the Round 1 libFuzzer harness and which from neys_fuzz (`ltn_oracle.py`, `ltn_seeds.json`, the seed-mode dimensions, the swap/permute/stable tolerances, the Boolean law, dtype and `Quantifier([])` checks).

**Observed:** ____

### Check 7: the tolerances match the manual fuzzers

```bash
grep -n "3e-6\|1e-5\|3e-4" ltn_combined/*.py
grep -n "3e-6\|1e-5\|3e-4" neys_fuzz/fuzzing/ltn/fuzz_swap*.py neys_fuzz/fuzzing/ltn/fuzz_permute*.py neys_fuzz/fuzzing/ltn/fuzz_stable*.py
```

**Expected:** the same three numbers in both: swap 3e-6, permute 1e-5, stable flip 3e-4.

**Observed:** ____

### Check 8: the ported checks found the manual fuzzers' bugs again

From `findings_valid/*/meta.json` (10 Oct 2026):

| Bug (originally found by neys_fuzz) | Found by ported check | Folder | First found after |
|---|---|---|---|
| F24 `ImpliesGoguen(stable=True)` gives 0 for false → false | `FormulaViolation` | `96b8e7b1f7` | 42,863 inputs |
| F24 (same bug, second check) | `MetamorphicViolation` stable flip | `82cd98139b`, `1ae966e95d` | 47,111–53,865 inputs |
| F26 dtype: `NotGodel` | `DtypeViolation` | `5d17c074af` | 174 inputs |
| F26 dtype: masked `AggregMin` | `DtypeViolation` | `0ffd0ad9bb` | 373 inputs |
| F26 dtype: guarded quantifier | `DtypeViolation` | `ad91e9f6c0` | 9,195 inputs |
| F27 NaN gradient, unstable operators | `GradientViolation` | `b624d8c3fd` | 34,086 inputs |
| F25 `Quantifier([])` claims free variables | `StructureViolation` | `findings_misuse/…` | (misuse campaign) |

```bash
for d in 96b8e7b1f7 82cd98139b 5d17c074af 0ffd0ad9bb ad91e9f6c0 b624d8c3fd; do
  grep -h '"signature"\|first_seen_exec' ltn_combined/findings_valid/$d/meta.json
done
grep -h '"signature"' ltn_combined/findings_misuse/*/meta.json | grep -i structure
```

**Expected:** the signatures and counts in the table above.

**Observed:** ____

### Check 9 (optional): without the ported checks, these bugs are not found

The old Round 1 oracle can be switched on in the same fuzzer:

```bash
cd ~/nesy_libfuzzer_ltn/ltn_combined
ORACLES=basic ./run_chunks.sh basic 0 1 300
ls findings_basic/ 2>/dev/null | wc -l
grep -h '"signature"' findings_basic/*/meta.json 2>/dev/null | cut -c1-80
```

**Expected:** F1 and F2 signatures (crashes, NaN), but **no** `FormulaViolation`, `MetamorphicViolation` or `DtypeViolation`. The ported checks are what find F24 and F26.

If the findings folder has a different name, check with `ls -d findings_*`.

**Observed:** ____

---

## Part D: What was not ported yet

```bash
ls neys_fuzz/fuzz/ltn_quant.py neys_fuzz/fuzz/ltn_vs_lnn.py
grep -ln "finite\|numerical_grad\|eps\s*=\|gradcheck" neys_fuzz/fuzz/*.py neys_fuzz/fuzzing/ltn/*.py
```

| Not ported | What it does | Why it would help |
|---|---|---|
| Brute-force quantifier check (`ltn_quant.py`) | recomputes Forall/Exists with an explicit loop over individuals and compares with LTN | would check quantifier **values**, not just crashes/NaN; quantifiers are where F1–F3 live |
| LTN vs LNN differential (`ltn_vs_lnn.py`) | runs the same logic in LTN and LNN and compares | finds disagreements between two implementations; also prepares the LNN step |
| Finite-difference gradient check | compares LTN's gradients with numerical slopes | finds **wrong** gradients, not only NaN ones |

These are oracle improvements, not missing dimensions, so the requirement ("adapt the idea of the manual fuzzers' dimensions") is met. They are planned for the next round.

**Observed:** ____

---

## Part E: What libFuzzer added that the manual fuzzers did not explore

| libFuzzer dimension | Not in the manual fuzzers | Bug it found |
|---|---|---|
| D2 = 0 individuals | manual fuzzers always used non-empty inputs | **F1** |
| D5 p = 0, judged correctly | manual fuzzers tried p = 0 but counted **any** exception as a pass | **F2** |
| D7 quantify an unused variable | not explored | **F3** |
| D4 nesting depth ≤ 3, D6 SatAgg + backward | single operator calls only | (coverage of whole-program paths) |

So the combination works both ways: libFuzzer's dimensions found **3 new bugs**, and the manual fuzzers' checks let libFuzzer **re-find 4 known bugs**.

---

## Summary

| # | Check | Result |
|---|---|---|
| 1 | manual fuzzers located at the exact commit | ____ |
| 2 | their dimensions (M1–M7) and checks identified | ____ |
| 3 | seed mode adapts M1–M7 to byte-driven choices | ____ |
| 4 | 105 seed calls, identical to neys_fuzz | ____ |
| 5 | reference oracle copied unchanged | ____ |
| 6 | origin of every part documented in README | ____ |
| 7 | tolerances match the manual fuzzers | ____ |
| 8 | ported checks re-found F24, F25, F26, F27 | ____ |
| 9 | (optional) without them, F24/F26 are not found | ____ |
| D | three ideas not yet ported, identified | recorded |

### Text for the report

> We studied the manual LTN fuzzers (neys_fuzz, commit 9512567), which vary one operator call at a time along seven dimensions (operator, boundary values, perturbation, complement, stable flag, p, mask) and judge results with strong oracles (reference formulas, Boolean laws, swap, permutation and stable-flip metamorphic relations, dtype and structure checks). We adapted them to libFuzzer by turning each dimension into a byte-driven choice in a seed mode that mutates 105 operator calls recorded from LTN's own tests, and by porting the oracles into the harness, keeping coverage as the only guidance. The combined fuzzer re-found the four bugs known from the manual fuzzers (F24–F27), while libFuzzer's own dimensions (empty domains, p = 0 judged strictly, API misuse) found three new ones (F1–F3). Three manual-fuzzer oracles remain to be ported: brute-force quantifier evaluation, an LTN-vs-LNN differential, and a finite-difference gradient check.
