# Combined LTN fuzzer: libFuzzer coverage + neys_fuzz oracles

This is the libFuzzer harness from `ltn_fuzz/` with the dimensions and oracles of the manual fuzzers in
**neys_fuzz** (branch `fuzz-ltn-lnn-probfuzz-adaptation`, commit `9512567`) added.

* **Guidance is unchanged:** libFuzzer keeps an input only when it reaches new code in the `ltn` package.
* **The oracle is stronger:** every fuzzy operator and aggregator call is judged by the checks below.

## What came from where

| Part | Origin |
|---|---|
| Program builder, dimensions D1–D7, crash / NaN / range / gradient oracle | `ltn_fuzz/` (libFuzzer harness) |
| `ltn_oracle.py` (documented closed formulas, float64) | copied unchanged from `neys_fuzz/fuzz/ltn_oracle.py` |
| `ltn_seeds.json` (105 operator calls recorded from LTN's own tests) | `neys_fuzz/results/seeds/ltn_seeds.json` (made by `fuzz/trace_ltn.py`) |
| Seed mode: boundary / perturb / complement values, stable flip, p, mask | dimensions of `neys_fuzz/fuzzing/ltn/fuzz_*.py` |
| Swap, permute, stable-flip tolerances (3e-6, 1e-5, 3e-4) | `neys_fuzz/fuzzing/ltn/fuzz_swap/permute/stable.py` |
| Boolean-boundary law, dtype check, `Quantifier([])` shape check | `neys_fuzz/fuzz/ltn_fuzz.py` (conn laws, F25, F26) |

## Two input modes (chosen by the first byte)

1. **Program mode (3 of 4 inputs):** bytes build a whole LTN program (variables, predicates, formulas, quantifiers,
   SatAgg, backward pass). See `../ltn_fuzz/README.md` for D1–D7.
2. **Seed mode (1 of 4 inputs):** bytes pick one operator call recorded from LTN's own tests, then choose
   the stable flag, p, an optional mask, and which values to set to boundaries / perturb / complement.

## Oracles

| | Check | Failure class |
|---|---|---|
| O0 | unexpected exception (not one of LTN's own validation errors) | the exception |
| O0 | closed formula / SatAgg NaN or outside [0,1] with finite inputs | `OracleViolation` |
| O1 | LTN in float64 vs documented formula | `FormulaViolation` |
| O1 | on inputs that are exactly 0/1, connectives must follow Boolean logic (±3e-4) | `FormulaViolation` |
| O2 | swap operands of commutative connectives; reverse individuals along an aggregated axis; flip `stable` | `MetamorphicViolation` |
| O3 | output dtype = input dtype (operators); quantifier returns float32 | `DtypeViolation` |
| O4 | every formula's value has one axis per free variable, of the right size | `StructureViolation` |
| O5 | NaN gradient at truth values (stable ops: bug; unstable ops: low severity) | `GradientViolation` |
| P | float32 result far from formula while float64 is right (never stops the fuzzer) | `PrecisionNote` |

## Run

```bash
./run_chunks.sh valid 0 4 300      # 4 x 5 min, valid programs
./run_chunks.sh misuse 1 4 300     # also D7 API-boundary inputs
ORACLES=basic ./run_chunks.sh basic 0 4 300   # the original weak oracle, for comparison
LTNFUZZ_NO_INSTRUMENT=1 python repro.py findings_valid/<id>/input.bin
```

Campaigns run in chunks because Atheris' instrumentation grows memory by about 10 KB per execution; libFuzzer resumes
from the corpus at every chunk.

## Round 3: `ltn_round3.py`

A separate Atheris harness (it does not use the seed corpus of this folder) that adds the choices the program builder
above cannot generate. Bytes choose one of eleven tests:

| Test | Dimension added |
|---|---|
| quantifier | per-call `p` / `stable`, two-variable quantification in both orders, guard on another variable |
| duality | Forall(φ) = 1 − Exists(¬φ) for pMean / pMeanError |
| diag | `diag` / `undiag`, round trip leaves the caller's variables unchanged, result = mean over pairs |
| variable | `add_batch_dim=False`, 0-d / 3-d tensors, float16 / float64 / int / bool, trainable `Constant` |
| connective | broadcasting of formulas over different variables, unary Not, caller's operands unchanged |
| SatAgg | raw tensors, empty and non-scalar inputs, backward pass |
| function | vector-valued `Function`, constants inside formulas, gradient into a trainable constant |
| same label | two variables with one label but different sizes |
| nested | ∀x ∃y φ with every operator family |
| mask | all-true guard must equal no guard; dtype must be preserved |
| reuse | one intermediate result used in two formulas must not be changed by the first use |

Known causes are filtered in the NaN check (empty variables, p ≤ 0), so the findings are new or different failures.

```bash
PYTHONPATH=../ltn_torch python3 ltn_round3.py -max_total_time=240 -rss_limit_mb=3000
python3 - <<'PY'            # list the findings
import json, glob
for f in sorted(glob.glob("findings_r3/*/meta.json")):
    m = json.load(open(f)); print(f.split("/")[1], m["kind"], m["loc"], m["detail"][:90])
PY
```

New bugs found this way: F29 (operands modified in place), F30 (same label, different sizes), F31 (`SatAgg()` with no
formulas), F33 (scalar `Variable`). See `../docs/step7_round3_new_bugs.md`. Standalone reproduction:
`PYTHONPATH=../ltn_torch python3 ../check_all_bugs.py F29   # or any bug id; "list" shows them all`.
