# Coverage-guided fuzzing of LTN (LTNtorch) with libFuzzer

Target: LTNtorch, commit `d1bd981` (https://github.com/tommasocarraro/LTNtorch).
Engine: libFuzzer through **Atheris 3.1** (Google's Python front end for libFuzzer).
Guidance: code coverage only (no custom feedback).

## Files

| File | Purpose |
|---|---|
| `ltn_fuzz.py` | The fuzz target. Decodes libFuzzer's bytes into an LTN program, runs it, applies the oracle. |
| `repro.py` | Replays an input file and prints the LTN program it decodes to. |
| `confirm_findings.py` | Harness-free reproducers for each finding (public LTN API only). |
| `measure_cov.py` | Replays a corpus under coverage.py to get line/branch coverage of `ltn/`. |
| `run_campaign.sh` | Runs one campaign: `./run_campaign.sh <name> <misuse 0/1> <seconds> <seed>` |
| `corpus_valid/`, `corpus_misuse/` | Corpora libFuzzer kept (inputs that reached new coverage). |
| `findings_*` | One folder per unique failure: `input.bin`, decoded `program.txt`, `meta.json` (hit count). |

## Setup (Ubuntu / WSL)

```bash
python3 -m venv venv && . venv/bin/activate
pip install atheris torch numpy coverage
git clone https://github.com/tommasocarraro/LTNtorch.git ltn_torch   # next to this folder
```

## Run

```bash
# stop at first failure (classic libFuzzer behaviour)
python ltn_fuzz.py corpus_valid -max_total_time=600

# keep going, de-duplicate failures into findings/
LTNFUZZ_COLLECT=1 python ltn_fuzz.py corpus_valid -max_total_time=600

# also exercise API-boundary inputs (dimension D7)
LTNFUZZ_COLLECT=1 LTNFUZZ_MISUSE=1 python ltn_fuzz.py corpus_misuse -max_total_time=600

# replay / triage
LTNFUZZ_NO_INSTRUMENT=1 python repro.py findings_valid/<id>/input.bin
python confirm_findings.py
```

## Fuzzing dimensions

| | Dimension | Choices read from the bytes |
|---|---|---|
| D1 | Data values | feature values, incl. 0, 1, 1e-4, NaN, ±inf, 1e30, 3.4e38, denormals |
| D2 | Data shape | 1–3 variables, 0–5 individuals each (0 allowed), feature width 1–3, 1-D vs 2-D, constants |
| D3 | Predicate model | sigmoid / crisp step / clamp, arity 1–2, weights; optional function term f(x) |
| D4 | Formula structure | Not, And, Or, Implies, Equiv trees up to depth 3; every operator family; stable flag |
| D5 | Quantification | Forall/Exists; Min, Mean, pMean, pMeanError; p ∈ {1,2,3,6,10,0,−1,0.5}; diag; guarded (masked) |
| D6 | Satisfaction | SatAgg operator, 1–2 closed formulas, optional backward pass |
| D7 | API boundary (opt-in) | vacuous quantification, SatAgg on an open formula |

## Oracle

1. Crash: any exception that is not one of LTN's own validation errors (matched by message prefix).
2. Value: a closed formula or SatAgg result that is NaN or outside [0, 1] although every input was finite.
3. Gradient: NaN/inf gradient at the predicate truth values with finite inputs and only `stable=True` operators.
