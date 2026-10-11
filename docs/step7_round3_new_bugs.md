# Step 7 – Round 3: extending the fuzzer and the new bugs

## What changed in the fuzzer (adaptation A8)
Round 3 adds the dimensions the earlier harness could not reach (file `ltn_round3.py`):
per-call `p`/`stable` on aggregators; `diag`/`undiag`; `Variable(add_batch_dim=False)` and odd dtypes/shapes;
`Predicate(model=nn.Module)` and `Predicate(func=)`; vector-valued `Function`; trainable `Constant`;
multi-variable quantifiers in both orders; nested quantifiers; raw tensors to `SatAgg`; two variables with one label;
and **reuse of an intermediate result in two formulas**.
New oracles: caller-object mutation, output shape vs free variables, Forall/Exists duality,
all-true mask == no mask, quantifier variable order, diag == mean over pairs, dtype preservation.

## Run it
```bash
cd ~/nesy_libfuzzer_ltn && source venv/bin/activate
cp /path/to/ltn_round3.py ltn_combined/
cd ltn_combined
PYTHONPATH=../ltn_torch python3 ltn_round3.py -max_total_time=240 -rss_limit_mb=3000
# findings land in findings_r3/<signature>/meta.json
```
Expected: ~1,800 exec/s, coverage ~268, `[FINDING ...]` lines for the signatures below.

## Reproduce without the fuzzer
```bash
cd ~/nesy_libfuzzer_ltn && PYTHONPATH=ltn_torch python3 check_all_bugs.py
```

| ID | Bug | Where | Severity |
|----|-----|-------|----------|
| F29 | `Connective` (and other ops) **modify their operand in place** when it comes from a model with a grad_fn: `free_vars ['x']→['x','y']`, shape `(3,)→(12,)`. Reusing the sub-formula afterwards gives wrong shapes/results. A detached predicate is not affected, so behaviour is inconsistent. | core.py `process_ltn_objects` (copy only if `grad_fn is None` or Variable/Constant) | High |
| F30 | Two variables with the same label but different sizes: 1 vs 2 individuals is silently accepted (wrong shape), 3 vs 4 gives a raw torch `RuntimeError`. | core.py:307/359 | Medium |
| F31 | `SatAgg()` with no formulas: raw `RuntimeError: stack expects a non-empty TensorList`. | fuzzy_ops.py:1669 | Low |
| F33 | `Variable(torch.tensor(0.5))` (0-d) accepted, every use crashes with `IndexError`. | core.py:307 | Low |

Rediscovered in this round: F1 (empty domain), F2 (p<=0), F26 (mask changes dtype).
Harness note: `core.py:616` IndexError came from the harness's own model on 1-D input, so it is not an LTN bug.
