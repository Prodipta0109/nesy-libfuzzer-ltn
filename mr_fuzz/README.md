# Metamorphic-relation guided fuzzing of LTN's fuzzy operators

Idea: instead of waiting for a crash or NaN, check **algebraic laws** that the operators must obey
(from Metcalfe, *Fundamentals of Fuzzy Logics*) and let the fuzzer hunt for inputs that break them.

| File | What it is |
|---|---|
| `mrs.py` | The metamorphic relations (23 of them) and `bucket()`, the guidance. Start here. |
| `mr_fuzz.py` | The harness: bytes -> three vectors in [0,1] -> run one MR -> report violation |
| `selftest.py` | Plants bugs in LTN operators and checks that the MRs catch them (checks the MRs, not LTN) |
| `run.sh` | Runs the harness under Atheris/libFuzzer |

## How it works
1. Byte 0 of the input picks an MR, the rest become the vectors `x, y, z`.
2. The MR runs the operators on `x, y, z` and on a transformed version (swap arguments, re-bracket,
   apply De Morgan, ...) and returns a **margin**: how far apart the two sides are (0 = law holds).
3. **Guidance:** `mrs.bucket(margin)` has one code line per margin size (1e-15 ... 0.5). libFuzzer only
   rewards new code edges, so inputs with a bigger margin count as new coverage, get kept, and are mutated more.
4. Margin above `TOL` (1e-9) = violation: saved once per MR in `findings/<mr>/input.bin` + `info.txt`.

## Run
```bash
cd mr_fuzz
python selftest.py                       # MRs catch planted bugs? (should print CAUGHT 8 times)
python mr_fuzz.py --random 20000         # no Atheris needed, random inputs only
./run.sh 120                             # real guided fuzzing, 2 minutes (after ../setup.sh)
```

## Adding an MR
Write `def my_mr(x, y, z): return margin` in `mrs.py` and add it to the `MRS` list.
Use `err(a, b)` for `a == b` and `below(a, b)` for `a <= b`.

## Notes
* Only the exact variants are tested (`stable=False`); the stable ones nudge values on purpose.
* Contraposition is only checked for Lukasiewicz / Kleene-Dienes / Reichenbach, where it holds (Lemma 7).
* Not yet covered: quantifiers over masks/`dim`, `SatAgg`, predicates with neural models.
