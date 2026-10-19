# How Coverage-Guided Fuzzing Works, and How We Apply It to LTN

## Abstract

Fuzzing tests a program by running it on a very large number of automatically generated inputs and watching for failures. Pure random inputs waste almost all their runs, because programs reject malformed input early and never execute the code behind the first check. *Coverage-guided* fuzzing fixes this with one feedback signal: after each run it asks which parts of the program's code were executed, and it keeps and mutates only the inputs that reached something new. This document explains that algorithm in five parts, works through a complete 30-step example whose output is real, and then shows how the same loop is attached to Logic Tensor Networks (LTN) in this repository. All numbers in the worked example come from running `ltn_raw/toy_fuzzer.py`. No results about LTN itself are reported here; those are in the findings reports.

---

## 1. The problem

A program under test takes an input and either behaves correctly or fails (crashes, hangs, or returns something it should not). We want to find inputs that make it fail, and we cannot enumerate them by hand.

**Pure random fuzzing** generates random bytes, runs the program, and records any failure:

```
repeat:
    data = random bytes
    run program on data
    if it failed: save data
```

This is how fuzzing began (Miller, Fredriksen and So, 1990 [1]) and it still finds bugs in code with shallow input handling. It breaks down when the program checks its input in stages. Suppose a function only does real work if the first byte is `0x41` and the second is `0x42`:

```
if data[0] == 0x41:
    if data[1] == 0x42:
        ... the interesting code ...
```

A random pair of bytes passes both checks with probability 1/256 × 1/256 = 1/65 536. The expected cost is about 65 000 runs, and every further stage multiplies it by 256. Random generation gets no credit for passing the first check, so it cannot build on it.

## 2. The idea: use coverage as feedback

*Coverage* is the set of code locations (lines, branches, or edges) a run executed. If the program is instrumented to report it, the fuzzer can tell that an input which passed the first check behaved differently from one that did not, even though neither crashed. That difference is the signal that lets it build on partial progress:

```
input [0x00, 0x00]  → reaches the first if, fails it
input [0x41, 0x00]  → reaches the first if, PASSES it, fails the second   ← new coverage: keep this one
input [0x41, 0x42]  → reaches the interesting code                        ← new coverage: keep this one
```

Each stage now costs about 256 tries instead of multiplying: roughly 2 × 256 = 512 runs in total instead of 65 000. This is arithmetic on the example above, not a measurement.

## 3. The algorithm

```
corpus = [ one small starting input ]
seen   = { }                                  # every code location ever executed

repeat until a stop condition:
    parent = choose an input from corpus
    child  = mutate(parent)                   # a small random change
    hit    = run(program, child)              # also returns the set of locations executed
    if the run failed or an oracle fired:
        save child as a finding
    if hit contains a location not in seen:
        seen   = seen ∪ hit
        corpus = corpus + [child]             # keep it
    else:
        discard child
```

Five components make it work.

| Component | Role | In this repository |
|---|---|---|
| **Corpus** | The inputs worth mutating. It only grows when an input earns a place. | the `corpus_*` folders |
| **Mutation** | A small random edit of a kept input, so the search stays near known-good inputs. libFuzzer's edits include changing a byte or bit, inserting, erasing or shuffling bytes, copying one slice over another, adding or subtracting one from a stored integer, and splicing two inputs [2]. | done by libFuzzer |
| **Feedback** | Instrumentation that reports which code ran. | `atheris.instrument_imports(include=["ltn"])` |
| **Selection rule** | Keep a child only if it reached new coverage. | libFuzzer |
| **Oracle** | The test that decides whether a run was a bug. Crashes are detected automatically; anything subtler needs a check written by us. | the harness (section 5) |

Two further details matter in practice. First, a run is deterministic: the same bytes always reach the same code, so any finding replays exactly from its input file. Second, libFuzzer also *reduces* inputs, replacing a kept input with a shorter one that reaches the same coverage, which keeps the corpus small.

## 4. A complete worked example

The program in `ltn_raw/toy_fuzzer.py` is deliberately tiny so every step can be followed. It has two stages: the harness decodes two bytes into a table size and a power, and the "library" under test uses them.

```python
def harness(data):                      # bytes -> two choices
    rows = data[0] % 6                  # number of rows: 0..5
    p    = [1, 2, 3, 0][data[1] % 4]    # the power: 1, 2, 3 or 0
    return rows, p

def library(rows, p, hit):              # the code under test
    hit.add("start")
    if rows == 0:
        hit.add("empty-table branch")
        return "BUG  empty table gives nan"
    hit.add("normal-table branch")
    if p == 0:
        hit.add("p-zero branch")
        return "BUG  divide by zero"
    hit.add(f"p={p} branch")
    return "ok"
```

Two bugs are planted: an empty table, and a zero power. The program has seven coverage locations in all: `start`, `empty-table`, `normal-table`, `p-zero`, and one each for `p=1`, `p=2`, `p=3`. The starting input is `[3, 1]` (3 rows; byte 1 gives `p = 2`). Each step mutates **one byte** of a randomly chosen saved input. This is the real output of the run, trimmed to the steps where something was kept, plus a few that were discarded:

```
step    parent  ->      child  rows  p  result                      new coverage          decision
   1    [3, 1]  ->  [130, 1]     4  2  ok                          start, normal-table,  KEEP
                                                                    p=2 branch
   2    [3, 1]  ->  [3, 230]     3  3  ok                          p=3 branch            KEEP
   3  [130, 1]  -> [130,107]     4  0  BUG divide by zero           p-zero branch         KEEP
   4    [3, 1]  ->   [3, 14]     3  3  ok                          -                     discard
   7    [3, 1]  ->   [3, 15]     3  0  BUG divide by zero           -                     discard
   9 [130,107]  -> [216,107]     0  0  BUG empty table gives nan    empty-table branch    KEEP
  28 [130,107]  -> [130,180]     4  1  ok                          p=1 branch            KEEP
```

How to read it:

- **Step 1.** Byte 0 changes from 3 to 130. `130 % 6 = 4`, so the table has 4 rows. The program runs, executes `start`, `normal-table` and `p=2`, all new, so the input is kept.
- **Step 2.** Byte 1 changes to 230. `230 % 4 = 2`, so `p` is the third entry, 3. A new branch, `p=3`, so it is kept.
- **Step 3.** The parent is the kept input `[130, 1]`; byte 1 becomes 107. `107 % 4 = 3`, so `p = 0`. This runs the `p-zero` branch for the first time and triggers the **first bug**. Kept.
- **Step 4.** Another change to a kept input reaches nothing new. It is discarded. The corpus does not grow.
- **Step 7.** This input triggers the divide-by-zero bug again, but every location it touched was already in `seen`, so it is discarded as a *corpus* entry. The bug is still reported; coverage only decides what is kept for mutation. (A real fuzzer such as libFuzzer would stop and save a crash. This toy just records it.)
- **Step 9.** A single byte change on `[130, 107]` gives `[216, 107]`. `216 % 6 = 0`, so the table is empty. New branch, and the **second bug**. Kept.
- **Step 28.** `[130, 107] → [130, 180]`; `180 % 4 = 0` selects the first power, `p = 1`. The last unseen branch. Kept.

After 30 steps:

```
bugs found:   divide by zero, empty table gives nan
saved corpus: [3,1]  [130,1]  [3,230]  [130,107]  [216,107]  [130,180]      (the seed plus 5 kept inputs)
coverage:     7 of 7 locations
```

Only 5 of 30 children were kept, and each kept input is a stepping stone. The input that reached the empty-table bug (`[216, 107]`) was made by mutating `[130, 107]`, which had been kept because it reached `p-zero`. The fuzzer did not search for that bug directly; it accumulated coverage, and the bug fell out of the exploration.

## 5. Applying the loop to LTN

The loop is unchanged. Only the program under test and the oracle differ.

```
corpus ──mutate──► bytes ──► harness decodes ──► builds & runs LTN ──► coverage of the ltn package
   ▲                                                    │ new → keep in corpus
   └────────────────────────────────────────────────────┤ oracle fires → save to findings_*/
                                                        └ nothing new → discard
```

**The harness** is the code in `TestOneInput(data)`. It is the bridge from meaningless bytes to a valid LTN call:

- `ltn_fuzz/ltn_fuzz.py` reads bytes *on demand* through `FuzzedDataProvider` and builds a whole random program: variables, predicates, a formula tree, quantifiers, a satisfaction score, and optionally a training step.
- `ltn_raw/ltn_raw_fuzz.py` builds no structure: one byte selects an operator and the rest are reinterpreted as float32 numbers.
- `ltn_raw/ltn_hybrid_fuzz.py` reads a seven-byte header of choices (target, operator, aggregator, flags) and uses the rest as float data.

**The feedback** comes from instrumenting only the `ltn` package. Probes are inserted into `ltn/core.py` and `ltn/fuzzy_ops.py` at import time; PyTorch is left uninstrumented. Coverage is measured on LTN's own code and on nothing else.

**The oracles.** A crash is not the only failure that matters for LTN, because the library's typical failure is silent: a NaN or an out-of-range truth value that flows into training. The harnesses therefore check:

1. *Exceptions*: any exception that is not one of LTN's own documented validation messages.
2. *Value*: a closed formula or satisfaction score that is NaN or outside [0, 1] while every input was finite.
3. *Gradient* (`ltn_fuzz.py` only): a non-finite gradient while all inputs were finite and only the stable operator variants were used.

The conditions "while all inputs were finite" and "only stable operators" prevent the oracle from blaming LTN for a failure the input itself caused or that LTN documents as expected.

**Findings** are written to one folder per distinct failure signature, containing the input bytes (so the failure replays exactly) and a readable description of the program.

**Measuring progress.** `ltn_raw/cov_hybrid.py` replays a saved corpus under coverage.py and reports covered lines per file and the functions never called. `ltn_raw/run_until_plateau.py` repeats fuzzing rounds until the covered-line count is identical for three rounds in a row.

### How the LTN case differs from the toy

| | Toy example | LTN |
|---|---|---|
| Input meaning | 2 bytes, 2 choices | bytes decode into data, predicates, formulas |
| Code reached | 7 locations | thousands of lines across `core.py` and `fuzzy_ops.py` |
| Bug visible as | a returned string | usually *not* a crash: a NaN or out-of-range value |
| Needs a written oracle | no | yes, the oracle is most of the design |

## 6. Limitations

- **Coverage is a proxy.** Reaching a line does not mean testing it meaningfully; the value reaching it may be harmless. High coverage with a weak oracle finds nothing.
- **A plateau does not mean completeness.** Three rounds without new coverage mean the fuzzer is stuck for this harness and round length, not that no further bugs exist.
- **The harness bounds what can be found.** Code the harness never calls is never covered. For example, none of the harnesses currently run a real `nn.Module` predicate (`Predicate(model=...)`).
- **The oracle encodes our assumptions.** The expected-message list decides which exceptions count as legitimate; a wrong entry hides a bug or reports a false one.
- **`ltn_hybrid_fuzz.py` and the coverage scripts are untested.** They have been syntax-checked and reviewed against LTN's source but not yet run.

## 7. Reproducing the example

```bash
cd ltn_raw
python toy_fuzzer.py          # plain Python, no dependencies; prints the table in section 4
```

## References

1. B. P. Miller, L. Fredriksen, B. So. *An empirical study of the reliability of UNIX utilities.* Communications of the ACM 33(12), 1990.
2. LLVM project. *libFuzzer: a library for coverage-guided fuzz testing.* llvm.org/docs/LibFuzzer.html.
3. Google. *Atheris: a coverage-guided Python fuzzing engine.* github.com/google/atheris.
