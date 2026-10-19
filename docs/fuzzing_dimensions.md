# The Fuzzing Dimensions

## 1. Definition

A **dimension** is a group of related choices that the fuzzer makes while constructing one LTN program from a stream of bytes. Each choice has a finite set of admissible values. The fuzzer can only exercise behaviour that some dimension allows it to produce: if no dimension can generate an empty variable, no run, however long, can find a fault that requires one.

The method is *category partition* from software testing. The specification (here, LTN's API and source) is read, the independent categories of input are listed (the dimensions), and each category is divided into representative choices: ordinary values, boundary values, and special values. The difference from the classical method is that the combinations are not enumerated by a person. Each input is a byte string, a decoder (`Builder` in `ltn_fuzz/ltn_fuzz.py`) maps it to one value per dimension, and a coverage-guided engine (libFuzzer through Atheris) searches the byte space for combinations that reach new code.

Formally, let a dimension D_k be a finite set of choices, and let a *program configuration* be a tuple c = (c_1, ..., c_6) with each c_k in D_k. The decoder is a total function from byte strings to configurations, and the fuzzer explores configurations through the bytes. Dimension D7 is optional and extends the tuple only when the misuse mode is enabled.


| Dim | Name              | What it varies                                           | LTN element exercised                        |
| --- | ----------------- | -------------------------------------------------------- | -------------------------------------------- |
| D1  | Data values       | the numbers inside tensors                               | `Variable`, `Constant` contents              |
| D2  | Data shape        | how many things, how wide, how many variables            | `Variable`, `Constant` shape                 |
| D3  | Predicate model   | the function that turns features into truth values       | `Predicate`, `Function`                      |
| D4  | Formula structure | the tree of connectives                                  | `Connective`, `fuzzy_ops` operators          |
| D5  | Quantification    | how a variable is aggregated away                        | `Quantifier`, `diag`, guarded quantification |
| D6  | Satisfaction      | how formulas become one score, and whether it is trained | `SatAgg`, `backward()`                       |
| D7  | API boundary      | deliberately legal-looking but invalid calls (opt-in)    | validation in `Quantifier`, `SatAgg`         |


The remaining sections give each dimension's choices as implemented. All values and probabilities below are read from the harness source; the file and function that produces each one is named.

---



## 2. D1: data values

**What it varies.** The numerical content of every variable, constant, and predicate or function weight.

**Choices** (`Builder.num`, `Builder.weight`, list `SPECIALS`):

- With probability 1/4, a value drawn uniformly from a list of 16 *special values*:
`0, 1, −1, 0.5, 1e−4, 1−1e−4, 1e−8, 1e30, −1e30, NaN, +∞, −∞, 1.4e−45 (smallest subnormal), 88, −88, 3.4e38 (near the float32 maximum)`.
- Otherwise a uniform float in [−10, 10] (data), or in [−5, 5] (weights, which take a special value with probability 1/8).

**Rationale.** Fuzzy-logic operators are defined on truth values in [0, 1], and their numerical hazards sit at the endpoints and outside them: `0` and `1` make `xᵖ` and `(1−x)ᵖ` degenerate, `1e−8` and `1−1e−4` sit next to the stability margin ε = 10⁻⁴ that LTN's stable operators add, `±88` lies just below the point where `exp` overflows in float32 (about 88.7), and NaN and infinities test propagation. Ordinary values would never produce these.

**Bookkeeping.** The harness records whether every value was finite (`all_finite`). The value and gradient checks are applied only when it is true, so that a failure caused by an input NaN is not attributed to LTN.

## 3. D2: data shape

**What it varies.** The number and size of the objects the formula quantifies over.

**Choices** (`Builder.make_domain`):


| Choice                   | Range                      | Note                                  |
| ------------------------ | -------------------------- | ------------------------------------- |
| number of variables      | 1–3 (labels `x`, `y`, `z`) | more variables give more broadcasting |
| individuals per variable | 0–5                        | **0 is allowed**: an empty variable   |
| feature width            | 1–3                        |                                       |
|                          |                            |                                       |


**Rationale.** Each variable owns one axis of every tensor that mentions it, so the number of variables fixes the rank of intermediate results and the broadcasting that `process_ltn_objects` performs. An empty axis is the canonical edge case: aggregating over nothing is a division of zero by zero.

## 4. D3: predicate model

**What it varies.** The function from individuals' features to a degree of truth, and the optional term function.

**Choices** (`Builder.make_predicate`, `make_function`):

- 1 or 2 predicates, each of arity 1 or 2.
- The predicate computes a score `s = b + Σⱼ wⱼ · sum(xⱼ)` from a random bias `b` and weights `wⱼ` (D1 values, created with `requires_grad=True`), then applies one of three squashing functions: *sigmoid* (smooth, strictly inside (0, 1)), *crisp step* (`s > 0`, exactly 0 or 1, zero gradient), or *clamp* to [0, 1] (linear inside, exactly 0 or 1 outside).
- 0 or 1 function symbol `f(x) = a · x`, with a random scale `a`, usable inside a term, for example `P(f(x))`.

**Rationale.** The three squashing functions produce three regimes of truth values at the operators' inputs: interior values, exactly {0, 1}, and a mixture. LTN requires a predicate's output to lie in [0, 1] and checks it; all three satisfy the requirement. The weights are trainable so that gradients reach the predicate outputs, which D6 uses.

**Limit.** Predicates are built with `func=`, so no `torch.nn.Module` model is exercised through `Predicate(model=…)`.

## 5. D4: formula structure

**What it varies.** The shape of the logical expression.

**Choices** (`Builder.formula`, `connective`, `atom`, `term`):

A formula is generated recursively. At each node, while the depth is below 3, a number 0–9 is read:


| Draw | Node                                      |     |
| ---- | ----------------------------------------- | --- |
| 0–2  | atom: a predicate applied to terms        |     |
| 3    | unary connective applied to a formula     |     |
| 4–6  | binary connective applied to two formulas |     |
| 7–9  | quantifier applied to a formula (D5)      |     |


At depth 3 and beyond an atom is forced, so formulas nest at most three connective or quantifier levels.

- **Unary connectives:** `NotStandard`, `NotGodel`.
- **Binary connectives**, by family: And (`AndMin`, `AndProd`, `AndLuk`), Or (`OrMax`, `OrProbSum`, `OrLuk`), Implies (`ImpliesKleeneDienes`, `ImpliesGodel`, `ImpliesReichenbach`, `ImpliesGoguen`, `ImpliesLuk`), and Equivalence built from a randomly chosen And and Implies.
- `stable` **flag** chosen at random for the four operators that accept it (`AndProd`, `OrProbSum`, `ImpliesReichenbach`, `ImpliesGoguen`).
- **Terms** inside atoms: a variable, a constant, or (when a function symbol exists and the term depth is below 2) the function applied to another term.

**Rationale.** The fuzzy operators agree on the classical values 0 and 1 and differ elsewhere, so each family has its own failure points; the recursion lets a single harness reach both shallow and deep expressions, and nesting makes outputs of one operator the inputs of the next.

## 6. D5: quantification

**What it varies.** How a free variable is eliminated from a formula.

**Choices** (`Builder.quantified`, `aggregator`):


| Choice                                    | Values                                                                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| quantifier                                | `Forall` (`"f"`) or `Exists` (`"e"`)                                                                             |
| aggregator                                | `AggregMin`, `AggregMean`, `AggregPMean`, `AggregPMeanError`                                                     |
| exponent `p` (for the p-mean aggregators) | 1, 2, 3, 6, 10, **0, −1, 0.5**                                                                                   |
| `stable` flag (p-mean aggregators)        | random                                                                                                           |
| quantified variables                      | a random non-empty prefix of the formula's free variables                                                        |
| diagonal quantification                   | with probability 1/6, when two variables have equal size and a binary predicate exists: `Q(diag(x, y), P(x, y))` |
| guarded quantification                    | with probability 1/4: aggregation restricted by a condition `Σ features < θ`, where θ is a D1 value              |


**Rationale.** The set of `p` values is a boundary-value choice: ordinary (1, 2, 3), large (6, 10), and edge or invalid (0, −1, 0.5). The value 0 makes the exponent `1/p` undefined, and negative `p` raises zeros to a negative power. Diagonal and guarded quantification take code paths of `Quantifier.__call__` and `compute_mask` that plain quantification does not, including the replacement of NaN by a default value when a guard leaves nothing to aggregate. Any aggregator may be combined with either quantifier; LTN does not enforce the usual pairing (a strict aggregator for Forall, a lenient one for Exists), so all combinations are legal and are generated.

## 7. D6: satisfaction and training

**What it varies.** How closed formulas are combined, and whether the result is differentiated.

**Choices** (`Builder.run`, `close`):

- 1 or 2 closed formulas. A formula that still has free variables is closed by wrapping them in a `Forall` with a randomly chosen aggregator.
- The aggregator inside `SatAgg`, drawn from the same four as D5.
- With probability 1/2, a backward pass: `loss = 1 − SatAgg`, then `loss.backward()`, after which the gradients at the predicate outputs are inspected.

**Rationale.** The forward score is a single number, but LTN's purpose is to supply a training signal, so a fault in a connective, quantifier, or aggregator that corrupts only the derivative (a zero or non-finite gradient from a formula whose value is fine) is invisible without this step.

## 8. D7: API boundary (opt-in)

**What it varies.** Calls that are syntactically valid but outside the intended use, where LTN's own validation may be missing or wrong.

**Enabled by** the environment variable `LTNFUZZ_MISUSE=1`. With probability 1/10 each, the harness generates:

- **vacuous quantification**: a quantifier over a variable that does not occur in the formula;
- `SatAgg` **on an open formula**: a formula that still has free variables is passed to `SatAgg` without being closed.

The combined fuzzer (`ltn_combined/`) additionally tries `Quantifier` with an empty variable list.

**Rationale.** This dimension targets gaps in input validation, where an operation neither works nor raises one of LTN's documented errors.

---



## Source

All dimensions are defined in `ltn_fuzz/ltn_fuzz.py` (class `Builder`) and summarised in `ltn_fuzz/README.md`. The step-by-step verification of this list is in `docs/step4_dimensions.md`.