## How an LTN rule is built, and which kinds of building block exist

This section explains, in plain language, what LTN (Logic Tensor Networks) is made of, how one rule is put together from those parts, and how many kinds of each part the library offers. It is the vocabulary needed to read the fuzzing harnesses and the findings. The source is `ltn/core.py` and `ltn/fuzzy_ops.py` in LTNtorch at commit `d1bd981`.

### The idea in one paragraph

LTN lets you write a rule in logic ("every cat is an animal") and turn it into a number between 0 and 1 saying how true the rule is. A neural network supplies the truth of the simple facts ("this picture is a cat: 0.9"). Fixed formulas combine those facts according to the rule. The final number is used as a training signal, so the network is pushed toward answers that obey the rule. Because the combining formulas sit between the network and the training signal, any NaN or infinite value produced there reaches the network's gradient. That is why they are worth fuzzing.

### The pieces, from data to one number

A rule is built from the inside out. Each step produces a new object that the next step consumes.

**1. Variable: the data.**
A variable is a named batch of things, for example 6 animals described by 2 numbers each. Its tensor has one axis for the individuals and one for their features (shape `[6, 2]`). A **Constant** is a single thing (shape `[2]`) that does not range over a batch.

**2. Predicate: a question about each thing.**
A predicate takes the features of each individual and returns a degree of truth between 0 and 1: "how much is this a cat?". In real use it is a neural network ending in a sigmoid. LTN checks the output and refuses anything outside [0, 1]. Applied to a variable it gives one truth value per individual (shape `[6]`). A predicate applied to a variable is called an **atom**, the smallest formula. A **Function** works like a predicate but returns features instead of a truth value, so it can be nested inside an atom, as in `Cat(f(x))`.

**3. Connective: combining truth values.**
A connective takes one or two columns of truth values and returns one column: Not, And, Or, Implies, Equivalent. It always works element by element, so the number of values does not change. "If cat then furry" is `Implies(Cat(x), Furry(x))`, still one value per individual.

**4. Quantifier: collapsing a whole batch into one value.**
A quantifier takes a formula that still depends on a variable and removes that dependence. `Forall(x, ...)` says "for every x" and `Exists(x, ...)` says "for at least one x". The letter you pass to LTN is `"f"` or `"e"`. Doing the collapse is the job of an aggregator (see below). A formula with no variable left is called **closed** and has a single truth value (shape `[]`).

**5. SatAgg: scoring several rules together.**
`SatAgg` takes several closed formulas and combines them into one number, the satisfaction of the whole knowledge base. Training minimises `1 - SatAgg`.

Two helper functions sit beside these: `diag` pairs the individuals of two variables one to one instead of in every combination, and `undiag` undoes it. A **guarded** quantifier restricts the aggregation to individuals that satisfy a condition, for example "for all x with a large score".

#### A worked example

```
Forall(x,                       <- step 4, collapses the 3 individuals into one number
   Implies(                     <- step 3, combines two columns
      Cat(x),                   <- step 2, an atom: one value per individual
      Furry(x)))                <- step 2, an atom
x = Variable(3 animals)         <- step 1
```

The shapes along the way are `[3, 2]` (the variable), `[3]` (each atom), `[3]` (the implication), and `[]` (the closed formula). Rules for building one correctly:

- a predicate is applied to variables or constants, never to formulas;
- a connective is applied to formulas, and the number of inputs must match (Not takes one, the others take two);
- a quantifier is applied to a variable and a formula;
- every variable must be quantified before the formula is scored;
- every truth value, at every step, must lie in [0, 1], or LTN raises its own validation error.

### How many kinds of each piece exist

LTN fixes the logic pieces; only the predicates and functions are free-form. The counts below are what `fuzzy_ops.py` provides.

**Connectives: 14 operators, in 5 families.** "And" has several valid definitions in fuzzy logic. The definitions agree when every value is exactly 0 or 1 (ordinary logic) and differ in between, so LTN offers several of each.

| Family | Operators | Formula for inputs `x`, `y` |
|---|---|---|
| Not (2) | `NotStandard` | `1 - x` |
| | `NotGodel` | 1 if `x` is 0, otherwise 0 |
| And (3) | `AndMin` | the smaller of `x`, `y` |
| | `AndProd` | `x * y` |
| | `AndLuk` | `max(0, x + y - 1)` |
| Or (3) | `OrMax` | the larger of `x`, `y` |
| | `OrProbSum` | `x + y - x*y` |
| | `OrLuk` | `min(1, x + y)` |
| Implies (5) | `ImpliesKleeneDienes` | `max(1 - x, y)` |
| | `ImpliesGodel` | 1 if `x <= y`, otherwise `y` |
| | `ImpliesReichenbach` | `1 - x + x*y` |
| | `ImpliesGoguen` | 1 if `x <= y`, otherwise `y / x` |
| | `ImpliesLuk` | `min(1, 1 - x + y)` |
| Equivalent (1) | `Equiv` | And of (`x` implies `y`) and (`y` implies `x`); you choose which And and which Implies it uses |

**Aggregators: 4.** An aggregator turns a list of truth values into one value. It knows nothing about logic; it is arithmetic on a list.

| Aggregator | What it computes | Behaviour |
|---|---|---|
| `AggregMin` | the smallest value | strictest: one bad item decides the result |
| `AggregMean` | the average | forgiving: bad items are diluted |
| `AggregPMean(p)` | `(mean of xᵖ) ^ (1/p)` | a dial between the two: negative `p` leans toward the minimum, `p = 1` is the average, large `p` leans toward the maximum |
| `AggregPMeanError(p)` | `1 - (mean of (1-x)ᵖ) ^ (1/p)` | the same dial applied to the errors, which punishes bad items smoothly; the usual choice for "for all" |

**Quantifiers: 2 meanings.** `Forall` (`"f"`) and `Exists` (`"e"`). A quantifier is an aggregator plus a meaning plus the bookkeeping of which axis to collapse. Any aggregator can be paired with either letter, but "for all" should be paired with a strict aggregator and "exists" with a lenient one. The letter decides the answer when a guarded quantification has nothing to aggregate: 1.0 for "for all" (nothing breaks the rule) and 0.0 for "exists" (nothing satisfies it).

**Satisfaction aggregator: 1.** `SatAgg` wraps one of the 4 aggregators (default `AggregPMeanError(p=2)`) to score several rules at once.

**Core objects: 6.** `Constant`, `Variable`, `Predicate`, `Function`, `Connective`, `Quantifier`, plus the two helper functions `diag` and `undiag`.

### The stability switch

Six of these classes accept `stable=True` (the default): `AndProd`, `OrProbSum`, `ImpliesReichenbach`, `ImpliesGoguen`, `AggregPMean` and `AggregPMeanError`. Their exact formulas have unusable gradients at exactly 0 and exactly 1 (for example a division by zero, or a derivative that is zero). The stable version first nudges every input slightly away from the endpoints, by about 1e-4 (`0` becomes `0.0001`, `1` becomes `0.9999`). The result is off by roughly that amount and the gradient stays finite. `stable=False` gives the exact textbook formula and is documented as gradient-unsafe. This is why the harnesses treat a NaN under `stable=True` as a more serious finding than a NaN under `stable=False`.

### Why this matters for the fuzzing

Everything above is a small, fixed set of building blocks (14 + 4 + 2 + 1 + 6), combined in a few legal ways. That makes the input space structured: a harness must pick a legal combination before the interesting arithmetic can run. The harnesses differ in how much of that structure they build themselves:

- `ltn_raw/ltn_raw_fuzz.py` builds none of it and only calls the operators of `fuzzy_ops.py` on raw tensors.
- `ltn_raw/ltn_hybrid_fuzz.py` reads a short header of choices and calls one part of the library per input.
- `ltn_fuzz/ltn_fuzz.py` builds whole random rules from the five steps above, runs them forward, and optionally trains one step to check the gradients.
