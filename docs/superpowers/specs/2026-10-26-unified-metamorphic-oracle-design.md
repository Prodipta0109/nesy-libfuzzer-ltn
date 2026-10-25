# Unified metamorphic oracle for LTN, LNN and NeuPSL: design

Status: draft for approval. Nothing in this document has been built.

## 1. Intent

The existing libFuzzer studies (LTN, LNN, DeepProbLog) check each library with oracles written separately for that library:
a closed formula, dtype, and a handful of metamorphic checks (swap, permute, `stable` flip). The professor asked for
metamorphic fuzzing. The idea chosen: derive the metamorphic relations (MRs) from the theory of fuzzy logic and
first-order logic, and use **one catalogue of MRs, one formula generator and one oracle for all three libraries**, with a
thin adapter per library.

Why this is possible: Lukasiewicz logic is the common core of the three.

- LTNtorch can be configured with Lukasiewicz operators.
- LNN's default semantics is (weighted) Lukasiewicz; at weights = 1 and default bias it is plain Lukasiewicz.
- PSL, the core of NeuPSL, is Lukasiewicz by definition (hinge-loss MRF over Lukasiewicz relaxations).

Why derive MRs from theory rather than from each library's docs: the theory says *which identities must hold in which
semantics*, so the oracle needs no reference implementation and the same relations transfer across libraries.

## 2. Goals, non-goals, anti-goals

Goals
- G1. One MR catalogue (Section 5), written once, each MR with its logic source and its side-condition.
- G2. One random formula generator and one rewrite engine; MRs are rewrite rules applied at random positions.
- G3. One adapter interface (Section 4) implemented for LTN, LNN, NeuPSL.
- G4. Report, per library and per MR, violations with a minimised replayable formula.
- G5. Evidence that the MRs add value: planted-bug (mutation) kill rate compared with the existing oracles, using the
  approach of `lnn/mutation_check.py`.
- G6. Free by-product: differential testing, since all three must agree on Lukasiewicz values.

Non-goals
- N1. Not replacing the coverage-guided libFuzzer harnesses; the MR layer plugs into them as an oracle.
- N2. Not testing training or convergence behaviour.
- N3. Not testing non-Lukasiewicz semantics in the shared set (Godel, product configurations of LTN stay in the
  LTN-specific MRs, Section 6).

Anti-goals (would make this a failure even if it runs)
- X1. False positives from applying a classical law outside its valid semantics. Every MR carries a side-condition and
  is only generated when it holds.
- X2. Tolerances picked by hand. Every tolerance is derived (Section 7).
- X3. MRs that only restate the per-operator closed-form oracle. The catalogue must emphasise compositions,
  quantifier and variable handling, and inference, where no closed form exists.

## 3. Sources of the relations

| Source | Used for |
|---|---|
| Handbook of Mathematical Fuzzy Logic, Vol 2, Ch. VI (MV-algebras), Ch. VII (Godel), Ch. IX (McNaughton) | propositional MRs A1-A13 |
| Handbook Vol 1 / Hajek, *Metamathematics of Fuzzy Logic* | prelinearity, divisibility, residuation (not read; to be confirmed) |
| van Krieken, Acar, van Harmelen, AIJ 302 (2022) | aggregator relations, derivative closed forms, implication property tables |
| LTN paper (arXiv 2012.13635), Appendix B | LTN operator and aggregator definitions |
| FOL | quantifier and variable MRs B1-B8 |

Only pages 1-13 of van Krieken and the table of contents of the Handbook have been read. Items marked "to confirm" must be
checked against the text before implementation.

## 4. Architecture

```
formula generator -> rewrite engine (MR catalogue) -> (phi, phi')
                                                         |
          adapter(LTN)  adapter(LNN)  adapter(NeuPSL)  <-+
                                                         |
                          oracle: |eval(phi) - eval(phi')| <= tol(phi, phi', lib)
```

Adapter interface (the only library-specific code):

- `build(formula, domain, atoms) -> handle`: construct the library's object for a formula tree over a finite domain,
  with all atom truth values fixed.
- `eval(handle) -> float in [0,1]`: truth of the formula.
- `declared_quantifiers() -> {forall: name, exists: name}`: used to decide which B-group MRs are valid.
- Optional `infer(handle)`, `bounds(handle)` and `map_state(handle)` for the C group.

How each adapter evaluates:
- LTN: Lukasiewicz connectives, grounded atoms as constants; read the output tensor.
- LNN: leaves with L = U = value, weights 1, default bias; run upward inference; read the root bounds, which must be a
  point interval.
- NeuPSL (unverified): observed atoms; encode phi as the rule `true -> phi`; truth = 1 - distance to satisfaction.

The generator draws formulas from the grammar common to all three: atoms, constants 0/1, `not`, `strong and`,
`strong or`, `weak and (min)`, `weak or (max)`, `implies`, and (B group) `forall`, `exists` over a finite domain.
Library-specific operators are excluded from the shared set.

## 5. Shared MR catalogue (Lukasiewicz core)

Notation: (x) strong conjunction, (+) strong disjunction, ^ min, v max, -> implication, ~ negation.

### A. Propositional (rewrite rules, always valid in Lukasiewicz)

| ID | Rewrite | Source |
|---|---|---|
| A1 | commute (x), (+), ^, v | MV-algebra axioms |
| A2 | re-associate (x), (+), ^, v | MV-algebra axioms |
| A3 | ~~a -> a | MV axiom |
| A4 | a (x) b -> ~(~a (+) ~b), and the dual | De Morgan |
| A5 | a -> b  =  ~a (+) b | Lukasiewicz implication |
| A6 | a -> b  =  ~b -> ~a | contraposition |
| A7 | a v b = (a -> b) -> b ; ~(~x (+) y) (+) y = ~(~y (+) x) (+) x | MV-algebra |
| A8 | prelinearity: (a -> b) v (b -> a) evaluates to 1 | BL-algebra (to confirm) |
| A9 | divisibility: a ^ b = a (x) (a -> b) | BL-algebra (to confirm) |
| A10 | \|a-b\| = (a (x) ~b) (+) (b (x) ~a) | MV distance function |
| A11 | monotonicity: increasing an input in the monotone direction never moves the output the wrong way | residuated lattice |
| A12 | Boolean boundary: on {0,1} atoms the output follows the classical truth table | all |
| A13 | McNaughton: the output is piecewise linear in each atom with integer slope, |slope| <= occurrences of that atom | Handbook Ch. IX (to confirm) |

Separating relations (checked to FAIL, to catch a library silently using another semantics):
S1. `a v ~a` must not be identically 1 (fails at a = 0.5). S2. `a (x) a` must not equal `a` at a = 0.5.
A passing S1 or S2 is reported as a semantics mismatch, not as a crash.

### B. Domain and quantifier (FOL layer; valid per declared quantifier semantics)

| ID | Relation | Side-condition |
|---|---|---|
| B1 | permute the domain objects | none |
| B2 | rename bound variables | none |
| B3 | swap adjacent same-kind quantifiers | none |
| B4 | not forall x P  =  exists x not P | exists is the exact dual of forall |
| B5 | forall x phi = phi when x is not free | forall is min or exists is max (fails for a (x)-fold: expected) |
| B6 | forall x P <= P(c) <= exists x P | forall/exists are min/max or (x)/(+)-fold |
| B7 | singleton domain: forall = exists = P(c) | none |
| B8 | aggregate over a concatenated domain = T(A(left), A(right)) | T-based aggregators only (van Krieken eq. 2) |

B5 is deliberately a discriminator: it exposes which aggregator semantics a library implements versus declares.
Libraries differ (LTN: configurable pmean-style; PSL: grounds forall into separate weighted rules; LNN: not yet
confirmed), which is why side-conditions come from `declared_quantifiers()`.

### C. Inference-level (adapter-specific, same spirit)

- LNN: shrinking leaf bounds never widens parent bounds (inclusion monotonicity); downward inference only tightens.
- NeuPSL (unverified): scaling all rule weights by c > 0 leaves the MAP state unchanged; duplicating a rule equals
  doubling its weight; rule order is irrelevant; replacing a rule by an A-equivalent rule leaves the MAP objective
  unchanged; adding an already satisfied rule changes nothing.
- LTN (Section 6).

## 6. Library-specific extensions (outside the shared set)

LTN only, from van Krieken: autograd gradients equal the closed-form derivatives (eq. 14, 15, 19, 23, 26, 28, 32-35);
derivative duality `dS(1-a,1-b)/d(1-a) = dT(a,b)/da`; Godel-only formulas are single-passing; non-vanishing-gradient
fractions are 1/n! (Lukasiewicz aggregator) and 1/2^(n-1) (nilpotent); implication property matrix (Tables 4-5) with
negative expectations; Godel order-invariance (`out(f(x)) = f(out(x))` for strictly increasing f with f(0)=0, f(1)=1);
pmean/pmean_error: p = 1 equals the arithmetic mean, monotone in p.

## 7. Oracle and tolerances

- Compare in float64 where the library allows; also run float32 and classify as precision note, not violation, as the
  existing harness does.
- Tolerance for a rewrite = (operation count of phi + phi') x machine epsilon of the dtype, times a Lukasiewicz slope
  bound (Lukasiewicz maps are 1-Lipschitz per occurrence). This replaces the hand-picked 3e-6 / 1e-5 constants. (Builder: derive and
  justify in code comments.)
- LTN stable-projection operators are excluded from the shared set; where used in Section 6 the documented eps = 1e-4
  bound applies.

## 8. Evaluation

- E1. Mutation study: plant bugs in each library (in memory, like `mutation_check.py`) covering: wrong operand order in
  implication, min in place of Lukasiewicz sum, off-by-one in axis handling, swapped quantifier dimension. Report which
  oracle kills each mutant: existing oracles only, MR oracle only, both.
- E2. Run the catalogue on the real libraries at the commits already pinned (LTN d1bd981, LNN c88f513) and report new
  violations, each minimised to a replay script.
- E3. Success for the professor's question: at least one class of defect found by the MR layer that the existing
  closed-form/dtype/swap oracles do not find, or a clear negative result stating that none exists.

## 9. Risks and open items

- R1. NeuPSL is not in the repo and its API has not been inspected. PSL's core is JVM-based as far as I recall, so
  Atheris may not instrument it; a different fuzzer or plain random testing may be needed for that adapter. Unverified.
- R2. LNN weighted semantics: A1-A10 hold only at default weights and bias.
- R3. Only a fraction of the cited sources has been read; A8, A9, A13 and the van Krieken implication tables must be
  confirmed against the text.
- R4. Anti-goal X3: many A-group rules may be caught by the existing closed-form oracle. E1 will show this honestly.

## 10. Calls made by the author of this draft (not agreed with the user)

1. Lukasiewicz as the shared semantics (not min/Godel or product).
2. Rewrite-rule formulation of MRs with a generic generator, instead of hand-written per-MR tests.
3. Tolerances derived from operation count and the 1-Lipschitz property rather than fixed constants.
4. NeuPSL treated as the least certain adapter; implemented last.
5. Separating relations S1 and S2 included as expected-failure checks.
6. Mutation kill rate as the main evidence of value.

## 11. Left to the builder

Concrete formula grammar sizes and depth limits, the rewrite engine data structure, minimisation algorithm, file layout
under `ltn_combined/`, `lnn/` and a new shared directory, and how to hook the oracle into the existing libFuzzer
`TestOneInput` functions.
