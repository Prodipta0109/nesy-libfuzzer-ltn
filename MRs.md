# Metamorphic relations for fuzzy-logic libraries (A and B groups)

This catalogue lists the metamorphic relations (MRs) used to test LTN, LNN and the PSL core. All three compute
"how true is this formula" with fuzzy logic (truth values in [0, 1]). The relations come from Łukasiewicz logic,
the fuzzy logic the three share. Each relation says that two formulas, or one formula in two equivalent
environments, must have the same truth value. No expected values and no reference implementation are needed: a
library that disagrees with itself is wrong.

- **A group:** propositional relations (A1 to A13).
- **B group:** domain and quantifier relations (B1 to B8).

## Notation

| symbol | code op | meaning | formula |
|---|---|---|---|
| `¬a` | `not` | negation | `1 − a` |
| `a ⊗ b` | `and_s` | strong AND (Łukasiewicz t-norm) | `max(a + b − 1, 0)` |
| `a ⊕ b` | `or_s` | strong OR (Łukasiewicz t-conorm) | `min(a + b, 1)` |
| `a ∧w b` | `and_w` | weak AND | `min(a, b)` |
| `a ∨w b` | `or_w` | weak OR | `max(a, b)` |
| `a → b` | `imp` | Łukasiewicz implication | `min(1 − a + b, 1)` |

Quantifiers over a domain `{d1, …, dn}`, with `vi = φ(di)`:

| mode | `∀x φ` | `∃x φ` |
|---|---|---|
| `minmax` | `min(v1, …, vn)` | `max(v1, …, vn)` |
| `fold` | `v1 ⊗ … ⊗ vn = max(Σvi − (n−1), 0)` | `v1 ⊕ … ⊕ vn = min(Σvi, 1)` |

On an empty domain, `∀ = 1` and `∃ = 0`.

## How a relation is checked

1. A random formula (and a random truth-value table, the environment) is generated.
2. A rewrite rule is applied at one random position, giving a second formula that is equal by the law.
3. Both formulas are evaluated by the library with the same environment.
4. The gap is compared with a tolerance of `(size(φ) + size(φ')) · eps`, where `eps` is the machine epsilon of the
   library's arithmetic (about 2e-16 for float64, 1e-7 for float32, 1e-5 for PSL).
5. A gap above the tolerance is a violation.

Rewrite rules (most A rules, B2 to B5) change the formula. Environment rules (B1, B7) change the environment.
Property relations (A11 to A13, B6, B8) check one formula against a law about its behaviour and return a margin, the
amount by which the law is broken. A margin of 0 means the law holds.

## A group: propositional relations

"Direction" says which directions are implemented as rewrite rules. All hold in Łukasiewicz logic.

| ID | Law | Direction |
|---|---|---|
| A1 | `a ∘ b = b ∘ a` for `∘` in ⊗, ⊕, ∧w, ∨w | one rule |
| A2 | `(a ∘ b) ∘ c = a ∘ (b ∘ c)` for the same four operators | both |
| A3 | `¬¬a = a` | elimination only |
| A4 | `a ⊗ b = ¬(¬a ⊕ ¬b)` (De Morgan) | both |
| A5 | `a → b = ¬a ⊕ b` | both |
| A6 | `a → b = ¬b → ¬a` (contraposition) | one rule |
| A7 | `a ∨w b = (a → b) → b` | both |
| A8 | `(a → b) ∨w (b → a) = 1` (prelinearity) | forward only |
| A9 | `a ∧w b = a ⊗ (a → b)` (divisibility) | both |
| A10 | `a ⊗ ¬b = ¬(a → b)` (negated implication; a building block of the MV distance `\|a − b\|`, which itself is not in the grammar) | both |
| A11 | With all atoms in {0, 1}, the output equals the classical truth value | property |

Notes on directions:
- A3 is elimination only (`¬¬a → a`); the reverse would match every node and grow the formula.
- The backward direction of A8 is not implemented: the rewrite would have nothing to build `a` and `b` from.
- Each A rule that has both directions is two rewrite rules (for example `A4_demorgan_fwd` and `A4_demorgan_bwd`),
  16 directions in total for A1 to A10.

Arithmetic form of the rewrite rules (both sides expressed in the operators above):

| ID | Left side | Right side |
|---|---|---|
| A3 | `1 − (1 − a)` | `a` |
| A4 | `max(a + b − 1, 0)` | `1 − min((1 − a) + (1 − b), 1)` |
| A5 | `min(1 − a + b, 1)` | `min((1 − a) + b, 1)` |
| A6 | `min(1 − a + b, 1)` | `min(1 − (1 − b) + (1 − a), 1)` |
| A7 | `max(a, b)` | `min(1 − min(1 − a + b, 1) + b, 1)` |
| A8 | `max(min(1 − a + b, 1), min(1 − b + a, 1))` | `1` |
| A9 | `min(a, b)` | `max(a + min(1 − a + b, 1) − 1, 0)` |
| A10 | `max(a − b, 0)` | `1 − min(1 − a + b, 1)` |

## B group: domain and quantifier relations

The side-condition says when a relation is generated. It is taken from each adapter's declared quantifiers.

| ID | Law | Kind | Side-condition |
|---|---|---|---|
| B1 | Permuting (rotating) the domain objects leaves `∀` and `∃` unchanged | environment rule | none |
| B2 | Renaming a bound variable changes nothing: `∀x φ = ∀y φ[y/x]`, `y` fresh in `φ` | rewrite | none |
| B3 | Swapping two adjacent quantifiers of the same kind: `∀x ∀y φ = ∀y ∀x φ` (and for `∃`) | rewrite | none |
| B4 | Duality: `¬∀x φ = ∃x ¬φ`, in both directions | rewrite | `∃` is the exact dual of `∀` (minmax, or fold with fold) |
| B5 | Vacuous quantifier: `∀x φ = φ` when `x` does not occur in `φ`, in both directions | rewrite | aggregators are min and max |
| B6 | Instantiation: `∀x φ ≤ φ[c/x] ≤ ∃x φ` for any object `c` | property | min/max, or fold |
| B7 | On a one-object domain, `∀x φ = ∃x φ = φ(c)` | environment rule | domain has one object |
| B8 | A quantifier over a concatenated domain `L + R` equals the combination of its two halves. For `minmax`: `∀(L+R) = ∀L ∧w ∀R` and `∃(L+R) = ∃L ∨w ∃R`. For `fold`: `∀(L+R) = ∀L ⊗ ∀R` and `∃(L+R) = ∃L ⊕ ∃R` | property | aggregator is a T-norm fold or min/max |


## Where the laws come from, and what is confirmed

| Source | Used for | Status |
|---|---|---|
| van Krieken, Acar, van Harmelen, "Analyzing Differentiable Fuzzy Logic Operators", arXiv:2002.06100 (Artificial Intelligence 302, 2022) | Operator definitions (Tables 1 to 5, eq. A.1), aggregators (eq. 1 and 2, Table 3), strong negation (Def. 13), commutative and associative t-norms (Def. 14, 16), contrapositive symmetry (Def. 19), S-implications (Def. 20), symmetric aggregators (App. A.4) | Read for this catalogue (arXiv version; page and equation numbers may differ in the journal version) |
| Handbook of Mathematical Fuzzy Logic, Vol. 2, Ch. VI, VII, IX; Hájek, *Metamathematics of Fuzzy Logic* | A1 to A13 (MV-algebra, BL-algebra, McNaughton) | Cited at book and chapter level only; A8, A9 and A13 are unconfirmed |
| First-order logic | B1 to B7 | Standard |

What the van Krieken paper supports, by relation: A1 and A2 (Def. 14, 16), A3 (Def. 13), A4 (eq. A.1), A5 (Def. 20),
A6 (Def. 19(d) and Table 4), B1 (App. A.4), B8 (eq. 2 with Def. 14). A7 to A13, B2 to B6 and B7 are not in that paper.

The mathematics of every rewrite rule was also checked numerically against a float64 reference evaluator, which
confirms the laws but not the citations.
