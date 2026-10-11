# LNN (IBM Logical Neural Networks 1.1.0) – libFuzzer findings

Target: LNN 1.1.0, commit c88f513, used from a source clone. Engine: libFuzzer via Atheris 3.1.0, guidance by code coverage only. Date: 11 October 2026.
Method as for LTN and DeepProbLog: dimensions L1–L7 derived from the source (173 parameters, 160 unvalidated; L4 neuron settings travel in one `activation={...}` dict and were listed by hand), byte-driven harness with nine test families, oracles independent of the library, per-bug runnable demonstrations (`check_lnn_bugs.py`, 9 of 9).

Adaptations: A1–A12 reused; A13 every inference call has a step limit and reaching it is a finding (LNN's `infer()` has no default limit); A14 the limit is also passed through `Model.train(max_steps=...)`.

Round 1 campaign: 52,589 executions (8 chunks × 170 s, 2 workers), coverage plateau 673 edges, 18 failure signatures, 5 bugs (+1 consequence demo); see round 2/3 below for 3 more. The first campaign, without a step limit, ended in libFuzzer timeouts (kept in `phase1_hangs/`).

Result: propositional inference is exact against closed-form Łukasiewicz (and weighted) intervals; the bugs are about termination, formula identity and unfinished features.

| ID | Severity | Bug | Where |
|---|---|---|---|
| L1 | high | `infer()` never terminates for weighted And/Or/Implies with unequal weights when the lighter-weight operand has an exact value (each step shifts the bounds by one float32 step, above the 1e-7 threshold) | model.py:507 |
| L1b | high | consequence of L1: `Model.train()` hangs once weights are unequal | model.py:619/635 |
| L2 | high | `infer()` never terminates for quantifiers over a predicate with informative data; the neuron is rebuilt every step | unary_operator.py:248 |
| L5 | medium | formulas are identified by printed name; a formula built twice is half-orphaned; result depends on add_knowledge order | model.py:275 |
| L3 | medium | NeuralActivation.Godel / Frechet offered, modules missing (ModuleNotFoundError) | neural_activation.py |
| L4 | medium | Loss.LOGICAL UnboundLocalError for Iff / XOr | connective_neuron.py:170 |

Limits: propositional and unary-predicate quantifiers only; Gödel untestable; quantifier oracle is containment with tolerance; learning oracle has no gradient check.
Next: report upstream (L1, L2 first); binary predicates and nested quantifiers; NeurASP.

## Round 2/3 (11 Oct 2026): metamorphic oracles, binary predicates, nested quantifiers
lnn_round3.py: t_meta_prop (commutativity, De Morgan, material implication, double negation, Iff symmetry), t_meta_model (add_knowledge order, repeated infer, flush), t_contradiction, t_fol2 (binary predicates, nested quantifiers, complete classical data), t_fol_shapes, t_learn_ranges. 6,654 executions, 700 edges, 157 inputs. Propositional metamorphic checks: no violation. Three new bugs (all first-order), demos L6–L8 in check_lnn_bugs.py (9 of 9):
| L6 | high | consistent quantified sentence reported as contradiction: Forall x S(x) with S(a) true, S(b) false (or Exists with same data); downward inference pushes a grounding to (1,0) | unary_operator.py:267-272 |
| L7 | medium | quantifier over a predicate with another free variable (Exists(y, R(x,y))) crashes at construction, ValueError; library tests for it are commented out | unary_operator.py:83 |
| L8 | medium | predicate with repeated variable R(x,x) combined with any other formula: pandas Length mismatch | grounding tables |
Planted-bug check: metamorphic family catches And/Or/Implies mutants (16-40 of 40). Total: 8 bugs (L1-L8) + L1b consequence demo.

Evidence kit (11 Oct 2026): runnable with/without experiments and minimal pairs for every dimension and adaptation (evidence_kit.zip).
