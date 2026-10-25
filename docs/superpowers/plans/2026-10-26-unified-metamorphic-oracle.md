# Unified Metamorphic Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One catalogue of Lukasiewicz-derived metamorphic relations, one formula generator and one oracle that test LTN, LNN and NeuPSL through thin adapters.

**Architecture:** A library-independent core (`Node` AST, reference evaluator, rewrite rules, property relations, oracle, minimiser, byte-driven fuzz harness) lives in a new package `mr_unified/`. Each library gets an adapter exposing `evaluate(node, env) -> float`. MRs are rewrite rules applied at random positions; the oracle compares `evaluate(phi)` with `evaluate(phi')` under a derived tolerance.

**Tech Stack:** Python (LNN needs >= 3.11, so a new env), numpy, torch (CPU), pytest, optional Atheris. LTNtorch at `ltn_torch/` (commit d1bd981), LNN cloned by `lnn/setup_lnn.sh` (commit c88f513).

**Spec:** `docs/superpowers/specs/2026-10-26-unified-metamorphic-oracle-design.md`

## Global Constraints

- Shared semantics is Lukasiewicz only; Godel/product configurations stay out of the shared set (spec N3).
- Every MR carries a side-condition and is generated only when it holds (spec X1). Tolerances are derived, not hand-picked (spec X2).
- LNN adapter runs at default weights (1.0) and default bias (spec R2). LTN adapter uses Lukasiewicz operators only, no `stable` operators.
- Existing code is not modified. `mr_fuzz/` (LTN-only operator-level MR fuzzer, commit c5c8ef1) stays as is; `bucket()` guidance is copied, not imported, because `mr_fuzz/mrs.py` imports `ltn` at module load.
- Python 3.9 syntax is NOT assumed: the new env is Python >= 3.11, but keep to `typing`-style annotations and dataclasses so the core imports on 3.9 (pure-core tests run without torch).
- Commit steps run only if the user has asked for commits in this session; otherwise leave changes uncommitted.
- Run everything from the repo root with `PYTHONPATH=ltn_torch:lnn/src`.

## Review Focus

- Atoms exactly 0.0 / 1.0, and domains of size 1: no crash, no NaN, no false violation (Task 6 tests).
- Deep formulas (depth 8) must not trigger float false positives; calibration test in Task 6.
- Rewrites that grow the formula (A7, A9, A10 duplicate subtrees) must be size-capped at `MAX_NODES = 200`; formula is skipped, not truncated (Task 2).
- A rewrite that yields a structurally identical formula (e.g. commuting `And(a, a)`) is a no-op and must be skipped; LNN keys nodes by printed name and would silently merge them (Tasks 2, 5).
- A saved finding must replay deterministically in a fresh process (hash randomisation, set ordering) (Task 7).

---

## File Structure

```
mr_unified/
  __init__.py
  formula.py        Node AST, Env, size(), paths(), ref_eval()      (Task 1)
  rewrites.py       Rule dataclass, A-group + B-group rules, RULES  (Tasks 2, 4)
  properties.py     A11-A13 and separating relations S1, S2          (Task 3)
  adapters/base.py  Adapter protocol                                 (Task 5)
  adapters/ltn_adapter.py, lnn_adapter.py, neupsl_adapter.py         (Tasks 5, 6, 10)
  oracle.py         tolerance, check_rewrite, cross_check, minimise  (Task 7)
  harness.py        Bytes reader, decode, TestOneInput, main, bucket (Task 8)
  mutation_study.py planted bugs, kill table                         (Task 9)
  ltn_extras.py     LTN-only extensions (spec Section 6)             (Task 11)
  tests/            one test file per module, conftest.py sets paths
```

---

### Task 1: Environment and formula core

**Files:** Create `mr_unified/__init__.py`, `mr_unified/formula.py`, `mr_unified/tests/conftest.py`, `mr_unified/tests/test_formula.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Node: op: str; kids: Tuple[Node, ...] = (); name: str = ""; args: Tuple[str, ...] = (); value: float = 0.0`. `op` in `{"atom","const","not","and_s","or_s","and_w","or_w","imp","forall","exists"}`; for quantifiers `name` is the bound variable.
  - `@dataclass class Env: atoms: Dict[str, float]; preds: Dict[str, Dict[Tuple[str, ...], float]]; domain: Tuple[str, ...]`
  - `size(n: Node) -> int`, `paths(n: Node) -> List[Tuple[int, ...]]`, `get(n, path) -> Node`, `replace(n, path, new) -> Node`
  - `ref_eval(n: Node, env: Env, quant: str = "minmax") -> float` (float64 numpy; `quant` in `{"minmax", "fold"}`; `"fold"` = n-fold strong and / strong or over the domain)
  - `MAX_NODES = 200`

- [ ] **Step 1: Environment.** `conda create -y -n mru python=3.11 && conda activate mru`; `pip install numpy pytest torch --index-url https://download.pytorch.org/whl/cpu` (fall back to default index if it fails); `pip install atheris` (optional; skip if no wheel); `bash lnn/setup_lnn.sh`. Needs network; ask the user before running.
  Verify: `PYTHONPATH=ltn_torch:lnn/src python -c "import ltn, lnn; print('ok')"` prints `ok`.
- [ ] **Step 2: Write failing tests** in `test_formula.py`: `test_ref_eval_luk_connectives` (a=0.7,b=0.6: `and_s`=0.3 within 1e-12, `or_s`=1.0, `imp(0.7,0.6)`=0.9, `not`=0.3), `test_ref_eval_min_max`, `test_replace_roundtrip` (`replace(n, p, get(n, p)) == n` for every path), `test_forall_minmax_vs_fold` (domain 3 objects all 0.5: minmax -> 0.5, fold -> 0.0), `test_size_counts_all_nodes`.
- [ ] **Step 3: Run** `pytest mr_unified/tests/test_formula.py -q`; expect FAIL (module missing).
- [ ] **Step 4: Implement** `formula.py` per the interfaces. Lukasiewicz: `and_s=max(a+b-1,0)`, `or_s=min(a+b,1)`, `imp=min(1-a+b,1)`, `and_w=min`, `or_w=max`, `not=1-a`.
- [ ] **Step 5: Run** the tests; expect PASS. Commit (if asked): `feat: mr_unified formula core`.

### Task 2: Propositional rewrite rules (A1-A10)

**Files:** Create `mr_unified/rewrites.py`, `mr_unified/tests/test_rewrites.py`

**Interfaces:**
- Consumes: `Node`, `ref_eval`, `size`, `MAX_NODES` (Task 1).
- Produces: `@dataclass(frozen=True) class Rule: id: str; source: str; side_condition: str; match: Callable[[Node], bool]; apply: Callable[[Node], Node]`; `RULES: Dict[str, Rule]`; `apply_at(n: Node, rule: Rule, path) -> Optional[Node]` (returns `None` if no match, if result equals input, or if result has more than `MAX_NODES` nodes); `random_formula(rng: random.Random, depth: int, atoms: Sequence[str]) -> Node` (no quantifiers).

Rule ids and rewrites exactly as spec Section 5 A1-A10: `A1_commute` (and_s, or_s, and_w, or_w), `A2_assoc` (re-bracket a 3-way nesting of the same op), `A3_double_neg`, `A4_demorgan` (and_s <-> not or_s of nots, both directions), `A5_imp_def` (`imp(a,b)` <-> `or_s(not a, b)`), `A6_contrapose`, `A7_max_def` (`or_w(a,b)` <-> `imp(imp(a,b),b)`), `A8_prelinear` (`or_w(imp(a,b), imp(b,a))` <-> `const 1`), `A9_divisible` (`and_w(a,b)` <-> `and_s(a, imp(a,b))`), `A10_distance` (`and_s(a, not b)` <-> `not imp(a, b)`, the building block of the MV distance `|a-b| = (a (x) ~b) (+) (b (x) ~a)`; the full distance has no grammar form, so spec A10 is covered through this identity). Builder: verify numerically in the test below before keeping each rule; drop and record in the rule table any rule that fails.

- [ ] **Step 1: Write failing tests:** `test_every_rule_preserves_ref_eval` (for each rule, 300 random formulas depth <= 4 and 3 random atoms, every matching path: `abs(ref_eval(n) - ref_eval(apply_at(...))) <= 1e-12`), `test_noop_rewrite_returns_none` (`apply_at(and_s(a,a), A1_commute, ()) is None`), `test_growth_cap` (a formula of 190 nodes under A9 returns `None`), `test_each_rule_matches_somewhere` (every rule matches at least once in 2000 random formulas, so none is dead).
- [ ] **Step 2: Run;** expect FAIL.
- [ ] **Step 3: Implement** `rewrites.py`; `random_formula` picks ops uniformly, atoms from `atoms`, constants with probability 0.1.
- [ ] **Step 4: Run;** expect PASS. Any rule failing the preservation test is fixed or removed, and the removal noted in a comment above `RULES`.

### Task 3: Property relations and separating relations (A11-A13, S1, S2)

**Files:** Create `mr_unified/properties.py`, `mr_unified/tests/test_properties.py`

**Interfaces:**
- Consumes: `Node`, `Env`, `ref_eval`, an evaluate callable `Callable[[Node, Env], float]`.
- Produces, each returning a margin (0.0 = holds, larger = worse) given `(evaluate, node, env, rng)`:
  - `monotone_margin(...)` (A11): raise one atom by a random delta in the direction that cannot lower the output (polarity computed from the formula: `not` and the antecedent of `imp` flip it); margin = `max(0, before - after)` for an increasing output.
  - `boolean_margin(...)` (A12): atoms snapped to {0,1}; margin = `abs(evaluate - classical truth value)`.
  - `slope_margin(...)` (A13): evaluate at `x`, `x+h`, `x+2h` on one atom with `h = 1/64`; margin = distance of the second difference-quotient from the nearest integer when the segment lies inside one linear piece (checked by equal consecutive differences), and 0 otherwise.
  - `separating_report(evaluate) -> List[str]`: runs S1 (`or_w(a, not a)` at a = 0.5 must be < 1) and S2 (`and_s(a, a)` at a = 0.5 must be < 0.5) and returns messages for any that do NOT fail as expected.

- [ ] **Step 1: Write failing tests** using `ref_eval` as the evaluate callable: `test_monotone_margin_zero_for_ref`, `test_boolean_margin_zero_for_ref`, `test_slope_margin_zero_for_ref`, `test_separating_clean_for_ref` (`separating_report(ref) == []`), `test_separating_flags_minmax_mutant` (an evaluate where `and_s` is replaced by min returns a non-empty report), `test_monotone_margin_detects_flipped_not` (evaluate with `not` = identity gives margin > 0 on some input).
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS.

### Task 4: Quantifier rules (B1-B8)

**Files:** Modify `mr_unified/rewrites.py`; create `mr_unified/tests/test_quantifier_rules.py`

**Interfaces:**
- Consumes: Task 2 `Rule`, `RULES`; Task 1 `ref_eval` with `quant`.
- Produces: `QRULES: Dict[str, Rule]` with `side_condition` in `{"none","dual","minmax","fold","T-based"}`; `valid_for(rule: Rule, declared: Dict[str, str]) -> bool` where `declared = {"forall": "min"|"fold", "exists": "max"|"fold"}`; `random_quantified(rng, depth, domain: Tuple[str, ...]) -> Tuple[Node, Env]` (formulas over unary/binary predicates `P`, `Q` and one or two bound variables).

Rules as spec Section 5 B1-B8: `B1_permute_domain` (a transformation of `Env`, not of the node: implemented as `(node, env) -> (node, env')`, so `Rule.apply` for B1 and B7 takes and returns a pair; document this in the dataclass docstring), `B2_rename_var`, `B3_swap_quantifiers`, `B4_duality` (`not forall x P` <-> `exists x not P`; side `dual`), `B5_vacuous` (`forall x phi` <-> `phi`, x not free; side `minmax`), `B6_instantiate` (a property: `forall <= P(c) <= exists`; side `minmax`), `B7_singleton` (domain of one object: `forall` and `exists` agree), `B8_split` (aggregate over `domain = left + right` equals `and_w`/`and_s` of the two halves per aggregator; side `T-based`).

- [ ] **Step 1: Write failing tests:** `test_rules_preserve_ref_eval_under_declared_semantics` (for `quant` in `("minmax","fold")`, each rule whose side-condition is satisfied preserves `ref_eval` within 1e-12 on 300 random cases), `test_b5_fails_under_fold` (vacuous quantification over 3 objects at value 0.5 changes the value under `"fold"`, so `valid_for(B5, {"forall":"fold","exists":"fold"})` is False), `test_b4_holds_minmax_and_fold`, `test_domain_size_one_edge` (domain of one object: all valid rules still preserve).
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS.

### Task 5: Adapter interface and LTN adapter

**Files:** Create `mr_unified/adapters/__init__.py`, `adapters/base.py`, `adapters/ltn_adapter.py`; test `tests/test_ltn_adapter.py`

**Interfaces:**
- Consumes: `Node`, `Env`, `ref_eval`.
- Produces: `class Adapter(Protocol): name: str; dtype: str; def evaluate(self, node: Node, env: Env) -> float; def declared_quantifiers(self) -> Dict[str, str]`. `class LTNAdapter(dtype: str = "float64")` with `name = "ltn"`; `declared_quantifiers() == {"forall": "min", "exists": "max"}`.

Approach: `ltn.fuzzy_ops` classes `AndLuk`, `OrLuk`, `AndMin`, `OrMax`, `NotStandard`, `ImpliesLuk`; quantifiers via `ltn.Quantifier(ltn.Connective-free aggregator)` using `AggregMin` for forall and a max aggregator for exists (use `torch.amax`-based `ltn.fuzzy_ops.Aggreg`-compatible wrapper if LTN has no max aggregator; builder checks `ltn/fuzzy_ops.py`). Propositional atoms are constants; predicate atoms are tensors over the domain indexed by variable labels, following how `ltn_combined/ltn_fuzz_combined.py` builds `ltn.Variable` / `ltn.Predicate`.

- [ ] **Step 1: Write failing tests** (`pytest.importorskip("ltn")`): `test_ltn_matches_ref_propositional` (300 random formulas, depth <= 5, float64, `abs(adapter - ref) <= 1e-12`), `test_ltn_matches_ref_quantified_minmax`, `test_ltn_edge_atoms` (all atoms 0.0, all 1.0, domain size 1: finite and equal to ref), `test_declared_quantifiers`.
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS. Any mismatch here is a real LTN or adapter bug: triage before moving on.

### Task 6: LNN adapter

**Files:** Create `mr_unified/adapters/lnn_adapter.py`; test `tests/test_lnn_adapter.py`

**Interfaces:**
- Consumes: Task 5 `Adapter`.
- Produces: `class LNNAdapter` with `name = "lnn"`, `dtype = "float32"`; `evaluate` builds `Model()`, `Propositions`, `And/Or/Not/Implies` (strong connectives; weak `and_w`/`or_w` are not expressible and make `evaluate` raise `Unsupported`), adds leaves with point bounds `(v, v)`, runs `infer(Direction.UPWARD, max_steps=60)` and returns the root lower bound after asserting lower == upper within 1e-6. `declared_quantifiers()` returns `{"forall": "fold", "exists": "fold"}` if quantifier tests below pass in closed-world mode, else the adapter exposes no B-group support (returns `{}` and `evaluate` raises `Unsupported` on quantifiers). `class Unsupported(Exception)` in `base.py`.

Follow `lnn/lnn_fuzz.py` (`prop_setup`, `construct`, `stored`, `bounds_of`) for model construction. LNN keys nodes by printed name: build each distinct subformula once and read the root through `stored(m, root)`.

- [ ] **Step 1: Write failing tests** (`pytest.importorskip("lnn")`): `test_lnn_matches_ref_strong_connectives` (random formulas using only `not/and_s/or_s/imp`, depth <= 4, `abs(adapter - ref) <= 1e-5`), `test_lnn_weak_ops_unsupported`, `test_lnn_point_bounds_stay_points`, `test_lnn_quantifier_semantics` (determine empirically whether `Forall` over declared groundings is closed-world; encode the finding in the assertion and in `declared_quantifiers`), `test_deep_formula_within_tolerance` (depth 8, 100 formulas, error <= `tol` from Task 7: write this test after Task 7 exists, as part of Task 7 step 1).
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS. Record in the adapter docstring what LNN's open-world quantifier does (the spec left this unconfirmed).

### Task 7: Oracle, tolerance, minimiser, cross-check

**Files:** Create `mr_unified/oracle.py`; test `tests/test_oracle.py`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces:
  - `def tol(a: Node, b: Node, dtype: str) -> float` = `8 * (size(a) + size(b)) * eps(dtype)` where `eps` from `numpy.finfo`.
  - `@dataclass class Violation: lib: str; rule: str; path: Tuple[int, ...]; before: Node; after: Node; env: Env; got: Tuple[float, float]; margin: float`
  - `def check_rewrite(adapter, rule, node, path, env) -> Optional[Violation]` (returns `None` on pass; an adapter exception other than `Unsupported` becomes a Violation with `margin = inf`).
  - `def check_property(adapter, prop, node, env, rng) -> Optional[Violation]`
  - `def cross_check(adapters: Sequence[Adapter], node, env) -> Optional[Violation]` (pairwise difference above `tol(node, node, lowest precision dtype)`).
  - `def minimise(v: Violation, adapter) -> Violation` (delta-debugging: repeatedly replace a subtree by one of its children or by a constant, keep the change if the violation persists).

- [ ] **Step 1: Write failing tests:** `test_tol_scales_with_size_and_dtype`; `test_check_rewrite_passes_for_ref_adapter` (a `RefAdapter` wrapping `ref_eval`, every rule, 200 cases: all `None`); `test_check_rewrite_catches_planted_bug` (a `RefAdapter` whose `imp` omits the `min(.,1)` clamp makes at least one of A5/A6/A7 return a Violation); `test_exception_becomes_violation`; `test_minimise_shrinks_and_still_violates` (minimised size < original size, `check_rewrite` on it is still a Violation); `test_cross_check_flags_disagreement`. Also add `test_deep_formula_within_tolerance` for LTN and LNN adapters (depth 8, 100 formulas, adapter vs `ref_eval` within `tol(n, n, dtype)`).
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS. If the deep-formula calibration fails on float32, the builder raises the constant 8 to the smallest value that passes 1000 cases and records the value and the measurement in a comment in `tol`.

### Task 8: Byte-driven fuzz harness

**Files:** Create `mr_unified/harness.py`, `mr_unified/replay.py`; test `tests/test_harness.py`

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces: `class Bytes: def __init__(self, data: bytes); def int(self, lo: int, hi: int) -> int; def unit(self) -> float` (reads past the end yield 0 / 0.0); `def decode(data: bytes, adapter) -> Case` where `Case = (node, env, kind, rule_id_or_prop, path)`; `def bucket(margin: float) -> None` (seven `if margin > 1e-15 ... 5e-1: pass` lines copied from `mr_fuzz/mrs.py`, with a comment saying so); `def TestOneInput(data: bytes) -> None`; `def main()` with `--lib ltn|lnn|all`, `--random N`, and Atheris mode when `atheris` imports. Findings go to `mr_unified/findings/<lib>/<rule>/` as `input.bin` plus `meta.json` (rule, path, formula text, env, got, margin), once per `(lib, rule)`. `replay.py <input.bin> --lib <lib>` re-runs `decode` and `check_*` and prints the violation.

- [ ] **Step 1: Write failing tests:** `test_decode_is_deterministic_across_processes` (decode the same bytes in two `subprocess` runs with different `PYTHONHASHSEED`, compare the printed formula), `test_short_input_padded`, `test_random_mode_runs_clean_on_ref` (`--random 500 --lib ref` exits 0, using a `ref` adapter registered for tests only), `test_finding_written_and_replays` (planted-bug adapter writes a finding; `replay.py` on its `input.bin` reproduces the same `rule`).
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS.
- [ ] **Step 5: Smoke run:** `python -m mr_unified.harness --lib ltn --random 20000`; record any real violations in `docs/superpowers/plans/2026-10-26-findings-log.md` with the minimised formula (triage each: adapter bug, tolerance issue, or library bug).

### Task 9: Mutation study

**Files:** Create `mr_unified/mutation_study.py`; test `tests/test_mutation_study.py`

**Interfaces:**
- Consumes: Tasks 5-8.
- Produces: `MUTANTS: Dict[str, Callable[[], ContextManager]]` (context managers that monkeypatch a library operator in memory, as in `mr_fuzz/selftest.py` and `lnn/mutation_check.py`), at least: LTN `ImpliesLuk` without the `min(.,1)` clamp, `AndLuk` using min, `OrLuk` without clamp, `AggregMin` ignoring the last element, `NotStandard` returning `1 - a` clipped at 0.5; LNN: strong `And` computed as min, `Implies` with swapped arguments, `Not` identity. `def run(lib: str, n_cases: int = 2000, seed: int = 0) -> Dict[str, Dict[str, bool]]` returning, per mutant, `{"closed_form": bool, "mr": bool}` where `closed_form` = `abs(adapter - ref_eval)` exceeds tol on any case (the existing-oracle stand-in) and `mr` = any rewrite, property or separating check fires. CLI prints a table with the columns mutant / closed-form / MR / kind (`both`, `MR only`, `closed-form only`, `missed`).

- [ ] **Step 1: Write failing tests:** `test_unmutated_libs_have_no_mr_violations` (fixed seed, 500 cases), `test_every_mutant_is_caught_by_something` (no `missed` row for LTN), `test_table_has_expected_columns`.
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS. Save the printed table to `docs/superpowers/plans/2026-10-26-mutation-table.md` for spec goal G5 / E1.

### Task 10: NeuPSL spike (time-boxed, gated on availability)

**Files:** Create `mr_unified/adapters/neupsl_adapter.py` or `docs/superpowers/plans/2026-10-26-neupsl-blocker.md`

**Interfaces:**
- Produces: if feasible, `class NeuPSLAdapter` with `name = "neupsl"`, `dtype = "float64"`, `evaluate(node, env)` = `1 - distance_to_satisfaction` of the ground rule `true -> phi` with all atoms observed, and `declared_quantifiers()` per what PSL does with universally quantified rules (grounded to separate weighted rules). C-group relations (weight scaling leaves MAP unchanged, rule duplication equals weight doubling, rule order irrelevant, adding a satisfied rule changes nothing) go in `tests/test_neupsl_c_group.py`.

- [ ] **Step 1: Ask the user** where NeuPSL is installed or which repository to use. Do not guess a package name.
- [ ] **Step 2: Establish** whether NeuPSL can be driven from Python in-process (PSL's core is JVM-based per the spec's unverified note). Record the findings: how to build a ground rule with observed atoms, how to read rule satisfaction, how to run MAP.
- [ ] **Step 3: Decide:** if in-process evaluation works, implement the adapter and run `test_ltn_adapter`-style tests (`test_neupsl_matches_ref_strong_connectives`, tolerance 1e-6); otherwise write the blocker note (what was tried, why it fails, what a JVM harness would need) and stop. Either outcome closes the task.
- [ ] **Step 4: Run** the harness with `--lib neupsl` or document that it was not possible.

### Task 11: LTN-only extensions (spec Section 6)

**Files:** Create `mr_unified/ltn_extras.py`; test `tests/test_ltn_extras.py`

**Interfaces:**
- Consumes: `LTNAdapter` and `ltn.fuzzy_ops`.
- Produces `def run_extras(n_cases: int = 500, seed: int = 0) -> List[Violation]` covering: autograd gradient equals the closed form for `AndLuk`, `OrLuk`, `AndMin`, `OrMax`, `AggregMin`, `AggregPMeanError` (van Krieken eq. 14, 15, 19, 32, 33); derivative duality `dS(1-a,1-b)/d(1-a) == dT(a,b)/da` for the Lukasiewicz and min/max pairs; Godel order-invariance (`out(f(x)) == f(out(x))` for `f(t) = t**2` on `AndMin`, `OrMax`, `ImpliesGodel`, `NotGodel`); `AggregPMean(p=1)` equals `AggregMean`; `AggregPMeanError` non-increasing in p, `AggregPMean` non-decreasing in p (stable=False).

- [ ] **Step 1: Write failing tests:** `test_extras_clean_on_unmutated_ltn`; `test_extras_catch_wrong_gradient_mutant` (a mutant whose `AndLuk` backward returns 0 everywhere; extras report a violation).
- [ ] **Step 2: Run;** expect FAIL. **Step 3: Implement.** **Step 4: Run;** expect PASS.

---

## Self-Review

**Spec coverage:** G1 -> Tasks 2-4; G2 -> Tasks 1, 2, 4, 8; G3 -> Tasks 5, 6, 10; G4 -> Tasks 7, 8; G5/E1 -> Task 9; G6 -> Task 7 (`cross_check`) and Task 8 (`--lib all`); E2 -> Task 8 step 5; E3 -> findings log plus mutation table; separating relations S1, S2 -> Task 3; tolerances (Section 7) -> Task 7; LNN C-group -> Task 6 (bounds inclusion is added as a test there if LNN exposes interval inputs: builder adds `test_lnn_interval_inclusion`); NeuPSL C-group -> Task 10; Section 6 LTN extras -> Task 11; risks R1-R4 -> Tasks 10, 6, spec confirmation notes, Task 9.
**Gaps noted:** A8, A9, A13 and van Krieken's implication tables are "to confirm" in the spec; Task 2 and Task 3 verify them numerically against `ref_eval`, which confirms the mathematics but not the source citation.
**Type consistency:** `Node`, `Env`, `Rule`, `Violation`, `Adapter` names and fields are defined once (Tasks 1, 2, 5, 7) and reused unchanged. B1 and B7 are transformations over `(Node, Env)` and are flagged as such in Task 4.
**Proportion:** about 1.7x the spec's length, mostly signatures, test names and commands.
