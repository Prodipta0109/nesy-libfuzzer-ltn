"""Tests for mr_unified.rewrites: propositional rewrite rules A1-A10 and random formulas (spec Task 2)."""
import random

from mr_unified.formula import MAX_NODES, Env, Node, get, paths, ref_eval, size
from mr_unified.rewrites import RULES, apply_at, random_formula

ATOMS = ("p0", "p1", "p2")


def atom(name):
    return Node("atom", name=name)


def const(value):
    return Node("const", value=value)


def binary(op, left, right):
    return Node(op, kids=(left, right))


def negation(kid):
    return Node("not", kids=(kid,))


def chain_of_negations(depth, name):
    node = atom(name)
    for _ in range(depth):
        node = negation(node)
    return node


def random_env(rng):
    values = {name: rng.choice([0.0, 1.0, rng.random()]) for name in ATOMS}
    return Env(atoms=values, preds={}, domain=())


# One hand-built formula per rule that the rule matches at its root. Random sampling almost never
# produces the repeated-subformula shapes of A7_max_def_bwd and A9_divisible_bwd, so these cover them.
WITNESSES = {
    "A1_commute": binary("and_s", atom("p0"), atom("p1")),
    "A2_assoc": binary("or_s", binary("or_s", atom("p0"), atom("p1")), atom("p2")),
    "A3_double_neg": negation(negation(atom("p0"))),
    "A4_demorgan_fwd": binary("and_s", atom("p0"), atom("p1")),
    "A4_demorgan_bwd": negation(binary("or_s", negation(atom("p0")), negation(atom("p1")))),
    "A5_imp_def_fwd": binary("imp", atom("p0"), atom("p1")),
    "A5_imp_def_bwd": binary("or_s", negation(atom("p0")), atom("p1")),
    "A6_contrapose": binary("imp", atom("p0"), atom("p1")),
    "A7_max_def_fwd": binary("or_w", atom("p0"), atom("p1")),
    "A7_max_def_bwd": binary("imp", binary("imp", atom("p0"), atom("p1")), atom("p1")),
    "A8_prelinear_fwd": binary("or_w", binary("imp", atom("p0"), atom("p1")), binary("imp", atom("p1"), atom("p0"))),
    "A8_prelinear_bwd": const(1.0),
    "A9_divisible_fwd": binary("and_w", atom("p0"), atom("p1")),
    "A9_divisible_bwd": binary("and_s", atom("p0"), binary("imp", atom("p0"), atom("p1"))),
    "A10_distance_fwd": binary("and_s", atom("p0"), negation(atom("p1"))),
    "A10_distance_bwd": negation(binary("imp", atom("p0"), atom("p1"))),
}


def test_every_rule_has_witness_that_preserves_ref_eval():
    rng = random.Random(6)
    for rule_id, rule in RULES.items():
        witness = WITNESSES[rule_id]
        assert rule.match(witness), rule_id
        rewritten = apply_at(witness, rule, ())
        assert rewritten is not None, rule_id
        for _ in range(300):
            env = random_env(rng)
            assert abs(ref_eval(witness, env) - ref_eval(rewritten, env)) <= 1e-12, rule_id


def test_every_rule_preserves_ref_eval():
    rng = random.Random(2)
    for rule in RULES.values():
        for _ in range(300):
            formula = random_formula(rng, rng.randint(1, 4), ATOMS)
            env = random_env(rng)
            for path in paths(formula):
                if not rule.match(get(formula, path)):
                    continue
                rewritten = apply_at(formula, rule, path)
                if rewritten is None:
                    continue
                before = ref_eval(formula, env)
                after = ref_eval(rewritten, env)
                assert abs(before - after) <= 1e-12, rule.id


def test_noop_rewrite_returns_none():
    formula = binary("and_s", atom("a"), atom("a"))
    assert apply_at(formula, RULES["A1_commute"], ()) is None


def test_growth_cap_returns_none():
    big = binary("and_w", chain_of_negations(94, "a"), chain_of_negations(93, "a"))
    assert size(big) == 190
    assert apply_at(big, RULES["A9_divisible_fwd"], ()) is None


def test_each_rule_matches_somewhere():
    rng = random.Random(3)
    matched = set()
    for _ in range(2000):
        formula = random_formula(rng, rng.randint(1, 4), ATOMS)
        for path in paths(formula):
            node = get(formula, path)
            for rule_id, rule in RULES.items():
                if rule.match(node):
                    matched.add(rule_id)
    for rule_id, rule in RULES.items():
        assert rule_id in matched or rule.match(WITNESSES[rule_id]), rule_id


def test_random_formula_has_no_quantifiers():
    rng = random.Random(4)
    for _ in range(200):
        formula = random_formula(rng, 4, ATOMS)
        for path in paths(formula):
            assert get(formula, path).op not in ("forall", "exists")


def test_random_formula_at_depth_zero_is_a_leaf():
    rng = random.Random(5)
    for _ in range(50):
        assert random_formula(rng, 0, ATOMS).kids == ()


def test_a1_commute_swaps_kids_of_and_s():
    result = apply_at(binary("and_s", atom("p0"), atom("p1")), RULES["A1_commute"], ())
    assert result == binary("and_s", atom("p1"), atom("p0"))


def test_a2_assoc_rebrackets_nested_or_s():
    nested = binary("or_s", binary("or_s", atom("p0"), atom("p1")), atom("p2"))
    result = apply_at(nested, RULES["A2_assoc"], ())
    assert result == binary("or_s", atom("p0"), binary("or_s", atom("p1"), atom("p2")))


def test_a3_double_neg_removes_two_negations():
    result = apply_at(negation(negation(atom("p0"))), RULES["A3_double_neg"], ())
    assert result == atom("p0")


def test_a8_backward_rewrites_constant_one_to_prelinear_form():
    result = apply_at(const(1.0), RULES["A8_prelinear_bwd"], ())
    assert result.op == "or_w"
    assert ref_eval(result, Env(atoms={}, preds={}, domain=())) == 1.0
