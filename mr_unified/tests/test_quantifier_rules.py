"""Tests for quantifier rules B1-B8 (spec Task 4): rewrites, env rules, instantiation and split properties."""
import itertools
import random
import re

import pytest

from mr_unified.adapters.ref_adapter import RefAdapter
from mr_unified.formula import Env, Node, get, paths, ref_eval
from mr_unified.oracle import check_env_rule
from mr_unified.properties import instantiation_margin, split_margin
from mr_unified.rewrites import ENV_RULES, QRULES, apply_at, random_quantified, valid_for

DOMAIN = ("o0", "o1", "o2")
QUANTS = ("minmax", "fold")
CASES = 300
REF = RefAdapter()
DECLARED = {
    "minmax": {"forall": "min", "exists": "max"},
    "fold": {"forall": "fold", "exists": "fold"},
}


def atom(name):
    return Node(op="atom", name=name)


def pred(name, *args):
    return Node(op="atom", name=name, args=args)


def negation(kid):
    return Node(op="not", kids=(kid,))


def binary(op, left, right):
    return Node(op=op, kids=(left, right))


def forall(var, body):
    return Node(op="forall", name=var, kids=(body,))


def exists(var, body):
    return Node(op="exists", name=var, kids=(body,))


def all_nodes(n):
    return [get(n, path) for path in paths(n)]


def mentions(n, var):
    """True if variable `var` occurs anywhere in `n` (as a predicate argument or a binder)."""
    if var in n.args or (n.op in ("forall", "exists") and n.name == var):
        return True
    return any(mentions(kid, var) for kid in n.kids)


def compared_counts(rule_ids, quant, seed, domain=DOMAIN):
    """Apply each rule at every path of random formulas; assert preservation; return compared counts."""
    rng = random.Random(seed)
    counts = {rule_id: 0 for rule_id in rule_ids}
    for _ in range(CASES):
        formula, env = random_quantified(rng, rng.randint(1, 4), domain)
        for path in paths(formula):
            for rule_id in rule_ids:
                rewritten = apply_at(formula, QRULES[rule_id], path)
                if rewritten is None:
                    continue
                counts[rule_id] += 1
                before = ref_eval(formula, env, quant)
                after = ref_eval(rewritten, env, quant)
                assert abs(before - after) <= 1e-12, (rule_id, quant, before, after)
    return counts


def test_qrules_preserve_ref_eval_under_declared_semantics():
    for quant in QUANTS:
        valid = [rid for rid in QRULES if valid_for(QRULES[rid], DECLARED[quant])]
        assert valid, quant
        counts = compared_counts(valid, quant, seed=11)
        for rule_id in valid:
            assert counts[rule_id] > 0, (quant, rule_id)


def test_b5_fails_under_fold():
    env = Env(atoms={"p0": 0.5}, preds={}, domain=DOMAIN)
    node = forall("x", atom("p0"))
    rewritten = apply_at(node, QRULES["B5_vacuous_fwd"], ())
    assert rewritten == atom("p0")
    assert abs(ref_eval(node, env, "fold") - ref_eval(rewritten, env, "fold")) > 0.1
    assert not valid_for(QRULES["B5_vacuous_fwd"], DECLARED["fold"])


def test_b5_holds_under_minmax():
    env = Env(atoms={"p0": 0.5}, preds={}, domain=DOMAIN)
    node = forall("x", atom("p0"))
    rewritten = apply_at(node, QRULES["B5_vacuous_fwd"], ())
    assert ref_eval(node, env, "minmax") == pytest.approx(ref_eval(rewritten, env, "minmax"))
    assert valid_for(QRULES["B5_vacuous_fwd"], DECLARED["minmax"])


def test_b4_holds_minmax_and_fold():
    rule_ids = ["B4_duality_fwd", "B4_duality_bwd"]
    for quant in QUANTS:
        assert all(valid_for(QRULES[rid], DECLARED[quant]) for rid in rule_ids), quant
        counts = compared_counts(rule_ids, quant, seed=12)
        for rule_id in rule_ids:
            assert counts[rule_id] > 0, (quant, rule_id)


def test_domain_size_one_edge_preserves_valid_rules():
    for quant in QUANTS:
        valid = [rid for rid in QRULES if valid_for(QRULES[rid], DECLARED[quant])]
        counts = compared_counts(valid, quant, seed=13, domain=("o0",))
        for rule_id in valid:
            assert counts[rule_id] > 0, (quant, rule_id)


def test_b2_renames_bound_variable_to_fresh_name():
    node = forall("x", pred("P", "x"))
    assert apply_at(node, QRULES["B2_rename_var"], ()) == forall("y", pred("P", "y"))


def test_b2_refuses_when_new_name_already_occurs():
    node = forall("x", binary("and_s", pred("P", "x"), pred("Q", "y")))
    assert apply_at(node, QRULES["B2_rename_var"], ()) is None


def test_b3_swaps_nested_same_quantifiers():
    node = forall("x", forall("y", pred("R", "x", "y")))
    expected = forall("y", forall("x", pred("R", "x", "y")))
    assert apply_at(node, QRULES["B3_swap_quantifiers"], ()) == expected


def test_b3_refuses_mixed_quantifiers():
    node = forall("x", exists("y", pred("R", "x", "y")))
    assert apply_at(node, QRULES["B3_swap_quantifiers"], ()) is None


def test_b1_permute_domain_preserves_value_for_both_semantics():
    rng = random.Random(21)
    rule = ENV_RULES["B1_permute_domain"]
    compared = 0
    for _ in range(CASES):
        formula, env = random_quantified(rng, rng.randint(1, 4), DOMAIN)
        new_formula, new_env = rule.apply(formula, env)
        assert set(new_env.domain) == set(env.domain)
        for quant in QUANTS:
            compared += 1
            before = ref_eval(formula, env, quant)
            after = ref_eval(new_formula, new_env, quant)
            assert abs(before - after) <= 1e-12, quant
    assert compared > 0


def test_b7_singleton_forall_agrees_with_exists():
    rng = random.Random(22)
    rule = ENV_RULES["B7_singleton"]
    compared = 0
    for _ in range(CASES):
        formula, env = random_quantified(rng, rng.randint(1, 4), DOMAIN[:1])
        if formula.op != "forall":
            continue
        new_formula, new_env = rule.apply(formula, env)
        assert new_formula.op == "exists"
        assert new_env == env
        for quant in QUANTS:
            compared += 1
            before = ref_eval(formula, env, quant)
            after = ref_eval(new_formula, env, quant)
            assert abs(before - after) <= 1e-12, quant
    assert compared > 0


def test_b7_leaves_multi_object_domain_unchanged():
    rng = random.Random(23)
    rule = ENV_RULES["B7_singleton"]
    compared = 0
    for _ in range(CASES):
        formula, env = random_quantified(rng, rng.randint(1, 4), DOMAIN)
        if formula.op != "forall":
            continue
        assert rule.apply(formula, env) == (formula, env)
        assert check_env_rule(REF, rule, formula, env) is None
        compared += 1
    assert compared > 0


def test_b7_refuses_empty_domain():
    env = Env(atoms={}, preds={"P": {}}, domain=())
    node = forall("x", pred("P", "x"))
    new_formula, new_env = ENV_RULES["B7_singleton"].apply(node, env)
    assert new_formula == node
    assert new_env.domain == ()


def test_rule_ids_start_with_their_b_number():
    for rule_id in list(QRULES) + list(ENV_RULES):
        assert re.match(r"B\d+_", rule_id), rule_id


def test_valid_for_minmax_rejects_fold_declaration():
    rule = QRULES["B5_vacuous_fwd"]
    assert valid_for(rule, {"forall": "min", "exists": "max"})
    assert not valid_for(rule, {"forall": "min", "exists": "fold"})


def test_valid_for_dual_rejects_mixed_declaration():
    rule = QRULES["B4_duality_fwd"]
    assert valid_for(rule, {"forall": "min", "exists": "max"})
    assert valid_for(rule, {"forall": "fold", "exists": "fold"})
    assert not valid_for(rule, {"forall": "min", "exists": "fold"})


def test_random_quantified_reaches_required_shapes():
    rng = random.Random(23)
    shapes = {
        "nested_same_quantifier": 0,
        "vacuous_quantifier": 0,
        "not_over_quantifier": 0,
        "exists_over_not": 0,
        "two_variables_in_one_atom": 0,
    }
    for _ in range(CASES):
        formula, _ = random_quantified(rng, 4, DOMAIN)
        for node in all_nodes(formula):
            _count_shapes(node, shapes)
    for name, count in shapes.items():
        assert count > 0, name


def _count_shapes(node, shapes):
    """Add one to each shape counter that `node` exhibits."""
    if node.op in ("forall", "exists"):
        body = node.kids[0]
        shapes["nested_same_quantifier"] += body.op == node.op and body.name != node.name
        shapes["vacuous_quantifier"] += not mentions(body, node.name)
        shapes["exists_over_not"] += node.op == "exists" and body.op == "not"
    shapes["not_over_quantifier"] += node.op == "not" and node.kids[0].op in ("forall", "exists")
    shapes["two_variables_in_one_atom"] += len(node.args) == 2 and node.args[0] != node.args[1]


def test_random_quantified_env_covers_every_tuple():
    rng = random.Random(24)
    _, env = random_quantified(rng, 2, DOMAIN)
    assert set(env.preds["P"]) == {(obj,) for obj in DOMAIN}
    assert set(env.preds["R"]) == set(itertools.product(DOMAIN, repeat=2))


def forall_mutant(node, env, quant):
    """Mutant evaluator: every forall evaluates to 1.0."""
    if node.op == "forall":
        return 1.0
    return ref_eval(node, env, quant)


def fold_mutant(node, env, quant):
    """Mutant evaluator: every quantifier uses the fold law, whatever `quant` says."""
    return ref_eval(node, env, "fold")


def test_instantiation_margin_zero_for_ref():
    checked = 0
    for quant in QUANTS:
        rng = random.Random(31)
        for _ in range(CASES):
            formula, env = random_quantified(rng, rng.randint(1, 4), DOMAIN)
            if formula.op not in ("forall", "exists"):
                continue
            checked += 1
            margin = instantiation_margin(ref_eval, formula, env, quant, rng)
            assert margin <= 1e-12, quant
    assert checked > 0


def test_instantiation_margin_detects_forall_mutant():
    env = Env(atoms={}, preds={"P": {("o0",): 0.3}}, domain=("o0",))
    margin = instantiation_margin(forall_mutant, forall("x", pred("P", "x")), env, "minmax", random.Random(0))
    assert margin == pytest.approx(0.7)


def test_instantiation_margin_zero_on_single_object_domain():
    env = Env(atoms={}, preds={"P": {("o0",): 0.3}}, domain=("o0",))
    margin = instantiation_margin(ref_eval, forall("x", pred("P", "x")), env, "fold", random.Random(0))
    assert margin == pytest.approx(0.0)


def test_instantiation_margin_rejects_empty_domain():
    env = Env(atoms={}, preds={"P": {}}, domain=())
    with pytest.raises(ValueError):
        instantiation_margin(ref_eval, forall("x", pred("P", "x")), env, "minmax", random.Random(0))


def test_split_margin_zero_for_ref():
    checked = 0
    for quant in QUANTS:
        rng = random.Random(41)
        for _ in range(CASES):
            formula, env = random_quantified(rng, rng.randint(1, 4), DOMAIN)
            if formula.op not in ("forall", "exists") or _has_quantifier(formula.kids[0]):
                continue
            checked += 1
            margin = split_margin(ref_eval, formula, env, quant, rng)
            assert margin <= 1e-12, quant
    assert checked > 0


def test_split_margin_detects_fold_mutant_on_minmax():
    env = Env(atoms={}, preds={"P": {("o0",): 0.6, ("o1",): 0.6}}, domain=("o0", "o1"))
    node = forall("x", pred("P", "x"))
    margins = [split_margin(fold_mutant, node, env, "minmax", random.Random(seed)) for seed in range(20)]
    assert max(margins) == pytest.approx(0.4)


def test_split_margin_rejects_nested_quantifier_body():
    env = Env(atoms={}, preds={"P": {("o0",): 0.5}}, domain=("o0",))
    node = forall("x", forall("y", pred("P", "x")))
    with pytest.raises(ValueError):
        split_margin(ref_eval, node, env, "minmax", random.Random(0))


def _has_quantifier(n):
    """True if any node of `n` is a forall or exists."""
    return any(node.op in ("forall", "exists") for node in all_nodes(n))
