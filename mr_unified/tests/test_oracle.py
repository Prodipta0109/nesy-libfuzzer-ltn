"""Tests for the oracle (spec Task 7): tolerance, rewrite and env-rule checks, property checks, cross-check, and the
minimiser. Deep formulas (depth 8) must agree with ref_eval within tol() for the LTN and LNN adapters.
"""
import random
import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pytest

from mr_unified.oracle import ADMM_EPS, TOL_FACTOR, eps_of, tol

from mr_unified.adapters.base import Unsupported
from mr_unified.adapters.ref_adapter import RefAdapter
from mr_unified.formula import Env, Node, paths, ref_eval, size
from mr_unified.oracle import (
    as_quant_evaluate,
    check_env_rule,
    check_property,
    check_rewrite,
    cross_check,
    minimise,
    tol,
)
from mr_unified.properties import boolean_margin, instantiation_margin, monotone_margin
from mr_unified.generators import random_formula, random_quantified
from mr_unified.quantifier_rules import ENV_RULES, QRULES, valid_for
from mr_unified.rewrites import RULES, apply_at

ATOMS = ("p0", "p1", "p2")
PROP_ATOMS = ("p0", "p1")
DOMAIN = ("o0", "o1", "o2")
CASES = 200
DEEP_CASES = 100
WEAK_TO_STRONG = {"and_w": "and_s", "or_w": "or_s"}
EMPTY_ENV = Env(atoms={}, preds={}, domain=())
REF = RefAdapter()


class OffsetAdapter(RefAdapter):
    """Reference values shifted by a fixed offset: a stand-in for a library that disagrees with ref_eval."""

    name = "offset"

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def evaluate(self, node: Node, env: Env) -> float:
        """ref_eval plus the offset."""
        return REF.evaluate(node, env) + self.offset


class UnclampedImpAdapter(RefAdapter):
    """Planted bug: the implication 1 - a + b without the min(., 1) clamp."""

    name = "unclamped"

    def evaluate(self, node: Node, env: Env) -> float:
        """Propositional evaluation with the unclamped implication."""
        return _unclamped(node, env)


class RefusingAdapter(RefAdapter):
    """An adapter that cannot express any formula."""

    name = "refusing"

    def evaluate(self, node: Node, env: Env) -> float:
        """Always raises Unsupported."""
        raise Unsupported("no formula is supported")


class BrokenAdapter(RefAdapter):
    """An adapter whose evaluation crashes with an ordinary error."""

    name = "broken"

    def evaluate(self, node: Node, env: Env) -> float:
        """Always raises RuntimeError."""
        raise RuntimeError("boom")


def _unclamped(n: Node, env: Env) -> float:
    """Propositional truth value with imp = 1 - a + b; every other connective is the reference one."""
    if n.op == "atom":
        return env.atoms[n.name]
    if n.op == "const":
        return n.value
    xs = [_unclamped(kid, env) for kid in n.kids]
    if n.op == "imp":
        return 1.0 - xs[0] + xs[1]
    return ref_eval(Node(op=n.op, kids=tuple(Node(op="const", value=x) for x in xs)), EMPTY_ENV)


def _atom(name: str) -> Node:
    """A propositional atom node."""
    return Node(op="atom", name=name)


def _binary(op: str, left: Node, right: Node) -> Node:
    """A binary connective node."""
    return Node(op=op, kids=(left, right))


def _random_env(rng: random.Random) -> Env:
    """Env with random truth values (0.0, 1.0 or uniform) for every atom in ATOMS; no predicates."""
    return Env(atoms={name: rng.choice([0.0, 1.0, rng.random()]) for name in ATOMS}, preds={}, domain=())


def _prop_pairs(rng: random.Random, count: int) -> List[Tuple[Node, Env]]:
    """Random propositional (node, env) pairs of depth <= 4."""
    return [(random_formula(rng, 4, ATOMS), _random_env(rng)) for _ in range(count)]


def _quant_pairs(rng: random.Random, count: int) -> List[Tuple[Node, Env]]:
    """Random closed quantified (node, env) pairs of depth <= 4 over DOMAIN."""
    return [random_quantified(rng, 4, DOMAIN) for _ in range(count)]


def _count_passing_rewrites(rules: Dict, pairs: Iterable[Tuple[Node, Env]]) -> Dict[str, int]:
    """Every applicable (rule, path) of every pair must pass check_rewrite on REF; returns the count per rule."""
    counts: Dict[str, int] = {}
    for node, env in pairs:
        for rule in rules.values():
            for path in paths(node):
                if apply_at(node, rule, path) is None:
                    continue
                assert check_rewrite(REF, rule, node, path, env) is None, (rule.id, node, path)
                counts[rule.id] = counts.get(rule.id, 0) + 1
    return counts


def _to_strong(node: Node) -> Node:
    """Rewrite weak connectives to strong ones, which LNN supports; the result is a strong formula."""
    kids = tuple(_to_strong(kid) for kid in node.kids)
    op = WEAK_TO_STRONG.get(node.op, node.op)
    return Node(op=op, kids=kids, name=node.name, args=node.args, value=node.value)


def _monotone(adapter, node: Node, env: Env, rng: random.Random) -> float:
    """Property prop(adapter, node, env, rng): the monotonicity margin of the adapter."""
    return monotone_margin(adapter.evaluate, node, env, rng)


def _boolean(adapter, node: Node, env: Env, rng: random.Random) -> float:
    """Property prop(adapter, node, env, rng): the boolean snapping margin of the adapter."""
    return boolean_margin(adapter.evaluate, node, env, rng)


def _instantiation(adapter, node: Node, env: Env, rng: random.Random) -> float:
    """Property prop(adapter, node, env, rng): the instantiation margin (B6) of the adapter, minmax."""
    return instantiation_margin(as_quant_evaluate(adapter), node, env, "minmax", rng)


def test_tol_scales_with_size_and_dtype():
    small = _binary("and_s", _atom("p0"), _atom("p1"))
    large = _binary("and_s", small, small)
    assert tol(small, small, "float64") == (size(small) + size(small)) * np.finfo("float64").eps
    assert tol(large, small, "float64") > tol(small, small, "float64")
    assert tol(small, small, "float32") > tol(small, small, "float64")


def test_check_rewrite_passes_for_ref_adapter():
    rng = random.Random(701)
    counts = _count_passing_rewrites(RULES, _prop_pairs(rng, CASES))
    assert set(counts) == set(RULES)
    assert all(count > 0 for count in counts.values())


def test_check_quantified_rewrite_passes_for_ref_adapter():
    rng = random.Random(702)
    counts = _count_passing_rewrites(QRULES, _quant_pairs(rng, CASES))
    assert all(valid_for(rule, REF.declared_quantifiers()) for rule in QRULES.values())
    assert set(counts) == set(QRULES)
    assert all(count > 0 for count in counts.values())


def test_check_env_rule_b1_passes_for_ref_adapter():
    rng = random.Random(703)
    compared = 0
    for node, env in _quant_pairs(rng, CASES):
        assert check_env_rule(REF, ENV_RULES["B1_permute_domain"], node, env) is None
        compared += 1
    assert compared > 0


def test_check_env_rule_b7_passes_for_ref_adapter():
    rng = random.Random(712)
    compared = 0
    for _ in range(CASES):
        node, env = random_quantified(rng, 4, DOMAIN[:1])
        if node.op != "forall":
            continue
        assert check_env_rule(REF, ENV_RULES["B7_singleton"], node, env) is None
        compared += 1
    assert compared > 0


def test_check_rewrite_catches_planted_bug():
    rng = random.Random(704)
    planted = UnclampedImpAdapter()
    rule_ids = [rule_id for rule_id in RULES if rule_id[:2] in ("A5", "A6", "A7")]
    compared = 0
    caught = 0
    for node, env in _prop_pairs(rng, CASES):
        for rule_id in rule_ids:
            for path in paths(node):
                if apply_at(node, RULES[rule_id], path) is None:
                    continue
                compared += 1
                if check_rewrite(planted, RULES[rule_id], node, path, env) is not None:
                    caught += 1
    assert compared > 0
    assert caught > 0


def test_exception_becomes_violation():
    node = _binary("and_s", _atom("p0"), _atom("p1"))
    env = Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=())
    found = check_rewrite(BrokenAdapter(), RULES["A1_commute"], node, (), env)
    assert found is not None
    assert found.margin == float("inf")


def test_unsupported_propagates():
    node = _binary("and_s", _atom("p0"), _atom("p1"))
    env = Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=())
    with pytest.raises(Unsupported):
        check_rewrite(RefusingAdapter(), RULES["A1_commute"], node, (), env)


def test_check_property_monotone_passes_for_ref_adapter():
    rng = random.Random(705)
    compared = 0
    for node, env in _prop_pairs(rng, CASES):
        assert check_property(REF, _monotone, node, env, rng) is None
        compared += 1
    assert compared > 0


def test_check_property_boolean_passes_for_ref_adapter():
    rng = random.Random(713)
    compared = 0
    for node, env in _prop_pairs(rng, CASES):
        assert check_property(REF, _boolean, node, env, rng) is None
        compared += 1
    assert compared > 0


def test_check_property_flags_margin_above_tolerance():
    node = _binary("and_s", _atom("p0"), _atom("p1"))
    env = Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=())
    found = check_property(REF, lambda adapter, n, e, r: 0.5, node, env, random.Random(706))
    assert found is not None
    assert found.margin == 0.5


def test_check_property_passes_for_instantiation_on_ref_adapter():
    rng = random.Random(707)
    compared = 0
    for node, env in _quant_pairs(rng, CASES):
        if node.op not in ("forall", "exists"):
            continue
        assert check_property(REF, _instantiation, node, env, rng) is None
        compared += 1
    assert compared > 0


def test_cross_check_flags_disagreement():
    rng = random.Random(708)
    compared = 0
    for node, env in _prop_pairs(rng, CASES):
        found = cross_check([REF, OffsetAdapter(0.1)], node, env)
        assert found is not None
        assert found.margin == pytest.approx(0.1)
        compared += 1
    assert compared > 0


def test_cross_check_passes_for_agreeing_adapters():
    rng = random.Random(709)
    compared = 0
    for node, env in _prop_pairs(rng, CASES):
        assert cross_check([REF, OffsetAdapter(0.0)], node, env) is None
        compared += 1
    assert compared > 0


def test_cross_check_skips_unsupported_adapters():
    node = _binary("and_s", _atom("p0"), _atom("p1"))
    env = Env(atoms={"p0": 0.3, "p1": 0.9}, preds={}, domain=())
    assert cross_check([REF, RefusingAdapter()], node, env) is None
    assert cross_check([REF, RefusingAdapter(), OffsetAdapter(0.1)], node, env) is not None


def test_minimise_shrinks_and_still_violates():
    node = _binary("and_s", _binary("imp", _atom("p0"), _atom("p1")), _binary("and_s", _atom("p2"), _atom("p2")))
    env = Env(atoms={"p0": 0.2, "p1": 0.9, "p2": 0.5}, preds={}, domain=())
    planted = UnclampedImpAdapter()
    found = check_rewrite(planted, RULES["A5_imp_def_fwd"], node, (0,), env)
    assert found is not None
    small = minimise(found, planted)
    assert size(small.before) < size(found.before)
    assert check_rewrite(planted, RULES["A5_imp_def_fwd"], small.before, small.path, small.env) is not None


def test_deep_formula_within_tolerance_ltn():
    pytest.importorskip("ltn")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid escape sequence", category=DeprecationWarning)
        from mr_unified.adapters.ltn_adapter import LTNAdapter
    adapter = LTNAdapter()
    rng = random.Random(710)
    compared = 0
    for _ in range(DEEP_CASES):
        node, env = random_quantified(rng, 8, DOMAIN)
        bound = tol(node, node, adapter.dtype_for(node))
        assert abs(adapter.evaluate(node, env) - ref_eval(node, env, "minmax")) <= bound, node
        compared += 1
    assert compared == DEEP_CASES


def test_deep_formula_within_tolerance_lnn():
    pytest.importorskip("lnn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mr_unified.adapters.lnn_adapter import LNNAdapter
    adapter = LNNAdapter()
    rng = random.Random(711)
    compared = 0
    for _ in range(DEEP_CASES):
        node = _to_strong(random_formula(rng, 8, PROP_ATOMS))
        env = _random_env(rng)
        bound = tol(node, node, adapter.dtype_for(node))
        assert abs(adapter.evaluate(node, env) - ref_eval(node, env, "minmax")) <= bound, node
        compared += 1
    assert compared == DEEP_CASES


def test_tol_for_admm_precision_uses_admm_eps_times_sizes():
    """Law: tol(a, b, "admm") == TOL_FACTOR * (size(a) + size(b)) * ADMM_EPS, the empirical PSL precision."""
    a = Node(op="atom", name="p0")
    b = Node(op="and_s", kids=(a, a))
    assert tol(a, b, "admm") == pytest.approx(TOL_FACTOR * (size(a) + size(b)) * 1e-5)
    assert eps_of("admm") == ADMM_EPS


def test_eps_of_float_dtype_is_machine_epsilon():
    """Law: eps_of("float64") is the float64 machine epsilon, unchanged by the admm key."""
    assert eps_of("float64") == pytest.approx(np.finfo("float64").eps)
