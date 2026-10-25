"""Tests for the PSL adapter (spec Task 10): agreement with ref_eval on the propositional fragment, Unsupported outside
it, batch equals single, and the weight-scaling C-group relation. Run with: venv_psl/bin/python -m pytest mr_unified/tests/test_psl_adapter.py
"""
import random
import sys

import pytest

from mr_unified.formula import Env, Node, ref_eval
from mr_unified.rewrites import random_formula

pytest.importorskip("pslpython")
if "venv_psl" not in sys.prefix:
    pytest.skip("PSL tests run only under venv_psl/bin/python", allow_module_level=True)

from mr_unified.adapters.base import Unsupported  # noqa: E402
from mr_unified.adapters.psl_adapter import PSLAdapter  # noqa: E402

PROP_ATOMS = ("p0", "p1", "p2")
CASES = 120
# ADMM is accurate to about 1e-6 typically and 4e-6 worst case on 300 depth-4 formulas (see report); 2e-5 leaves margin.
TOLERANCE = 2e-5
WEAK_TO_STRONG = {"and_w": "and_s", "or_w": "or_s"}
ADAPTER = PSLAdapter()


def _random_env(rng):
    """Truth values for p0, p1, p2 drawn from {0.0, 1.0, uniform(0, 1)} so the edge values are reached often."""
    return Env(atoms={name: rng.choice([0.0, 1.0, rng.random()]) for name in PROP_ATOMS}, preds={}, domain=())


def _to_strong(node: Node) -> Node:
    """Rewrite every weak connective to its strong counterpart, keeping the rest of the tree."""
    kids = tuple(_to_strong(kid) for kid in node.kids)
    return Node(op=WEAK_TO_STRONG.get(node.op, node.op), kids=kids, name=node.name, args=node.args, value=node.value)


def _strong_cases(seed: int):
    """CASES random strong (node, env) pairs of depth <= 4 over p0, p1, p2."""
    rng = random.Random(seed)
    return [(_to_strong(random_formula(rng, 4, PROP_ATOMS)), _random_env(rng)) for _ in range(CASES)]


def test_psl_matches_ref_strong_connectives():
    cases = _strong_cases(1001)
    values = ADAPTER.evaluate_many(cases)
    compared = 0
    for value, (node, env) in zip(values, cases):
        assert abs(value - ref_eval(node, env)) <= TOLERANCE, (node, value)
        compared += 1
    assert compared == CASES


def test_psl_constant_root_matches_ref():
    node = Node(op="imp", kids=(Node(op="const", value=0.0), Node(op="atom", name="p0")))
    env = Env(atoms={"p0": 0.25, "p1": 0.0, "p2": 0.0}, preds={}, domain=())
    assert ADAPTER.evaluate(node, env) == pytest.approx(ref_eval(node, env), abs=TOLERANCE)


def test_psl_negated_only_body_matches_ref():
    node = Node(op="not", kids=(Node(op="or_s", kids=(Node(op="atom", name="p0"), Node(op="atom", name="p1"))),))
    env = Env(atoms={"p0": 0.3, "p1": 0.4, "p2": 0.0}, preds={}, domain=())
    assert ADAPTER.evaluate(node, env) == pytest.approx(ref_eval(node, env), abs=TOLERANCE)


def test_psl_ground_predicate_atom_matches_ref():
    node = Node(op="and_s", kids=(Node(op="atom", name="P", args=("o0",)), Node(op="atom", name="p0")))
    env = Env(atoms={"p0": 0.8, "p1": 0.0, "p2": 0.0}, preds={"P": {("o0",): 0.9}}, domain=("o0",))
    assert ADAPTER.evaluate(node, env) == pytest.approx(ref_eval(node, env), abs=TOLERANCE)


@pytest.mark.parametrize("op", ["and_w", "or_w"])
def test_psl_weak_connective_raises_unsupported(op):
    node = Node(op=op, kids=(Node(op="atom", name="p0"), Node(op="atom", name="p1")))
    with pytest.raises(Unsupported):
        ADAPTER.evaluate(node, Env(atoms={"p0": 0.5, "p1": 0.5, "p2": 0.0}, preds={}, domain=()))


@pytest.mark.parametrize("op", ["forall", "exists"])
def test_psl_quantifier_raises_unsupported(op):
    body = Node(op="atom", name="P", args=("x",))
    node = Node(op=op, name="x", kids=(body,))
    env = Env(atoms={}, preds={"P": {("o0",): 0.5}}, domain=("o0",))
    with pytest.raises(Unsupported):
        ADAPTER.evaluate(node, env)


def test_psl_batch_equals_single_runs():
    cases = _strong_cases(1002)[:20]
    batch = ADAPTER.evaluate_many(cases)
    singles = [ADAPTER.evaluate(node, env) for node, env in cases]
    compared = 0
    for got, want in zip(batch, singles):
        assert abs(got - want) <= TOLERANCE
        compared += 1
    assert compared == 20


def test_psl_weight_scaling_leaves_map_unchanged():
    """C-group: multiplying every rule and prior weight by 3 does not change the MAP values."""
    cases = _strong_cases(1003)[:20]
    scaled = PSLAdapter(scale=3.0).evaluate_many(cases)
    base = ADAPTER.evaluate_many(cases)
    compared = 0
    for got, want in zip(scaled, base):
        assert abs(got - want) <= TOLERANCE
        compared += 1
    assert compared == 20
