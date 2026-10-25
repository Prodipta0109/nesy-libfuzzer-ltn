"""C-group relations for the PSL adapter (spec Task 10): weight scaling leaves the MAP unchanged; duplicating every rule
equals doubling its weight (the MAP stays within tolerance of ref_eval); a bare atom or constant root is read off directly.
Run with: venv_psl/bin/python -m pytest mr_unified/tests/test_psl_c_group.py -q
"""
import random
import sys

import pytest

from mr_unified.formula import Env, Node, ref_eval
from mr_unified.rewrites import random_formula

pytest.importorskip("pslpython")
if "venv_psl" not in sys.prefix:
    pytest.skip("PSL tests run only under venv_psl/bin/python", allow_module_level=True)

from mr_unified.adapters.psl_adapter import PSLAdapter  # noqa: E402

PROP_ATOMS = ("p0", "p1", "p2")
DEPTH = 4
CASES = 20
TOLERANCE = 2e-5
ADAPTER = PSLAdapter()


def _depth4_cases(seed: int):
    """CASES random strong depth-4 (node, env) pairs over p0, p1, p2 (weak connectives rewritten to strong)."""
    rng = random.Random(seed)
    cases = []
    for _ in range(CASES):
        node = random_formula(rng, DEPTH, PROP_ATOMS)
        node = _strong(node)
        env = Env(atoms={a: rng.choice([0.0, 1.0, rng.random()]) for a in PROP_ATOMS}, preds={}, domain=())
        cases.append((node, env))
    return cases


def _strong(node: Node) -> Node:
    """Rewrite weak connectives to their strong counterparts."""
    op = {"and_w": "and_s", "or_w": "or_s"}.get(node.op, node.op)
    return Node(op=op, kids=tuple(_strong(kid) for kid in node.kids), name=node.name, args=node.args, value=node.value)


def _assert_same(values_a, values_b):
    """Every pair of values agrees within TOLERANCE; returns the number of pairs compared."""
    compared = 0
    for got, want in zip(values_a, values_b):
        assert abs(got - want) <= TOLERANCE, (got, want)
        compared += 1
    return compared


def test_weight_scaling_leaves_map_unchanged():
    """C-group: multiplying every rule and prior weight by 3 does not change the MAP values."""
    cases = _depth4_cases(1003)
    scaled = PSLAdapter(scale=3.0).evaluate_many(cases)
    assert _assert_same(scaled, ADAPTER.evaluate_many(cases)) == CASES


def test_rule_duplication_matches_doubled_weight():
    """C-group: emitting every rule twice (weight 2w each) gives the same MAP as the single rule set and as ref_eval."""
    cases = _depth4_cases(1004)
    duplicated = PSLAdapter(rule_copies=2).evaluate_many(cases)
    assert _assert_same(duplicated, ADAPTER.evaluate_many(cases)) == CASES
    refs = [ref_eval(node, env) for node, env in cases]
    assert _assert_same(duplicated, refs) == CASES


def test_bare_atom_root_matches_ref():
    node = Node(op="atom", name="p0")
    env = Env(atoms={"p0": 0.37, "p1": 0.0, "p2": 0.0}, preds={}, domain=())
    assert abs(ADAPTER.evaluate(node, env) - ref_eval(node, env)) <= TOLERANCE


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_root_matches_ref(value):
    node = Node(op="const", value=value)
    env = Env(atoms={"p0": 0.5, "p1": 0.5, "p2": 0.5}, preds={}, domain=())
    assert abs(ADAPTER.evaluate(node, env) - ref_eval(node, env)) <= TOLERANCE
