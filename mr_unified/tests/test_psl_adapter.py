"""Tests for the PSL adapter (spec Task 10): agreement with ref_eval on the propositional fragment within the ADMM tolerance,
Unsupported outside the fragment and beyond the validated depth, batch equals single, and no temp files left behind.
Run with: venv_psl/bin/python -m pytest mr_unified/tests/test_psl_adapter.py -q
ADMM at 1e-8 over 300 random depth-4 formulas: max error 2.76e-6, mean 1.7e-8 (measured with the validated depth 12).
"""
import random
import sys
import tempfile

import pytest

from mr_unified.formula import Env, Node, ref_eval
from mr_unified.rewrites import random_formula

pytest.importorskip("pslpython")
if "venv_psl" not in sys.prefix:
    pytest.skip("PSL tests run only under venv_psl/bin/python", allow_module_level=True)

from mr_unified.adapters import psl_adapter  # noqa: E402
from mr_unified.adapters.base import Unsupported  # noqa: E402
from mr_unified.adapters.psl_adapter import PSLAdapter  # noqa: E402

PROP_ATOMS = ("p0", "p1", "p2")
CASES = 120
# Batch tolerance: ADMM worst observed error is 2.8e-6 at 1e-8; 2e-5 leaves margin for batches of 120.
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


def _strong_cases(seed: int, count: int = CASES, depth: int = 4):
    """`count` random strong (node, env) pairs of random depth `depth` over p0, p1, p2."""
    rng = random.Random(seed)
    return [(_to_strong(random_formula(rng, depth, PROP_ATOMS)), _random_env(rng)) for _ in range(count)]


def _and_chain(length: int) -> Node:
    """A conjunction chain of `length` strong-and nodes over p1: compile depth exactly `length`."""
    node = Node(op="atom", name="p1")
    for _ in range(length):
        node = Node(op="and_s", kids=(Node(op="atom", name="p0"), node))
    return node


def test_psl_matches_ref_strong_connectives():
    cases = _strong_cases(1001)
    values = ADAPTER.evaluate_many(cases)
    compared = 0
    for value, (node, env) in zip(values, cases):
        assert abs(value - ref_eval(node, env)) <= TOLERANCE, (node, value)
        compared += 1
    assert compared == CASES


def test_psl_negated_only_body_matches_ref():
    node = Node(op="not", kids=(Node(op="or_s", kids=(Node(op="atom", name="p0"), Node(op="atom", name="p1"))),))
    env = Env(atoms={"p0": 0.3, "p1": 0.4, "p2": 0.0}, preds={}, domain=())
    assert abs(ADAPTER.evaluate(node, env) - ref_eval(node, env)) <= TOLERANCE


def test_psl_ground_predicate_atom_matches_ref():
    node = Node(op="and_s", kids=(Node(op="atom", name="P", args=("o0",)), Node(op="atom", name="p0")))
    env = Env(atoms={"p0": 0.8, "p1": 0.0, "p2": 0.0}, preds={"P": {("o0",): 0.9}}, domain=("o0",))
    assert abs(ADAPTER.evaluate(node, env) - ref_eval(node, env)) <= TOLERANCE


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


def test_formula_beyond_validated_depth_is_unsupported():
    too_deep = _and_chain(psl_adapter.MAX_VALIDATED_DEPTH + 1)
    env = Env(atoms={"p0": 0.5, "p1": 0.5, "p2": 0.0}, preds={}, domain=())
    with pytest.raises(Unsupported):
        ADAPTER.evaluate(too_deep, env)


def test_psl_batch_equals_single_runs():
    cases = _strong_cases(1002, count=20)
    batch = ADAPTER.evaluate_many(cases)
    compared = 0
    for got, (node, env) in zip(batch, cases):
        assert abs(got - ADAPTER.evaluate(node, env)) <= TOLERANCE
        compared += 1
    assert compared == 20


def test_deep_formula_in_batch_leaves_shallow_result_unchanged():
    """A formula at the validated depth shares a batch with a shallow one; the shallow value must not move."""
    env = Env(atoms={"p0": 0.9, "p1": 0.8, "p2": 0.1}, preds={}, domain=())
    shallow = Node(op="or_s", kids=(Node(op="atom", name="p0"), Node(op="atom", name="p2")))
    deep = _and_chain(psl_adapter.MAX_VALIDATED_DEPTH)
    alone = ADAPTER.evaluate(shallow, env)
    in_batch = ADAPTER.evaluate_many([(shallow, env), (deep, env)])[0]
    assert abs(in_batch - alone) <= TOLERANCE
    assert abs(in_batch - ref_eval(shallow, env)) <= TOLERANCE


def test_psl_leaves_no_temp_files_behind(tmp_path, monkeypatch):
    """Every solve runs in its own temp directory under tempfile's root and removes it: the root is unchanged afterwards."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    before = sorted(tmp_path.iterdir())
    ADAPTER.evaluate_many(_strong_cases(1004, count=5))
    ADAPTER.evaluate(_and_chain(3), Env(atoms={"p0": 0.5, "p1": 0.5, "p2": 0.0}, preds={}, domain=()))
    assert sorted(tmp_path.iterdir()) == before
