"""Tests for the LNN adapter (spec Task 6): agreement with ref_eval(quant="fold") on strong connectives, constants,
edge atoms, closed-world quantifiers over predicate atoms, and the Unsupported cases (weak connectives, odd quantifiers).
"""
import random
import warnings

import pytest

from mr_unified.formula import Env, Node, ref_eval
from mr_unified.oracle import as_quant_evaluate, tol
from mr_unified.properties import instantiation_margin
from mr_unified.rewrites import random_formula

# lnn may emit warnings at import time; they are silenced here, as in lnn/lnn_fuzz.py.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pytest.importorskip("lnn")
    from mr_unified.adapters.base import Unsupported
    from mr_unified.adapters.lnn_adapter import LNNAdapter

PROP_ATOMS = ("p0", "p1")
DOMAIN = ("o0", "o1", "o2")
CASES = 300
TOLERANCE = 1e-5
WEAK_TO_STRONG = {"and_w": "and_s", "or_w": "or_s"}
ADAPTER = LNNAdapter()


def _random_value(rng):
    """A truth value drawn from {0.0, 1.0, uniform(0, 1)} so the edge values are reached often."""
    return rng.choice([0.0, 1.0, rng.random()])


def _random_prop_env(rng):
    """Env with random truth values for p0 and p1 only (no predicates, no domain)."""
    return Env(atoms={name: _random_value(rng) for name in PROP_ATOMS}, preds={}, domain=())


def _to_strong(node: Node) -> Node:
    """Rewrite every weak connective to its strong counterpart; the value of the result is the strong formula's."""
    kids = tuple(_to_strong(kid) for kid in node.kids)
    op = WEAK_TO_STRONG.get(node.op, node.op)
    return Node(op=op, kids=kids, name=node.name, args=node.args, value=node.value)


def _count_matches(cases) -> int:
    """Check each (node, env) against ref_eval in the fold mode within TOLERANCE; return the count compared."""
    compared = 0
    for node, env in cases:
        lnn_value = ADAPTER.evaluate(node, env)
        ref_value = ref_eval(node, env, "fold")
        assert abs(lnn_value - ref_value) <= TOLERANCE, (node, lnn_value, ref_value)
        compared += 1
    return compared


def _prop_cases(rng, atoms_value=None):
    """Yield CASES random strong propositional (node, env) pairs, depth <= 4 (weak nodes rewritten to strong)."""
    for _ in range(CASES):
        env = _random_prop_env(rng) if atoms_value is None else _constant_prop_env(atoms_value)
        yield _to_strong(random_formula(rng, 4, PROP_ATOMS)), env


def _constant_prop_env(value):
    """Env where both propositional atoms have truth `value`."""
    return Env(atoms={name: value for name in PROP_ATOMS}, preds={}, domain=())


def _random_pred_env(rng, domain=DOMAIN):
    """Env with random unary tables for predicates P and Q over `domain`, and random p0, p1."""
    preds = {
        "P": {(obj,): _random_value(rng) for obj in domain},
        "Q": {(obj,): _random_value(rng) for obj in domain},
    }
    return Env(atoms={name: _random_value(rng) for name in PROP_ATOMS}, preds=preds, domain=domain)


def _random_quantified(rng, depth: int) -> Node:
    """Random strong formula over leaves p0, p1, forall x P(x), exists x Q(x), forall x Q(x), exists x P(x)."""
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(_quantified_leaves())
    if rng.random() < 0.25:
        return Node(op="not", kids=(_random_quantified(rng, depth - 1),))
    op = rng.choice(["and_s", "or_s", "imp"])
    return Node(op=op, kids=(_random_quantified(rng, depth - 1), _random_quantified(rng, depth - 1)))


def _quantified_leaves():
    """Leaves of the quantified generator: two atoms and four quantified predicate atoms."""
    x_atom_p = Node(op="atom", name="P", args=("x",))
    x_atom_q = Node(op="atom", name="Q", args=("x",))
    return [
        Node(op="atom", name="p0"),
        Node(op="atom", name="p1"),
        Node(op="forall", name="x", kids=(x_atom_p,)),
        Node(op="exists", name="x", kids=(x_atom_q,)),
        Node(op="forall", name="x", kids=(x_atom_q,)),
        Node(op="exists", name="x", kids=(x_atom_p,)),
    ]


def test_lnn_matches_ref_strong_connectives():
    assert _count_matches(_prop_cases(random.Random(601))) == CASES


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_lnn_edge_atoms_all_zero_or_all_one(value):
    assert _count_matches(_prop_cases(random.Random(602), atoms_value=value)) == CASES


@pytest.mark.parametrize("op", ["and_w", "or_w"])
def test_lnn_weak_connective_raises_unsupported(op):
    node = Node(op=op, kids=(Node(op="atom", name="p0"), Node(op="atom", name="p1")))
    with pytest.raises(Unsupported):
        ADAPTER.evaluate(node, Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=()))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_lnn_constant_leaf_matches_ref(value):
    node = Node(op="imp", kids=(Node(op="const", value=value), Node(op="atom", name="p0")))
    env = Env(atoms={"p0": 0.25, "p1": 0.0}, preds={}, domain=())
    assert ADAPTER.evaluate(node, env) == pytest.approx(ref_eval(node, env, "fold"), abs=TOLERANCE)


def test_lnn_point_bounds_stay_points():
    """evaluate() asserts lower == upper at the root; every random strong formula must pass that check."""
    rng = random.Random(603)
    compared = 0
    for _ in range(CASES):
        ADAPTER.evaluate(_to_strong(random_formula(rng, 4, PROP_ATOMS)), _random_prop_env(rng))
        compared += 1
    assert compared == CASES


def test_lnn_quantifier_semantics():
    """Finding: LNN's default Forall/Exists is open-world; with fully_grounded=True it is closed-world (the fold)."""
    rng = random.Random(604)
    cases = ((_random_quantified(rng, 4), _random_pred_env(rng)) for _ in range(CASES))
    assert _count_matches(cases) == CASES


def test_lnn_forall_is_closed_world_fold_numbers():
    """Closed world on point data: forall over {0.9, 0.8, 0.7} is 0.4 (fold of strong and), not 0.0 (open world)."""
    node = Node(op="forall", name="x", kids=(Node(op="atom", name="P", args=("x",)),))
    env = Env(atoms={}, preds={"P": {("o0",): 0.9, ("o1",): 0.8, ("o2",): 0.7}}, domain=("o0", "o1", "o2"))
    assert ADAPTER.evaluate(node, env) == pytest.approx(0.4, abs=TOLERANCE)


def test_lnn_exists_is_closed_world_fold_numbers():
    """Closed world on point data: exists over {0.5} is 0.5, not 1.0 (open world upper bound)."""
    node = Node(op="exists", name="x", kids=(Node(op="atom", name="P", args=("x",)),))
    env = Env(atoms={}, preds={"P": {("o0",): 0.5}}, domain=("o0",))
    assert ADAPTER.evaluate(node, env) == pytest.approx(0.5, abs=TOLERANCE)


@pytest.mark.parametrize("op, expected", [("forall", 1.0), ("exists", 0.0)])
def test_lnn_empty_domain_quantifier_identity(op, expected):
    node = Node(op=op, name="x", kids=(Node(op="atom", name="P", args=("x",)),))
    assert ADAPTER.evaluate(node, Env(atoms={}, preds={"P": {}}, domain=())) == expected


def test_lnn_quantifier_over_non_predicate_body_is_unsupported():
    node = Node(op="forall", name="x", kids=(Node(op="atom", name="p0"),))
    with pytest.raises(Unsupported):
        ADAPTER.evaluate(node, Env(atoms={"p0": 0.5}, preds={}, domain=DOMAIN))


def test_lnn_free_variable_formula_raises_value_error():
    node = Node(op="atom", name="P", args=("x",))
    env = Env(atoms={}, preds={"P": {("o0",): 0.5}}, domain=("o0",))
    with pytest.raises(ValueError):
        ADAPTER.evaluate(node, env)


def test_declared_quantifiers_are_fold():
    assert ADAPTER.declared_quantifiers() == {"forall": "fold", "exists": "fold"}


def test_dtype_for_is_float32_for_propositional_and_quantified_formulas():
    propositional = Node(op="not", kids=(Node(op="atom", name="p0"),))
    quantified = Node(op="forall", name="x", kids=(Node(op="atom", name="P", args=("x",)),))
    assert ADAPTER.dtype_for(propositional) == "float32"
    assert ADAPTER.dtype_for(quantified) == "float32"


def test_lnn_adapter_name_and_dtype():
    assert ADAPTER.name == "lnn"
    assert ADAPTER.dtype == "float32"


def test_lnn_ground_predicate_atom_equals_table_value():
    env = Env(atoms={}, preds={"P": {("o0",): 0.7, ("o1",): 0.2}}, domain=("o0", "o1"))
    assert abs(ADAPTER.evaluate(Node(op="atom", name="P", args=("o0",)), env) - 0.7) <= TOLERANCE


def test_lnn_instantiation_margin_on_unary_predicate_is_within_tolerance():
    env = Env(atoms={}, preds={"P": {("o0",): 0.7, ("o1",): 0.2}}, domain=("o0", "o1"))
    for quant_op in ("forall", "exists"):
        node = Node(op=quant_op, name="x", kids=(Node(op="atom", name="P", args=("x",)),))
        margin = instantiation_margin(as_quant_evaluate(ADAPTER), node, env, "fold", random.Random(0))
        assert margin <= tol(node, node, ADAPTER.dtype_for(node))


def test_lnn_instantiation_on_binary_predicate_is_unsupported():
    env = Env(atoms={}, preds={"R": {("o0", "o0"): 0.7, ("o1", "o1"): 0.2}}, domain=("o0", "o1"))
    node = Node(op="forall", name="x", kids=(Node(op="atom", name="R", args=("x", "x")),))
    with pytest.raises(Unsupported):
        instantiation_margin(as_quant_evaluate(ADAPTER), node, env, "fold", random.Random(0))


def test_bounds_in_raises_when_the_model_lacks_the_node():
    from types import SimpleNamespace

    from mr_unified.adapters.lnn_adapter import _bounds_in

    with pytest.raises(ValueError):
        _bounds_in(SimpleNamespace(nodes={}), SimpleNamespace(name="missing"))
