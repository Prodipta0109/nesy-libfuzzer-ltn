"""Tests for the LTN adapter (spec Task 5): agreement with ref_eval on propositional and quantified formulas.
Also covers edge atoms (0.0, 1.0, domain size 1), vacuous and empty-domain quantifiers, AggregMax, and dtype_for.
"""
import random
import warnings

import pytest
import torch

from mr_unified.formula import Env, Node, ref_eval
from mr_unified.rewrites import random_formula, random_quantified

# ltn emits DeprecationWarnings for invalid escape sequences in its own docstrings at import time.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="invalid escape sequence", category=DeprecationWarning)
    pytest.importorskip("ltn")
    from mr_unified.adapters.ltn_adapter import AggregMax, LTNAdapter

PROP_ATOMS = ("p0", "p1")
DOMAIN = ("o0", "o1", "o2")
CASES = 300
TOLERANCES = {"float64": 1e-12, "float32": 1e-6}
ADAPTER = LTNAdapter()


def _random_prop_env(rng):
    """Env with random truth values for p0 and p1 only (no predicates, no domain)."""
    atoms = {name: rng.choice([0.0, 1.0, rng.random()]) for name in PROP_ATOMS}
    return Env(atoms=atoms, preds={}, domain=())


def _prop_cases(rng):
    """Yield CASES random propositional (node, env) pairs, depth <= 5."""
    for _ in range(CASES):
        yield random_formula(rng, 5, PROP_ATOMS), _random_prop_env(rng)


def _quantified_cases(rng, domain):
    """Yield CASES random quantified (node, env) pairs over `domain`, depth <= 4."""
    for _ in range(CASES):
        yield random_quantified(rng, 4, domain)


def _constant_cases(rng, value):
    """Yield CASES quantified formulas evaluated in the constant env where every truth value is `value`."""
    for _ in range(CASES):
        node, _ = random_quantified(rng, 4, DOMAIN)  # the generated env is discarded: its truth values are random
        yield node, _constant_env(value, DOMAIN)


def _count_matches(cases) -> int:
    """Check each case is finite and within its dtype's tolerance of ref_eval (minmax); return the count checked."""
    compared = 0
    for node, env in cases:
        ltn_value = ADAPTER.evaluate(node, env)
        ref_value = ref_eval(node, env, "minmax")
        tolerance = TOLERANCES[ADAPTER.dtype_for(node)]
        assert ltn_value == ltn_value and abs(ltn_value) != float("inf"), (node, ltn_value)
        assert abs(ltn_value - ref_value) <= tolerance, (node, ltn_value, ref_value)
        compared += 1
    return compared


def _constant_env(value, domain):
    """Env where every propositional atom and every predicate tuple over `domain` has truth `value`."""
    preds = {
        "P": {(obj,): value for obj in domain},
        "Q": {(obj,): value for obj in domain},
        "R": {(a, b): value for a in domain for b in domain},
    }
    return Env(atoms={name: value for name in PROP_ATOMS}, preds=preds, domain=domain)


def test_ltn_matches_ref_propositional():
    rng = random.Random(501)
    for node, _ in _prop_cases(random.Random(501)):
        assert ADAPTER.dtype_for(node) == "float64"
    assert _count_matches(_prop_cases(rng)) == CASES


def test_ltn_matches_ref_quantified_minmax():
    assert _count_matches(_quantified_cases(random.Random(502), DOMAIN)) == CASES


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_ltn_edge_atoms_constant_truth(value):
    assert _count_matches(_constant_cases(random.Random(503), value)) == CASES


def test_ltn_edge_domain_size_one():
    assert _count_matches(_quantified_cases(random.Random(504), ("o0",))) == CASES


def test_dtype_for_is_float32_with_predicate_atoms():
    predicate = Node(op="atom", name="P", args=("x",))
    node = Node(op="not", kids=(Node(op="and_s", kids=(Node(op="atom", name="p0"), predicate)),))
    assert ADAPTER.dtype_for(node) == "float32"


@pytest.mark.parametrize("op", ["forall", "exists"])
def test_ltn_vacuous_quantifier_equals_body(op):
    body = Node(op="atom", name="p0")
    node = Node(op=op, name="x", kids=(body,))
    env = Env(atoms={"p0": 0.25, "p1": 0.0}, preds={}, domain=DOMAIN)
    assert ADAPTER.evaluate(node, env) == pytest.approx(0.25, abs=TOLERANCES["float64"])


def test_ltn_empty_domain_forall_is_one_and_exists_is_zero():
    atom = Node(op="atom", name="P", args=("x",))
    env = Env(atoms={}, preds={"P": {}}, domain=())
    assert ADAPTER.evaluate(Node(op="forall", name="x", kids=(atom,)), env) == 1.0
    assert ADAPTER.evaluate(Node(op="exists", name="x", kids=(atom,)), env) == 0.0


def test_ltn_free_variable_formula_raises_value_error():
    node = Node(op="atom", name="P", args=("x",))
    env = Env(atoms={}, preds={"P": {("o0",): 0.5}}, domain=("o0",))
    with pytest.raises(ValueError):
        ADAPTER.evaluate(node, env)


def test_aggreg_max_takes_max_over_dim():
    xs = torch.tensor([[0.1, 0.7], [0.4, 0.2]], dtype=torch.float64)
    assert torch.equal(AggregMax()(xs, dim=(0,)), torch.tensor([0.4, 0.7], dtype=torch.float64))


def test_aggreg_max_rejects_guarded_mask():
    xs = torch.tensor([0.1, 0.7], dtype=torch.float64)
    mask = torch.tensor([True, False])
    with pytest.raises(ValueError):
        AggregMax()(xs, dim=(0,), mask=mask)


def test_ltn_adapter_rejects_float32_dtype():
    with pytest.raises(ValueError):
        LTNAdapter("float32")


def test_declared_quantifiers_are_min_and_max():
    assert ADAPTER.declared_quantifiers() == {"forall": "min", "exists": "max"}


def test_ltn_adapter_name_and_dtype():
    assert ADAPTER.name == "ltn"
    assert ADAPTER.dtype == "float64"
