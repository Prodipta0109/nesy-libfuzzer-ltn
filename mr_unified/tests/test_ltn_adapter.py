"""Tests for the LTN adapter (spec Task 5): agreement with ref_eval on propositional and quantified formulas.
Also covers edge atoms (0.0, 1.0, domain size 1), vacuous and empty-domain quantifiers, and declared quantifiers.
"""
import random
import warnings

import pytest

from mr_unified.formula import Env, Node, ref_eval
from mr_unified.rewrites import random_formula, random_quantified

# ltn emits DeprecationWarnings for invalid escape sequences in its own docstrings at import time.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="invalid escape sequence", category=DeprecationWarning)
    pytest.importorskip("ltn")
    from mr_unified.adapters.ltn_adapter import LTNAdapter

PROP_ATOMS = ("p0", "p1")
DOMAIN = ("o0", "o1", "o2")
CASES = 300
TOLERANCE = 1e-12
FLOAT32_TOLERANCE = 1e-6
ADAPTER = LTNAdapter()


def _random_prop_env(rng):
    """Env with random truth values for p0 and p1 only (no predicates, no domain)."""
    atoms = {name: rng.choice([0.0, 1.0, rng.random()]) for name in PROP_ATOMS}
    return Env(atoms=atoms, preds={}, domain=())


def _assert_matches_ref(node, env):
    """Assert the LTN value is finite and within TOLERANCE of ref_eval (minmax semantics)."""
    ltn_value = ADAPTER.evaluate(node, env)
    ref_value = ref_eval(node, env, "minmax")
    assert ltn_value == ltn_value and abs(ltn_value) != float("inf")
    assert abs(ltn_value - ref_value) <= TOLERANCE, (node, ltn_value, ref_value)


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
    compared = 0
    for _ in range(CASES):
        node = random_formula(rng, 5, PROP_ATOMS)
        _assert_matches_ref(node, _random_prop_env(rng))
        compared += 1
    assert compared == CASES


PREDICATE_XFAIL = (
    "LTN bug: ltn.Predicate.__call__ casts its output to float32 (ltn_torch/ltn/core.py:625) "
    "even for float64 inputs, so predicate truth values lose ~1e-8 before the float64 connectives"
)


@pytest.mark.xfail(strict=True, reason=PREDICATE_XFAIL)
def test_ltn_matches_ref_quantified_minmax():
    rng = random.Random(502)
    compared = 0
    for _ in range(CASES):
        node, env = random_quantified(rng, 4, DOMAIN)
        _assert_matches_ref(node, env)
        compared += 1
    assert compared == CASES


def test_ltn_quantified_semantics_within_float32_tolerance():
    rng = random.Random(505)
    compared = 0
    for _ in range(CASES):
        node, env = random_quantified(rng, 4, DOMAIN)
        ltn_value = ADAPTER.evaluate(node, env)
        ref_value = ref_eval(node, env, "minmax")
        assert abs(ltn_value - ref_value) <= FLOAT32_TOLERANCE, (node, ltn_value, ref_value)
        compared += 1
    assert compared == CASES


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_ltn_edge_atoms_constant_truth(value):
    rng = random.Random(503)
    compared = 0
    for _ in range(CASES):
        node, _ = random_quantified(rng, 4, DOMAIN)
        _assert_matches_ref(node, _constant_env(value, DOMAIN))
        compared += 1
    assert compared == CASES


@pytest.mark.xfail(strict=True, reason=PREDICATE_XFAIL)
def test_ltn_edge_domain_size_one():
    rng = random.Random(504)
    compared = 0
    for _ in range(CASES):
        node, env = random_quantified(rng, 4, ("o0",))
        _assert_matches_ref(node, env)
        compared += 1
    assert compared == CASES


def test_ltn_vacuous_quantifier_equals_body():
    body = Node(op="atom", name="p0")
    node = Node(op="forall", name="x", kids=(body,))
    env = Env(atoms={"p0": 0.25, "p1": 0.0}, preds={}, domain=DOMAIN)
    assert ADAPTER.evaluate(node, env) == pytest.approx(0.25, abs=TOLERANCE)


def test_ltn_empty_domain_forall_is_one_and_exists_is_zero():
    atom = Node(op="atom", name="P", args=("x",))
    env = Env(atoms={}, preds={"P": {}}, domain=())
    assert ADAPTER.evaluate(Node(op="forall", name="x", kids=(atom,)), env) == 1.0
    assert ADAPTER.evaluate(Node(op="exists", name="x", kids=(atom,)), env) == 0.0


def test_declared_quantifiers_are_min_and_max():
    assert ADAPTER.declared_quantifiers() == {"forall": "min", "exists": "max"}


def test_ltn_adapter_name_and_dtype():
    assert ADAPTER.name == "ltn"
    assert ADAPTER.dtype == "float64"
