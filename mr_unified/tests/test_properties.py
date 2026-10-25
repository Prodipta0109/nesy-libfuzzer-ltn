"""Tests for mr_unified.properties: relations A11-A13 and separating checks S1, S2 (spec Task 3)."""
import random

import numpy as np
import pytest

from mr_unified.adapters.ref_adapter import RefAdapter
from mr_unified.formula import Env, Node, ref_eval
from mr_unified.oracle import eps_of, tol
from mr_unified.properties import (
    atom_names,
    boolean_margin,
    monotone_atoms,
    monotone_margin,
    polarities,
    separating_report,
    slope_margin,
    slope_window,
)
from mr_unified.rewrites import random_formula

ATOMS = ("p0", "p1", "p2")
CASES = range(200)


def atom(name):
    return Node(op="atom", name=name)


def negation(kid):
    return Node(op="not", kids=(kid,))


def binary(op, left, right):
    return Node(op=op, kids=(left, right))


def random_env(rng):
    values = {name: rng.choice([0.0, 1.0, rng.random()]) for name in ATOMS}
    return Env(atoms=values, preds={}, domain=())


def flipped_top_not(node, env):
    """Mutant evaluator: a top-level `not` is replaced by the identity."""
    if node.op == "not":
        return ref_eval(node.kids[0], env)
    return ref_eval(node, env)


def min_for_top_and(node, env):
    """Mutant evaluator: a top-level `and_s` is replaced by min over its children."""
    if node.op == "and_s":
        return min(ref_eval(kid, env) for kid in node.kids)
    return ref_eval(node, env)


def half_top_and(node, env):
    """Mutant evaluator: a top-level `and_s` is halved, so its slope is 0.5."""
    if node.op == "and_s":
        return 0.5 * ref_eval(node, env)
    return ref_eval(node, env)


def half_constant(node, env):
    """Mutant evaluator: every formula evaluates to 0.5."""
    return 0.5


def test_monotone_margin_zero_for_ref():
    checked = 0
    for seed in CASES:
        rng = random.Random(seed)
        node = random_formula(rng, 4, ATOMS)
        env = random_env(rng)
        if monotone_atoms(node):
            checked += 1
        assert monotone_margin(ref_eval, node, env, rng) == 0.0
    assert checked > 0


def test_monotone_margin_detects_flipped_not():
    node = negation(atom("p0"))
    env = Env(atoms={"p0": 0.5}, preds={}, domain=())
    margin = monotone_margin(flipped_top_not, node, env, random.Random(0))
    assert margin > 0.0


def test_monotone_margin_zero_without_atoms():
    node = Node(op="const", value=1.0)
    assert monotone_atoms(node) == []
    assert monotone_margin(ref_eval, node, random_env(random.Random(1)), random.Random(1)) == 0.0


def test_monotone_atoms_skips_mixed_polarity():
    node = binary("and_s", atom("p0"), negation(atom("p0")))
    assert monotone_atoms(node) == []


def test_polarities_imp_antecedent_is_negative():
    node = binary("imp", atom("p0"), atom("p1"))
    assert polarities(node, "p0") == {-1}


def test_polarities_imp_consequent_is_positive():
    node = binary("imp", atom("p0"), atom("p1"))
    assert polarities(node, "p1") == {1}


def test_boolean_margin_zero_for_ref():
    checked = 0
    for seed in CASES:
        rng = random.Random(seed)
        node = random_formula(rng, 4, ATOMS)
        env = random_env(rng)
        if atom_names(node):
            checked += 1
        assert boolean_margin(ref_eval, node, env, rng) == 0.0
    assert checked > 0


def test_boolean_margin_detects_constant_mutant():
    env = Env(atoms={"p0": 1.0}, preds={}, domain=())
    margin = boolean_margin(half_constant, atom("p0"), env, random.Random(0))
    assert margin == pytest.approx(0.5)


def rounded_to_float32(evaluate):
    """Evaluator whose output is rounded through float32, as a float32 library returns it."""

    def rounded(node, env):
        """The wrapped value rounded to float32 precision."""
        return float(np.float32(evaluate(node, env)))

    return rounded


def slope_sweep(evaluate, dtype):
    """Run slope_margin on 5000 random propositional formulas; return (compared, linear, failing margins)."""
    compared = 0
    linear = 0
    failing = []
    for seed in range(5000):
        node = random_formula(random.Random(seed), 4, ATOMS)
        env = random_env(random.Random(seed))
        eps = eps_of(dtype)
        if atom_names(node):
            compared += 1
        if slope_window(evaluate, node, env, random.Random(seed), eps) is not None:
            linear += 1
        margin = slope_margin(evaluate, node, env, random.Random(seed), eps)
        if margin > tol(node, node, dtype):
            failing.append(margin)
    return compared, linear, failing


def test_slope_margin_silent_on_float64_ref_over_random_formulas():
    compared, linear, failing = slope_sweep(RefAdapter().evaluate, "float64")
    assert compared > 0
    assert linear > 0
    assert failing == []


def test_slope_margin_silent_on_float32_rounded_ref_over_random_formulas():
    compared, linear, failing = slope_sweep(rounded_to_float32(ref_eval), "float32")
    assert compared > 0
    assert linear > 0
    assert failing == []


def test_slope_margin_detects_half_slope_mutant_in_float64():
    node = binary("and_s", atom("p0"), atom("p1"))
    env = Env(atoms={"p0": 0.5, "p1": 0.9, "p2": 0.0}, preds={}, domain=())
    margin = slope_margin(half_top_and, node, env, random.Random(0), eps_of("float64"))
    assert margin == pytest.approx(0.5, abs=1e-9)


def test_slope_margin_detects_half_slope_mutant_in_float32_rounded_evaluator():
    node = binary("and_s", atom("p0"), atom("p1"))
    env = Env(atoms={"p0": 0.5, "p1": 0.9, "p2": 0.0}, preds={}, domain=())
    evaluate = rounded_to_float32(half_top_and)
    margin = slope_margin(evaluate, node, env, random.Random(0), eps_of("float32"))
    assert margin > tol(node, node, "float32")
    assert margin == pytest.approx(0.5, abs=1e-3)


def test_slope_margin_is_zero_at_a_point_inside_a_kink():
    node = binary("and_s", atom("p0"), atom("p0"))
    env = Env(atoms={"p0": 0.5 - 1.0 / 128.0}, preds={}, domain=())
    assert slope_window(ref_eval, node, env, random.Random(0), eps_of("float64")) is None
    assert slope_margin(ref_eval, node, env, random.Random(0), eps_of("float64")) == 0.0


def test_separating_clean_for_ref():
    assert separating_report(ref_eval) == []


def test_separating_flags_minmax_mutant():
    report = separating_report(min_for_top_and)
    assert len(report) == 1
    assert report[0].startswith("S2")
