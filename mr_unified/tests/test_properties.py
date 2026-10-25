"""Tests for mr_unified.properties: relations A11-A13 and separating checks S1, S2 (spec Task 3)."""
import random

import pytest

from mr_unified.formula import Env, Node, ref_eval
from mr_unified.properties import (
    atom_names,
    boolean_margin,
    monotone_atoms,
    monotone_margin,
    polarities,
    separating_report,
    slope_margin,
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


def test_slope_margin_zero_for_ref():
    checked = 0
    for seed in CASES:
        rng = random.Random(seed)
        node = random_formula(rng, 4, ATOMS)
        env = random_env(rng)
        if atom_names(node):
            checked += 1
        assert slope_margin(ref_eval, node, env, rng) < 1e-9
    assert checked > 0


def test_slope_margin_detects_half_slope_mutant():
    node = binary("and_s", atom("p0"), atom("p1"))
    env = Env(atoms={"p0": 0.5, "p1": 0.9, "p2": 0.0}, preds={}, domain=())
    margin = slope_margin(half_top_and, node, env, random.Random(0))
    assert margin == pytest.approx(0.5)


def test_separating_clean_for_ref():
    assert separating_report(ref_eval) == []


def test_separating_flags_minmax_mutant():
    report = separating_report(min_for_top_and)
    assert len(report) == 1
    assert report[0].startswith("S2")
