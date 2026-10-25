"""Tests for mr_unified.formula: Lukasiewicz connectives, quantifier semantics and tree paths (spec Task 1)."""
import pytest

from mr_unified.formula import Env, Node, get, paths, ref_eval, replace, size


def atom(name):
    return Node("atom", name=name)


def const(value):
    return Node("const", value=value)


def binary(op, left, right):
    return Node(op, kids=(left, right))


def sample_formula():
    body = binary("imp", Node("atom", name="P", args=("x",)), binary("and_s", atom("b"), const(0.3)))
    return binary("or_w", Node("forall", kids=(body,), name="x"), Node("not", kids=(atom("a"),)))


def test_ref_eval_luk_connectives():
    env = Env(atoms={"a": 0.7, "b": 0.6}, preds={}, domain=())
    assert abs(ref_eval(binary("and_s", atom("a"), atom("b")), env) - 0.3) < 1e-12
    assert ref_eval(binary("or_s", atom("a"), atom("b")), env) == 1.0
    assert abs(ref_eval(binary("imp", atom("a"), atom("b")), env) - 0.9) < 1e-12
    assert abs(ref_eval(Node("not", kids=(atom("a"),)), env) - 0.3) < 1e-12


def test_ref_eval_min_max():
    env = Env(atoms={"a": 0.7, "b": 0.6}, preds={}, domain=())
    assert ref_eval(binary("and_w", atom("a"), atom("b")), env) == 0.6
    assert ref_eval(binary("or_w", atom("a"), atom("b")), env) == 0.7


def test_replace_roundtrip():
    formula = sample_formula()
    for path in paths(formula):
        assert replace(formula, path, get(formula, path)) == formula


def test_forall_minmax_vs_fold():
    env = Env(atoms={}, preds={"P": {("a",): 0.5, ("b",): 0.5, ("c",): 0.5}}, domain=("a", "b", "c"))
    forall = Node("forall", kids=(Node("atom", name="P", args=("x",)),), name="x")
    assert ref_eval(forall, env, quant="minmax") == pytest.approx(0.5)
    assert ref_eval(forall, env, quant="fold") == pytest.approx(0.0)


def test_size_counts_all_nodes():
    assert size(atom("a")) == 1
    assert size(sample_formula()) == 9
