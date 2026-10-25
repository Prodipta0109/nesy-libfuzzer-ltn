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


def two_atom_env():
    return Env(atoms={"a": 0.7, "b": 0.6}, preds={}, domain=())


def test_ref_eval_and_s_is_lukasiewicz_and():
    assert abs(ref_eval(binary("and_s", atom("a"), atom("b")), two_atom_env()) - 0.3) < 1e-12


def test_ref_eval_or_s_is_lukasiewicz_or():
    assert ref_eval(binary("or_s", atom("a"), atom("b")), two_atom_env()) == 1.0


def test_ref_eval_imp_is_lukasiewicz_implication():
    assert abs(ref_eval(binary("imp", atom("a"), atom("b")), two_atom_env()) - 0.9) < 1e-12


def test_ref_eval_not_is_complement():
    assert abs(ref_eval(Node("not", kids=(atom("a"),)), two_atom_env()) - 0.3) < 1e-12


def test_ref_eval_and_w_is_min():
    assert ref_eval(binary("and_w", atom("a"), atom("b")), two_atom_env()) == 0.6


def test_ref_eval_or_w_is_max():
    assert ref_eval(binary("or_w", atom("a"), atom("b")), two_atom_env()) == 0.7


def test_replace_roundtrip():
    formula = sample_formula()
    for path in paths(formula):
        assert replace(formula, path, get(formula, path)) == formula


def test_forall_minmax_vs_fold():
    env = Env(atoms={}, preds={"P": {("a",): 0.5, ("b",): 0.5, ("c",): 0.5}}, domain=("a", "b", "c"))
    forall = Node("forall", kids=(Node("atom", name="P", args=("x",)),), name="x")
    assert ref_eval(forall, env, quant="minmax") == pytest.approx(0.5)
    assert ref_eval(forall, env, quant="fold") == pytest.approx(0.0)


def test_exists_minmax_vs_fold():
    env = Env(atoms={}, preds={"P": {("a",): 0.5, ("b",): 0.5, ("c",): 0.5}}, domain=("a", "b", "c"))
    exists = Node("exists", kids=(Node("atom", name="P", args=("x",)),), name="x")
    assert ref_eval(exists, env, quant="minmax") == pytest.approx(0.5)
    assert ref_eval(exists, env, quant="fold") == pytest.approx(1.0)


def test_empty_domain_quantifiers():
    env = Env(atoms={}, preds={}, domain=())
    forall = Node("forall", kids=(atom("p"),), name="x")
    exists = Node("exists", kids=(atom("p"),), name="x")
    assert ref_eval(forall, env) == pytest.approx(1.0)
    assert ref_eval(exists, env) == pytest.approx(0.0)


def test_unknown_quant_raises_value_error():
    with pytest.raises(ValueError):
        ref_eval(atom("a"), two_atom_env(), quant="bogus")


def test_size_counts_all_nodes():
    assert size(atom("a")) == 1
    assert size(sample_formula()) == 9
