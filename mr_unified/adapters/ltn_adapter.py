"""LTN adapter for the unified metamorphic oracle (spec Task 5): translates a Node into ltn objects and evaluates it.
Connectives are Lukasiewicz (weak and/or are min/max), forall is min, exists is max, computed in float64.
"""
import itertools
from dataclasses import dataclass
from typing import Callable, Dict

import ltn
import torch
from ltn.fuzzy_ops import AggregationOperator, AggregMin, AndLuk, AndMin, ImpliesLuk, NotStandard, OrLuk, OrMax

from mr_unified.formula import QUANT_OPS, Env, Node

DTYPES = {"float64": torch.float64}
CONNECTIVE_OPS = {
    "not": NotStandard,
    "and_s": AndLuk,
    "or_s": OrLuk,
    "imp": ImpliesLuk,
    "and_w": AndMin,
    "or_w": OrMax,
}
QUANTIFIER_CODES = {"forall": "f", "exists": "e"}


class AggregMax(AggregationOperator):
    """Max aggregator with the AggregMin interface: exists x. p(x) == max over x of p(x)."""

    def __repr__(self) -> str:
        return "AggregMax()"

    def __call__(self, xs: torch.Tensor, dim=None, keepdim: bool = False, mask=None) -> torch.Tensor:
        """exists over dim is max: max of `xs` over `dim`; guarded quantification is not used."""
        if mask is not None:
            raise ValueError("AggregMax does not support guarded quantification")
        return torch.amax(xs, dim=dim, keepdim=keepdim)


AGGREGATORS = {"forall": AggregMin, "exists": AggregMax}


@dataclass(frozen=True)
class _Scope:
    """The Env a formula is evaluated in, and the dtype of every tensor built from it."""

    env: Env
    dtype: torch.dtype


class LTNAdapter:
    """Evaluates a Node with LTN, Lukasiewicz connectives and min/max quantifiers."""

    name = "ltn"

    def __init__(self, dtype: str = "float64") -> None:
        if dtype not in DTYPES:
            raise ValueError("unsupported dtype: %s" % dtype)
        self.dtype = dtype

    def declared_quantifiers(self) -> Dict[str, str]:
        """forall is min and exists is max."""
        return {"forall": "min", "exists": "max"}

    def dtype_for(self, node: Node) -> str:
        """float32 when `node` has a predicate atom (ltn.Predicate casts to float32), else float64."""
        return "float32" if _has_predicate(node) else "float64"

    def evaluate(self, node: Node, env: Env) -> float:
        """Translate `node` over `env` and return its truth value; the formula must be closed."""
        result = _translate(node, _Scope(env, DTYPES[self.dtype]))
        if result.free_vars:
            raise ValueError("formula has free variables: %s" % result.free_vars)
        return float(result.value.item())


def _has_predicate(node: Node) -> bool:
    """Whether some atom of `node` takes arguments, i.e. is a predicate atom P(x, ...)."""
    if node.op == "atom" and node.args:
        return True
    return any(_has_predicate(kid) for kid in node.kids)


def _translate(node: Node, scope: _Scope) -> ltn.LTNObject:
    """Dispatch on the node kind to its translation."""
    if node.op == "atom":
        return _atom(node, scope)
    if node.op == "const":
        return _const(node, scope)
    if node.op in QUANT_OPS:
        return _quantify(node, scope)
    return _connect(node, scope)


def _const(node: Node, scope: _Scope) -> ltn.LTNObject:
    """A constant truth value c as an ltn.Constant scalar."""
    return ltn.Constant(torch.tensor(node.value, dtype=scope.dtype))


def _atom(node: Node, scope: _Scope) -> ltn.LTNObject:
    """A propositional atom is a constant; a predicate atom P(x, ...) is applied to its variables."""
    if not node.args:
        return ltn.Constant(torch.tensor(scope.env.atoms[node.name], dtype=scope.dtype))
    return _predicate_atom(node, scope)


def _predicate_atom(node: Node, scope: _Scope) -> ltn.LTNObject:
    """Truth of P(x, ...) read from Env.preds at the domain indices of the variables."""
    table = _predicate_table(node.name, len(node.args), scope)
    predicate = ltn.Predicate(func=_lookup(table))
    return predicate(*[_variable(arg, scope) for arg in node.args])


def _predicate_table(name: str, arity: int, scope: _Scope) -> torch.Tensor:
    """Dense table T[i, j, ...] equal to Env.preds[name] at (domain[i], domain[j], ...)."""
    domain = scope.env.domain
    values = [scope.env.preds[name][key] for key in itertools.product(domain, repeat=arity)]
    return torch.tensor(values, dtype=scope.dtype).reshape((len(domain),) * arity)


def _lookup(table: torch.Tensor) -> Callable[..., torch.Tensor]:
    """A predicate function reading `table` at the domain index of each argument."""

    def func(*index_columns: torch.Tensor) -> torch.Tensor:
        """Table lookup: T[i, j, ...] at the flattened tuple of domain indices, no arithmetic."""
        indices = tuple(column[:, 0].long() for column in index_columns)
        return table[indices]

    return func


def _variable(label: str, scope: _Scope) -> ltn.Variable:
    """An ltn.Variable whose individuals are the domain indices 0, 1, ..., n-1."""
    count = len(scope.env.domain)
    indices = torch.arange(count, dtype=torch.float32).reshape(count, 1)
    return ltn.Variable(label, indices)


def _connect(node: Node, scope: _Scope) -> ltn.LTNObject:
    """Laws: not a == 1 - a; a -> b == min(1 - a + b, 1); a (x) b == max(a + b - 1, 0); weak and/or are min/max."""
    if node.op not in CONNECTIVE_OPS:
        raise ValueError("unknown op: %s" % node.op)
    kids = [_translate(kid, scope) for kid in node.kids]
    return ltn.Connective(CONNECTIVE_OPS[node.op]())(*kids)


def _quantify(node: Node, scope: _Scope) -> ltn.LTNObject:
    """Laws: forall x. p == min over the domain, exists x. p == max; if x is not free in p, both equal p."""
    if not scope.env.domain:
        return _empty_domain_value(node.op, scope)
    body = _translate(node.kids[0], scope)
    if node.name not in body.free_vars:
        return body
    quantifier = ltn.Quantifier(AGGREGATORS[node.op](), QUANTIFIER_CODES[node.op])
    return quantifier(_variable(node.name, scope), body)


def _empty_domain_value(op: str, scope: _Scope) -> ltn.LTNObject:
    """Laws: forall over an empty set == 1.0 (min identity), exists over an empty set == 0.0 (max identity)."""
    value = 1.0 if op == "forall" else 0.0
    return ltn.Constant(torch.tensor(value, dtype=scope.dtype))
