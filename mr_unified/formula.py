"""Formula AST and float64 reference evaluator for the unified metamorphic oracle.
Implements the formula core and reference semantics (Lukasiewicz connectives, min/max or n-fold quantifiers) of spec Task 1.
"""
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

MAX_NODES = 200
QUANT_OPS = ("forall", "exists")


@dataclass(frozen=True)
class Node:
    """One formula node; `name` is the bound variable for quantifiers."""

    op: str
    kids: Tuple["Node", ...] = ()
    name: str = ""
    args: Tuple[str, ...] = ()
    value: float = 0.0


@dataclass
class Env:
    """Truth values for propositional atoms, predicate tables, and the quantifier domain."""

    atoms: Dict[str, float]
    preds: Dict[str, Dict[Tuple[str, ...], float]]
    domain: Tuple[str, ...]


def size(n: Node) -> int:
    """Count every node in the tree, root included."""
    return 1 + sum(size(kid) for kid in n.kids)


def paths(n: Node) -> List[Tuple[int, ...]]:
    """List the child-index path of every node, root is the empty tuple."""
    out = [()]
    for index, kid in enumerate(n.kids):
        out.extend((index,) + sub for sub in paths(kid))
    return out


def get(n: Node, path: Tuple[int, ...]) -> Node:
    """Return the subtree at `path`."""
    for index in path:
        n = n.kids[index]
    return n


def replace(n: Node, path: Tuple[int, ...], new: Node) -> Node:
    """Return a copy of `n` with the subtree at `path` swapped for `new`."""
    if not path:
        return new
    index = path[0]
    kids = n.kids[:index] + (replace(n.kids[index], path[1:], new),) + n.kids[index + 1:]
    return dataclasses.replace(n, kids=kids)


def ref_eval(n: Node, env: Env, quant: str = "minmax") -> float:
    """Evaluate `n` in float64; quant is "minmax" (min/max) or "fold" (n-fold strong and/or)."""
    if quant not in ("minmax", "fold"):
        raise ValueError("unknown quant: %s" % quant)
    return _eval(n, env, quant, {})


def _eval(n: Node, env: Env, quant: str, bind: Dict[str, str]) -> np.float64:
    if n.op == "atom":
        return _atom(n, env, bind)
    if n.op == "const":
        return np.float64(n.value)
    if n.op in QUANT_OPS:
        return _quantify(n, env, quant, bind)
    kids = [_eval(kid, env, quant, bind) for kid in n.kids]
    return _connective(n.op, kids)


def _atom(n: Node, env: Env, bind: Dict[str, str]) -> np.float64:
    if not n.args:
        return np.float64(env.atoms[n.name])
    key = tuple(bind.get(arg, arg) for arg in n.args)
    return np.float64(env.preds[n.name][key])


def _connective(op: str, xs: List[np.float64]) -> np.float64:
    """Laws: not a == 1 - a; a -> b == min(1 - a + b, 1); a /\\w b == min(a, b); a \\/w b == max(a, b)."""
    if op == "not":
        return np.float64(1.0 - xs[0])
    if op == "and_s":
        return _and_s(xs[0], xs[1])
    if op == "or_s":
        return _or_s(xs[0], xs[1])
    if op == "imp":
        return np.float64(min(1.0 - xs[0] + xs[1], 1.0))
    if op == "and_w":
        return np.float64(min(xs[0], xs[1]))
    if op == "or_w":
        return np.float64(max(xs[0], xs[1]))
    raise ValueError("unknown op: %s" % op)


def _and_s(a: float, b: float) -> np.float64:
    """a (x) b == max(a + b - 1, 0)"""
    return np.float64(max(a + b - 1.0, 0.0))


def _or_s(a: float, b: float) -> np.float64:
    """a (+) b == min(a + b, 1)"""
    return np.float64(min(a + b, 1.0))


def _quantify(n: Node, env: Env, quant: str, bind: Dict[str, str]) -> np.float64:
    vals = [_eval(n.kids[0], env, quant, {**bind, n.name: obj}) for obj in env.domain]
    if n.op == "forall":
        return _forall(vals, quant)
    return _exists(vals, quant)


def _forall(vals: List[np.float64], quant: str) -> np.float64:
    if not vals:
        return np.float64(1.0)
    if quant == "minmax":
        return np.float64(min(vals))
    return _fold(vals, _and_s)


def _exists(vals: List[np.float64], quant: str) -> np.float64:
    if not vals:
        return np.float64(0.0)
    if quant == "minmax":
        return np.float64(max(vals))
    return _fold(vals, _or_s)


def _fold(vals: List[np.float64], op: Callable[[float, float], float]) -> np.float64:
    acc = vals[0]
    for val in vals[1:]:
        acc = op(acc, val)
    return acc
