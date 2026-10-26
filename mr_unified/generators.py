"""Random formula and quantified-formula generators, with their environments, for the unified metamorphic oracle.
Implements the random generators of spec Tasks 2 and 4: propositional formulas, closed quantified formulas, and envs.
"""
import itertools
import random
from typing import Dict, Sequence, Tuple

from mr_unified.formula import QUANT_OPS, Env, Node, get, paths
from mr_unified.rewrites import CONNECTIVE_OPS, PROP_ATOMS, not_node, op_node


CONST_PROBABILITY = 0.1
LEAF_PROBABILITY = 0.25
COPY_PROBABILITY = 0.3
PATTERN_PROBABILITY = 0.15
QUANT_PROBABILITY = 0.3
VACUOUS_PROBABILITY = 0.2
PROP_ATOM_PROBABILITY = 0.2
BINARY_PRED_PROBABILITY = 0.5
QUANT_VARS = ("x", "y")


def random_formula(rng: random.Random, depth: int, atoms: Sequence[str]) -> Node:
    """Draw a formula of depth <= `depth`: a leaf at depth 0 or with p=0.25, else a connective."""
    if depth == 0 or rng.random() < LEAF_PROBABILITY:
        return _random_leaf(rng, atoms)
    op = rng.choice(CONNECTIVE_OPS)
    if op == "not":
        return _negation_of(rng, depth - 1, atoms)
    left = random_formula(rng, depth - 1, atoms)
    right = _random_right(rng, op, left, depth - 1, atoms)
    return op_node(op, left, right)


def _random_right(
    rng: random.Random, op: str, left: Node, depth: int, atoms: Sequence[str]
) -> Node:
    """Draw the right child: a subtree of `left` (p=0.3), a pattern partner (p=0.15), else fresh."""
    roll = rng.random()
    if roll < COPY_PROBABILITY:
        return _copy_subtree(rng, left)
    if roll < COPY_PROBABILITY + PATTERN_PROBABILITY:
        return _pattern_partner(rng, op, left, depth, atoms)
    return random_formula(rng, depth, atoms)


def _negation_of(rng: random.Random, depth: int, atoms: Sequence[str]) -> Node:
    """Build `not x`; with p=0.15 the child is `not y (+) not z` instead (A4 backward shape)."""
    if rng.random() < PATTERN_PROBABILITY:
        pair = (not_node(random_formula(rng, depth, atoms)), not_node(random_formula(rng, depth, atoms)))
        return not_node(op_node("or_s", *pair))
    return not_node(random_formula(rng, depth, atoms))


def _copy_subtree(rng: random.Random, left: Node) -> Node:
    """Return a uniformly chosen subtree of `left`, including `left` itself, shared by reference."""
    return get(left, rng.choice(paths(left)))


def _pattern_partner(
    rng: random.Random, op: str, left: Node, depth: int, atoms: Sequence[str]
) -> Node:
    """Partner shapes: left -> new (A9), inner right of left (A7), reversed left (A8)."""
    if op == "and_s":
        return op_node("imp", left, random_formula(rng, depth, atoms))
    if op == "imp" and left.op == "imp":
        return left.kids[1]
    if op == "or_w" and left.op == "imp":
        return op_node("imp", left.kids[1], left.kids[0])
    return random_formula(rng, depth, atoms)


def _random_leaf(rng: random.Random, atoms: Sequence[str]) -> Node:
    """Draw a leaf: constant 0.0 or 1.0 with p=0.1, else an atom uniform over `atoms`."""
    if rng.random() < CONST_PROBABILITY:
        return Node(op="const", value=rng.choice([0.0, 1.0]))
    return Node(op="atom", name=rng.choice(atoms))


def random_quantified(rng: random.Random, depth: int, domain: Tuple[str, ...]) -> Tuple[Node, Env]:
    """Draw a closed quantified formula of depth <= `depth` over x, y and an Env over `domain`."""
    return _quant_formula(rng, depth, ()), _random_quant_env(rng, domain)


def _quant_formula(rng: random.Random, depth: int, scope: Tuple[str, ...]) -> Node:
    """Draw a formula whose predicate arguments are all bound in `scope`."""
    if depth == 0 or rng.random() < LEAF_PROBABILITY:
        return _quant_leaf(rng, scope)
    if rng.random() < QUANT_PROBABILITY:
        return _quant_node(rng, depth, scope)
    return _quant_connective(rng, depth, scope)


def _quant_connective(rng: random.Random, depth: int, scope: Tuple[str, ...]) -> Node:
    """Draw a propositional connective over sub-formulas drawn in the same scope."""
    op = rng.choice(CONNECTIVE_OPS)
    if op == "not":
        return not_node(_quant_formula(rng, depth - 1, scope))
    left = _quant_formula(rng, depth - 1, scope)
    right = _quant_formula(rng, depth - 1, scope)
    return op_node(op, left, right)


def _quant_node(rng: random.Random, depth: int, scope: Tuple[str, ...]) -> Node:
    """Draw forall/exists over x or y; with p=VACUOUS_PROBABILITY the body ignores the new variable."""
    op = rng.choice(QUANT_OPS)
    var = rng.choice(QUANT_VARS)
    body_scope = scope if rng.random() < VACUOUS_PROBABILITY else scope + (var,)
    return Node(op=op, name=var, kids=(_quant_formula(rng, depth - 1, body_scope),))


def _quant_leaf(rng: random.Random, scope: Tuple[str, ...]) -> Node:
    """Draw a constant, a propositional atom, or a predicate over variables bound in `scope`."""
    if rng.random() < CONST_PROBABILITY:
        return Node(op="const", value=rng.choice([0.0, 1.0]))
    if not scope or rng.random() < PROP_ATOM_PROBABILITY:
        return Node(op="atom", name=rng.choice(PROP_ATOMS))
    if rng.random() < BINARY_PRED_PROBABILITY:
        return Node(op="atom", name="R", args=(rng.choice(scope), rng.choice(scope)))
    return Node(op="atom", name=rng.choice(("P", "Q")), args=(rng.choice(scope),))


def _random_quant_env(rng: random.Random, domain: Tuple[str, ...]) -> Env:
    """Env with truth values for p0, p1 and for every tuple over `domain` in P, Q (unary) and R (binary)."""
    atoms = {name: _random_truth(rng) for name in PROP_ATOMS}
    preds = {
        "P": _random_table(rng, domain, 1),
        "Q": _random_table(rng, domain, 1),
        "R": _random_table(rng, domain, 2),
    }
    return Env(atoms=atoms, preds=preds, domain=domain)


def _random_table(rng: random.Random, domain: Tuple[str, ...], arity: int) -> Dict[Tuple[str, ...], float]:
    """A random truth value for every tuple of `arity` objects drawn from `domain`."""
    return {tup: _random_truth(rng) for tup in itertools.product(domain, repeat=arity)}


def _random_truth(rng: random.Random) -> float:
    """A truth value: 0.0, 1.0, or a uniform draw, as in the propositional `random_env`."""
    return rng.choice([0.0, 1.0, rng.random()])
