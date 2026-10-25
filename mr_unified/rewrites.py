"""Propositional rewrite rules A1-A10 and random formulas for the unified metamorphic oracle.
Implements the equivalence rules of spec Section 5 (A1-A10) and the random generator of spec Task 2.
"""
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from mr_unified.formula import MAX_NODES, Node, get, paths, replace, size

COMMUTATIVE_OPS = ("and_s", "or_s", "and_w", "or_w")
CONNECTIVE_OPS = ("not", "and_s", "or_s", "imp", "and_w", "or_w")
CONST_PROBABILITY = 0.1
LEAF_PROBABILITY = 0.25
COPY_PROBABILITY = 0.3
PATTERN_PROBABILITY = 0.15


@dataclass(frozen=True)
class Rule:
    """A named equivalence: `match` says where the rule applies, `apply` rewrites that subtree."""

    id: str
    source: str
    side_condition: str
    match: Callable[[Node], bool]
    apply: Callable[[Node], Node]


def _op(op: str, *kids: Node) -> Node:
    """Build a connective node with the given operator and children."""
    return Node(op=op, kids=tuple(kids))


def _not(kid: Node) -> Node:
    """Build the negation node `not kid`."""
    return _op("not", kid)


def _const_one() -> Node:
    """Build the constant node for 1."""
    return Node(op="const", value=1.0)


def _commute_match(n: Node) -> bool:
    """Commutative binary connectives: a op b == b op a."""
    return n.op in COMMUTATIVE_OPS and len(n.kids) == 2


def _a1_apply(n: Node) -> Node:
    """a op b == b op a for op in and_s, or_s, and_w, or_w."""
    return _op(n.op, n.kids[1], n.kids[0])


def _assoc_fwd_match(n: Node) -> bool:
    """(a op b) op c == a op (b op c) for op in and_s, or_s, and_w, or_w, left-nested input."""
    return n.op in COMMUTATIVE_OPS and n.kids[0].op == n.op


def _a2_fwd_apply(n: Node) -> Node:
    """(a op b) op c == a op (b op c) for op in and_s, or_s, and_w, or_w."""
    a, b = n.kids[0].kids
    return _op(n.op, a, _op(n.op, b, n.kids[1]))


def _assoc_bwd_match(n: Node) -> bool:
    """a op (b op c) == (a op b) op c for op in and_s, or_s, and_w, or_w, right-nested input."""
    return n.op in COMMUTATIVE_OPS and n.kids[1].op == n.op


def _a2_bwd_apply(n: Node) -> Node:
    """a op (b op c) == (a op b) op c for op in and_s, or_s, and_w, or_w."""
    b, c = n.kids[1].kids
    return _op(n.op, _op(n.op, n.kids[0], b), c)


def _double_neg_match(n: Node) -> bool:
    """not not a == a."""
    return n.op == "not" and n.kids[0].op == "not"


def _a3_apply(n: Node) -> Node:
    """not not a == a."""
    return n.kids[0].kids[0]


def _demorgan_fwd_match(n: Node) -> bool:
    """a (x) b == not(not a (+) not b): matches the conjunction."""
    return n.op == "and_s"


def _a4_fwd_apply(n: Node) -> Node:
    """a (x) b == not(not a (+) not b)."""
    a, b = n.kids
    return _not(_op("or_s", _not(a), _not(b)))


def _demorgan_bwd_match(n: Node) -> bool:
    """not(not a (+) not b) == a (x) b: matches the negated disjunction of negations."""
    return n.op == "not" and n.kids[0].op == "or_s" and all(k.op == "not" for k in n.kids[0].kids)


def _a4_bwd_apply(n: Node) -> Node:
    """not(not a (+) not b) == a (x) b."""
    a = n.kids[0].kids[0].kids[0]
    b = n.kids[0].kids[1].kids[0]
    return _op("and_s", a, b)


def _imp_def_fwd_match(n: Node) -> bool:
    """a -> b == not a (+) b: matches any implication."""
    return n.op == "imp"


def _a5_fwd_apply(n: Node) -> Node:
    """a -> b == not a (+) b."""
    a, b = n.kids
    return _op("or_s", _not(a), b)


def _imp_def_bwd_match(n: Node) -> bool:
    """not a (+) b == a -> b: matches a disjunction whose first child is a negation."""
    return n.op == "or_s" and n.kids[0].op == "not"


def _a5_bwd_apply(n: Node) -> Node:
    """not a (+) b == a -> b."""
    return _op("imp", n.kids[0].kids[0], n.kids[1])


def _contrapose_match(n: Node) -> bool:
    """a -> b == not b -> not a: matches any implication."""
    return n.op == "imp"


def _a6_apply(n: Node) -> Node:
    """a -> b == not b -> not a."""
    a, b = n.kids
    return _op("imp", _not(b), _not(a))


def _max_def_fwd_match(n: Node) -> bool:
    """a \\/w b == (a -> b) -> b: matches the weak disjunction."""
    return n.op == "or_w"


def _a7_fwd_apply(n: Node) -> Node:
    """a \\/w b == (a -> b) -> b."""
    a, b = n.kids
    return _op("imp", _op("imp", a, b), b)


def _max_def_bwd_match(n: Node) -> bool:
    """(a -> b) -> b == a \\/w b: matches an implication whose antecedent is an implication."""
    return n.op == "imp" and n.kids[0].op == "imp" and n.kids[0].kids[1] == n.kids[1]


def _a7_bwd_apply(n: Node) -> Node:
    """(a -> b) -> b == a \\/w b."""
    a = n.kids[0].kids[0]
    return _op("or_w", a, n.kids[1])


def _prelinear_fwd_match(n: Node) -> bool:
    """(a -> b) \\/w (b -> a) == 1: matches the weak disjunction of crossed implications."""
    if n.op != "or_w":
        return False
    left, right = n.kids
    if left.op != "imp" or right.op != "imp":
        return False
    return left.kids == (right.kids[1], right.kids[0])


def _a8_fwd_apply(n: Node) -> Node:
    """(a -> b) \\/w (b -> a) == 1."""
    return _const_one()


def _divisible_fwd_match(n: Node) -> bool:
    """a /\\w b == a (x) (a -> b): matches the weak conjunction."""
    return n.op == "and_w"


def _a9_fwd_apply(n: Node) -> Node:
    """a /\\w b == a (x) (a -> b)."""
    a, b = n.kids
    return _op("and_s", a, _op("imp", a, b))


def _divisible_bwd_match(n: Node) -> bool:
    """a (x) (a -> b) == a /\\w b: matches a strong conjunction with an implication on a."""
    return n.op == "and_s" and n.kids[1].op == "imp" and n.kids[1].kids[0] == n.kids[0]


def _a9_bwd_apply(n: Node) -> Node:
    """a (x) (a -> b) == a /\\w b."""
    return _op("and_w", n.kids[0], n.kids[1].kids[1])


def _distance_fwd_match(n: Node) -> bool:
    """a (x) not b == not(a -> b): matches a strong conjunction with a negated right child."""
    return n.op == "and_s" and n.kids[1].op == "not"


def _a10_fwd_apply(n: Node) -> Node:
    """a (x) not b == not(a -> b)."""
    a, not_b = n.kids
    return _not(_op("imp", a, not_b.kids[0]))


def _distance_bwd_match(n: Node) -> bool:
    """not(a -> b) == a (x) not b: matches a negated implication."""
    return n.op == "not" and n.kids[0].op == "imp"


def _a10_bwd_apply(n: Node) -> Node:
    """not(a -> b) == a (x) not b."""
    a, b = n.kids[0].kids
    return _op("and_s", a, _not(b))


# Dropped rules: A8_prelinear_bwd (1 == (a -> b) \/w (b -> a)). Its apply() has no formula to take
# a and b from, so the only instance is a = b = 1, a degenerate rewrite. A8_prelinear_fwd is kept.
# A3_double_neg stays elimination-only (~~a -> a), per spec. The other 16 directions pass.
RULES: Dict[str, Rule] = {
    rule.id: rule
    for rule in (
        Rule(
            "A1_commute",
            "A1: a op b == b op a",
            "op in and_s, or_s, and_w, or_w",
            _commute_match,
            _a1_apply,
        ),
        Rule(
            "A2_assoc_fwd",
            "A2: (a op b) op c == a op (b op c)",
            "left-nested same op",
            _assoc_fwd_match,
            _a2_fwd_apply,
        ),
        Rule(
            "A2_assoc_bwd",
            "A2: a op (b op c) == (a op b) op c",
            "right-nested same op",
            _assoc_bwd_match,
            _a2_bwd_apply,
        ),
        Rule(
            "A3_double_neg",
            "A3: not not a == a",
            "none",
            _double_neg_match,
            _a3_apply,
        ),
        Rule(
            "A4_demorgan_fwd",
            "A4: a (x) b == not(not a (+) not b)",
            "none",
            _demorgan_fwd_match,
            _a4_fwd_apply,
        ),
        Rule(
            "A4_demorgan_bwd",
            "A4: not(not a (+) not b) == a (x) b",
            "none",
            _demorgan_bwd_match,
            _a4_bwd_apply,
        ),
        Rule(
            "A5_imp_def_fwd",
            "A5: a -> b == not a (+) b",
            "none",
            _imp_def_fwd_match,
            _a5_fwd_apply,
        ),
        Rule(
            "A5_imp_def_bwd",
            "A5: not a (+) b == a -> b",
            "none",
            _imp_def_bwd_match,
            _a5_bwd_apply,
        ),
        Rule(
            "A6_contrapose",
            "A6: a -> b == not b -> not a",
            "none",
            _contrapose_match,
            _a6_apply,
        ),
        Rule(
            "A7_max_def_fwd",
            "A7: a \\/w b == (a -> b) -> b",
            "none",
            _max_def_fwd_match,
            _a7_fwd_apply,
        ),
        Rule(
            "A7_max_def_bwd",
            "A7: (a -> b) -> b == a \\/w b",
            "none",
            _max_def_bwd_match,
            _a7_bwd_apply,
        ),
        Rule(
            "A8_prelinear_fwd",
            "A8: (a -> b) \\/w (b -> a) == 1",
            "none",
            _prelinear_fwd_match,
            _a8_fwd_apply,
        ),
        Rule(
            "A9_divisible_fwd",
            "A9: a /\\w b == a (x) (a -> b)",
            "none",
            _divisible_fwd_match,
            _a9_fwd_apply,
        ),
        Rule(
            "A9_divisible_bwd",
            "A9: a (x) (a -> b) == a /\\w b",
            "none",
            _divisible_bwd_match,
            _a9_bwd_apply,
        ),
        Rule(
            "A10_distance_fwd",
            "A10: a (x) not b == not(a -> b)",
            "none",
            _distance_fwd_match,
            _a10_fwd_apply,
        ),
        Rule(
            "A10_distance_bwd",
            "A10: not(a -> b) == a (x) not b",
            "none",
            _distance_bwd_match,
            _a10_bwd_apply,
        ),
    )
}


def apply_at(n: Node, rule: Rule, path: Tuple[int, ...]) -> Optional[Node]:
    """Rewrite the subtree at `path` with `rule`; None if no match, no change, or over MAX_NODES."""
    target = get(n, path)
    if not rule.match(target):
        return None
    result = replace(n, path, rule.apply(target))
    if result == n or size(result) > MAX_NODES:
        return None
    return result


def random_formula(rng: random.Random, depth: int, atoms: Sequence[str]) -> Node:
    """Draw a formula of depth <= `depth`: a leaf at depth 0 or with p=0.25, else a connective."""
    if depth == 0 or rng.random() < LEAF_PROBABILITY:
        return _random_leaf(rng, atoms)
    op = rng.choice(CONNECTIVE_OPS)
    if op == "not":
        return _negation_of(rng, depth - 1, atoms)
    left = random_formula(rng, depth - 1, atoms)
    right = _random_right(rng, op, left, depth - 1, atoms)
    return _op(op, left, right)


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
        pair = (_not(random_formula(rng, depth, atoms)), _not(random_formula(rng, depth, atoms)))
        return _not(_op("or_s", *pair))
    return _not(random_formula(rng, depth, atoms))


def _copy_subtree(rng: random.Random, left: Node) -> Node:
    """Return a uniformly chosen subtree of `left`, including `left` itself, shared by reference."""
    return get(left, rng.choice(paths(left)))


def _pattern_partner(
    rng: random.Random, op: str, left: Node, depth: int, atoms: Sequence[str]
) -> Node:
    """Partner shapes: left -> new (A9), inner right of left (A7), reversed left (A8)."""
    if op == "and_s":
        return _op("imp", left, random_formula(rng, depth, atoms))
    if op == "imp" and left.op == "imp":
        return left.kids[1]
    if op == "or_w" and left.op == "imp":
        return _op("imp", left.kids[1], left.kids[0])
    return random_formula(rng, depth, atoms)


def _random_leaf(rng: random.Random, atoms: Sequence[str]) -> Node:
    """Draw a leaf: constant 0.0 or 1.0 with p=0.1, else an atom uniform over `atoms`."""
    if rng.random() < CONST_PROBABILITY:
        return Node(op="const", value=rng.choice([0.0, 1.0]))
    return Node(op="atom", name=rng.choice(atoms))
