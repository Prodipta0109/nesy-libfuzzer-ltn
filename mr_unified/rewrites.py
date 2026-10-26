"""Propositional rewrite rules A1-A10 for the unified metamorphic oracle, and the rule that applies one at a path.
Implements the propositional equivalence rules of spec Section 5 (A1-A10). A10 is the negated implication, a building
block of the MV distance, not the full distance of the spec.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from mr_unified.formula import MAX_NODES, Node, get, replace, size


COMMUTATIVE_OPS = ("and_s", "or_s", "and_w", "or_w")
CONNECTIVE_OPS = ("not", "and_s", "or_s", "imp", "and_w", "or_w")
PROP_ATOMS = ("p0", "p1")


@dataclass(frozen=True)
class Rule:
    """A named equivalence: `match` says where the rule applies, `apply` rewrites that subtree."""

    id: str
    source: str
    side_condition: str
    match: Callable[[Node], bool]
    apply: Callable[[Node], Node]


def op_node(op: str, *kids: Node) -> Node:
    """Build a connective node with the given operator and children."""
    return Node(op=op, kids=tuple(kids))


def not_node(kid: Node) -> Node:
    """Build the negation node `not kid`."""
    return op_node("not", kid)


def _const_one() -> Node:
    """Build the constant node for 1."""
    return Node(op="const", value=1.0)


def _commute_match(n: Node) -> bool:
    """Commutative binary connectives: a op b == b op a."""
    return n.op in COMMUTATIVE_OPS and len(n.kids) == 2


def _a1_apply(n: Node) -> Node:
    """a op b == b op a for op in and_s, or_s, and_w, or_w."""
    return op_node(n.op, n.kids[1], n.kids[0])


def _assoc_fwd_match(n: Node) -> bool:
    """(a op b) op c == a op (b op c) for op in and_s, or_s, and_w, or_w, left-nested input."""
    return n.op in COMMUTATIVE_OPS and n.kids[0].op == n.op


def _a2_fwd_apply(n: Node) -> Node:
    """(a op b) op c == a op (b op c) for op in and_s, or_s, and_w, or_w."""
    a, b = n.kids[0].kids
    return op_node(n.op, a, op_node(n.op, b, n.kids[1]))


def _assoc_bwd_match(n: Node) -> bool:
    """a op (b op c) == (a op b) op c for op in and_s, or_s, and_w, or_w, right-nested input."""
    return n.op in COMMUTATIVE_OPS and n.kids[1].op == n.op


def _a2_bwd_apply(n: Node) -> Node:
    """a op (b op c) == (a op b) op c for op in and_s, or_s, and_w, or_w."""
    b, c = n.kids[1].kids
    return op_node(n.op, op_node(n.op, n.kids[0], b), c)


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
    return not_node(op_node("or_s", not_node(a), not_node(b)))


def _demorgan_bwd_match(n: Node) -> bool:
    """not(not a (+) not b) == a (x) b: matches the negated disjunction of negations."""
    return n.op == "not" and n.kids[0].op == "or_s" and all(k.op == "not" for k in n.kids[0].kids)


def _a4_bwd_apply(n: Node) -> Node:
    """not(not a (+) not b) == a (x) b."""
    a = n.kids[0].kids[0].kids[0]
    b = n.kids[0].kids[1].kids[0]
    return op_node("and_s", a, b)


def _imp_def_fwd_match(n: Node) -> bool:
    """a -> b == not a (+) b: matches any implication."""
    return n.op == "imp"


def _a5_fwd_apply(n: Node) -> Node:
    """a -> b == not a (+) b."""
    a, b = n.kids
    return op_node("or_s", not_node(a), b)


def _imp_def_bwd_match(n: Node) -> bool:
    """not a (+) b == a -> b: matches a disjunction whose first child is a negation."""
    return n.op == "or_s" and n.kids[0].op == "not"


def _a5_bwd_apply(n: Node) -> Node:
    """not a (+) b == a -> b."""
    return op_node("imp", n.kids[0].kids[0], n.kids[1])


def _contrapose_match(n: Node) -> bool:
    """a -> b == not b -> not a: matches any implication."""
    return n.op == "imp"


def _a6_apply(n: Node) -> Node:
    """a -> b == not b -> not a."""
    a, b = n.kids
    return op_node("imp", not_node(b), not_node(a))


def _max_def_fwd_match(n: Node) -> bool:
    """a \\/w b == (a -> b) -> b: matches the weak disjunction."""
    return n.op == "or_w"


def _a7_fwd_apply(n: Node) -> Node:
    """a \\/w b == (a -> b) -> b."""
    a, b = n.kids
    return op_node("imp", op_node("imp", a, b), b)


def _max_def_bwd_match(n: Node) -> bool:
    """(a -> b) -> b == a \\/w b: matches an implication whose antecedent is an implication."""
    return n.op == "imp" and n.kids[0].op == "imp" and n.kids[0].kids[1] == n.kids[1]


def _a7_bwd_apply(n: Node) -> Node:
    """(a -> b) -> b == a \\/w b."""
    a = n.kids[0].kids[0]
    return op_node("or_w", a, n.kids[1])


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
    return op_node("and_s", a, op_node("imp", a, b))


def _divisible_bwd_match(n: Node) -> bool:
    """a (x) (a -> b) == a /\\w b: matches a strong conjunction with an implication on a."""
    return n.op == "and_s" and n.kids[1].op == "imp" and n.kids[1].kids[0] == n.kids[0]


def _a9_bwd_apply(n: Node) -> Node:
    """a (x) (a -> b) == a /\\w b."""
    return op_node("and_w", n.kids[0], n.kids[1].kids[1])


def _neg_implication_fwd_match(n: Node) -> bool:
    """a (x) not b == not(a -> b): matches a strong conjunction with a negated right child."""
    return n.op == "and_s" and n.kids[1].op == "not"


def _a10_fwd_apply(n: Node) -> Node:
    """a (x) not b == not(a -> b)."""
    a, not_b = n.kids
    return not_node(op_node("imp", a, not_b.kids[0]))


def _neg_implication_bwd_match(n: Node) -> bool:
    """not(a -> b) == a (x) not b: matches a negated implication."""
    return n.op == "not" and n.kids[0].op == "imp"


def _a10_bwd_apply(n: Node) -> Node:
    """not(a -> b) == a (x) not b."""
    a, b = n.kids[0].kids
    return op_node("and_s", a, not_node(b))


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
            "A10_neg_implication_fwd",
            "A10: a (x) not b == not(a -> b), a building block of the MV distance",
            "none",
            _neg_implication_fwd_match,
            _a10_fwd_apply,
        ),
        Rule(
            "A10_neg_implication_bwd",
            "A10: not(a -> b) == a (x) not b, a building block of the MV distance",
            "none",
            _neg_implication_bwd_match,
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


def apply_at(n: Node, rule: Rule, path: Tuple[int, ...]) -> Optional[Node]:
    """Rewrite the subtree at `path` with `rule`; None if no match, no change, or over MAX_NODES."""
    target = get(n, path)
    if not rule.match(target):
        return None
    result = replace(n, path, rule.apply(target))
    if result == n or size(result) > MAX_NODES:
        return None
    return result


