"""Quantifier rewrite rules B2-B5, environment rules B1 and B7, and the validity of each for declared quantifiers.
Implements the first-order rules of spec Section 5 (B1-B5, B7); valid_for encodes their side-conditions.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

from mr_unified.formula import QUANT_OPS, Env, Node
from mr_unified.rewrites import Rule, not_node


VACUOUS_VAR = "x"


@dataclass(frozen=True)
class EnvRule:
    """A rule on (formula, env) pairs: `apply` returns the rewritten pair; ids start with the B-number."""

    id: str
    source: str
    side_condition: str
    apply: Callable[[Node, Env], Tuple[Node, Env]]


def _mentions(n: Node, var: str) -> bool:
    """Conservative occurrence test: `var` is a predicate argument or a binder somewhere in `n`."""
    if var in n.args or (n.op in QUANT_OPS and n.name == var):
        return True
    return any(_mentions(kid, var) for kid in n.kids)


def substitute_var(n: Node, old: str, new: str) -> Node:
    """Replace the free occurrences of variable `old` by `new` in `n`; a rebinding of `old` shadows."""
    if n.op in QUANT_OPS and n.name == old:
        return n
    args = tuple(new if arg == old else arg for arg in n.args)
    kids = tuple(substitute_var(kid, old, new) for kid in n.kids)
    return Node(op=n.op, kids=kids, name=n.name, args=args, value=n.value)


def _other_var(var: str) -> str:
    """The fresh name for alpha-renaming: y for x, and x for anything else."""
    return "x" if var == "y" else "y"


def _alpha_match(n: Node) -> bool:
    """A quantifier whose fresh partner name does not occur in its body."""
    return n.op in QUANT_OPS and not _mentions(n.kids[0], _other_var(n.name))


def _alpha_apply(n: Node) -> Node:
    """forall x phi == forall y phi[y/x] (and likewise for exists), y fresh in phi."""
    new = _other_var(n.name)
    return Node(op=n.op, kids=(substitute_var(n.kids[0], n.name, new),), name=new)


def _swap_match(n: Node) -> bool:
    """forall x forall y phi with x != y (and likewise for exists)."""
    return n.op in QUANT_OPS and n.kids[0].op == n.op and n.kids[0].name != n.name


def _swap_apply(n: Node) -> Node:
    """forall x forall y phi == forall y forall x phi (and likewise for exists)."""
    inner = n.kids[0]
    return Node(op=n.op, name=inner.name, kids=(Node(op=n.op, name=n.name, kids=inner.kids),))


def _dual_fwd_match(n: Node) -> bool:
    """not forall x phi."""
    return n.op == "not" and n.kids[0].op == "forall"


def _dual_fwd_apply(n: Node) -> Node:
    """not forall x phi == exists x not phi."""
    quant = n.kids[0]
    return Node(op="exists", name=quant.name, kids=(not_node(quant.kids[0]),))


def _dual_bwd_match(n: Node) -> bool:
    """exists x not phi."""
    return n.op == "exists" and n.kids[0].op == "not"


def _dual_bwd_apply(n: Node) -> Node:
    """exists x not phi == not forall x phi."""
    body = n.kids[0].kids[0]
    return not_node(Node(op="forall", name=n.name, kids=(body,)))


def _vacuous_elim_match(n: Node) -> bool:
    """forall x phi with x not occurring in phi."""
    return n.op == "forall" and not _mentions(n.kids[0], n.name)


def _vacuous_elim_apply(n: Node) -> Node:
    """forall x phi == phi when x is not in phi (needs a non-empty domain)."""
    return n.kids[0]


def _vacuous_intro_match(n: Node) -> bool:
    """Any node phi in which the variable VACUOUS_VAR does not occur."""
    return not _mentions(n, VACUOUS_VAR)


def _vacuous_intro_apply(n: Node) -> Node:
    """phi == forall x phi when x is not in phi (needs a non-empty domain)."""
    return Node(op="forall", name=VACUOUS_VAR, kids=(n,))


QRULES: Dict[str, Rule] = {
    rule.id: rule
    for rule in (
        Rule(
            "B2_rename_var",
            "B2: forall x phi == forall y phi[y/x]",
            "none",
            _alpha_match,
            _alpha_apply,
        ),
        Rule(
            "B3_swap_quantifiers",
            "B3: forall x forall y phi == forall y forall x phi",
            "none",
            _swap_match,
            _swap_apply,
        ),
        Rule(
            "B4_duality_fwd",
            "B4: not forall x phi == exists x not phi",
            "dual",
            _dual_fwd_match,
            _dual_fwd_apply,
        ),
        Rule(
            "B4_duality_bwd",
            "B4: exists x not phi == not forall x phi",
            "dual",
            _dual_bwd_match,
            _dual_bwd_apply,
        ),
        Rule(
            "B5_vacuous_fwd",
            "B5: forall x phi == phi, x not in phi",
            "minmax",
            _vacuous_elim_match,
            _vacuous_elim_apply,
        ),
        Rule(
            "B5_vacuous_bwd",
            "B5: phi == forall x phi, x not in phi",
            "minmax",
            _vacuous_intro_match,
            _vacuous_intro_apply,
        ),
    )
}


def _permute_apply(n: Node, env: Env) -> Tuple[Node, Env]:
    """Law B1: a rotation of the domain keeps forall and exists (min/max and fold are order-free)."""
    rotated = env.domain[1:] + env.domain[:1]
    return n, Env(atoms=env.atoms, preds=env.preds, domain=rotated)


def _singleton_apply(n: Node, env: Env) -> Tuple[Node, Env]:
    """Law B7: on a one-object domain forall x phi == exists x phi; returns the pair unchanged otherwise."""
    if n.op != "forall" or len(env.domain) != 1:
        return n, env
    exists_node = Node(op="exists", name=n.name, kids=n.kids)
    return exists_node, env


ENV_RULES: Dict[str, EnvRule] = {
    rule.id: rule
    for rule in (
        EnvRule(
            "B1_permute_domain",
            "B1: permuting the domain leaves forall and exists unchanged",
            "none",
            _permute_apply,
        ),
        EnvRule(
            "B7_singleton",
            "B7: on a one-object domain forall x phi == exists x phi",
            "none",
            _singleton_apply,
        ),
    )
}


def valid_for(rule: Union[Rule, EnvRule], declared: Dict[str, str]) -> bool:
    """Whether `rule` holds when forall uses declared["forall"] and exists uses declared["exists"]."""
    side = rule.side_condition
    if side == "none":
        return True
    if side == "minmax":
        return declared["forall"] == "min" and declared["exists"] == "max"
    if side == "dual":
        return (declared["forall"], declared["exists"]) in (("min", "max"), ("fold", "fold"))
    raise ValueError("unknown side condition: %s" % side)
