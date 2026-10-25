"""Metamorphic oracle for the unified harness (spec Task 7): tolerance, rewrite and property checks, cross-check,
and the minimiser. A check returns None when the law holds and a Violation when it fails. Unsupported propagates.
"""
import itertools
import random
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from mr_unified.adapters.base import Adapter, Unsupported
from mr_unified.formula import Env, Node, get, paths, replace, size
from mr_unified.rewrites import ENV_RULES, QRULES, RULES, EnvRule, Rule, apply_at

# Calibration: over 1000 random depth-8 cases the worst |adapter - ref_eval| needed only k = 0.12 (LNN, float32) and
# k = 0.03 (LTN) times 2 * size * eps. k = 8 passes with a wide margin, so it stays (see task-7-report.md).
TOL_FACTOR = 8
NO_VALUE = (float("nan"), float("nan"))
PropertyCheck = Callable[[Adapter, Node, Env, random.Random], float]


@dataclass
class Violation:
    """One failed law: the library and rule, where it applied, both sides, the env, both values and the margin."""

    lib: str
    rule: str
    path: Tuple[int, ...]
    before: Node
    after: Node
    env: Env
    got: Tuple[float, float]
    margin: float


def tol(a: Node, b: Node, dtype: str) -> float:
    """Tolerance for comparing a and b computed in `dtype`: 8 * (size(a) + size(b)) * eps(dtype)."""
    return TOL_FACTOR * (size(a) + size(b)) * float(np.finfo(dtype).eps)


def check_rewrite(adapter: Adapter, rule: Rule, node: Node, path: Tuple[int, ...], env: Env) -> Optional[Violation]:
    """Law of `rule` at `path`: phi == rewrite(phi) in `env`. None if the rule does not apply or the law holds.
    An adapter exception other than Unsupported becomes a Violation with margin inf; Unsupported propagates.
    """
    after = apply_at(node, rule, path)
    if after is None:
        return None
    return _compare(adapter, rule.id, path, node, env, after, env)


def check_env_rule(adapter: Adapter, rule: EnvRule, node: Node, env: Env) -> Optional[Violation]:
    """Law of an env rule (B1, B7): phi in env == rewritten phi in rewritten env.
    None if nothing changed or the law holds. Exceptions and Unsupported are handled as in check_rewrite.
    """
    after, after_env = rule.apply(node, env)
    if after == node and after_env == env:
        return None
    return _compare(adapter, rule.id, (), node, env, after, after_env)


def check_property(
    adapter: Adapter, prop: PropertyCheck, node: Node, env: Env, rng: random.Random, name: str = "property"
) -> Optional[Violation]:
    """Law checked by a margin: `prop` returns how far the adapter is from the law; above tol(node, node, dtype) fails.
    Exceptions raised by `prop` itself are not caught: they are harness errors, not adapter answers.
    """
    margin = prop(adapter, node, env, rng)
    if margin <= tol(node, node, adapter.dtype_for(node)):
        return None
    return _violation(adapter, name, (), node, node, env, NO_VALUE, margin)


def as_quant_evaluate(adapter: Adapter) -> Callable[[Node, Env, str], float]:
    """Wrap adapter.evaluate in the (node, env, quant) shape of instantiation_margin and split_margin.
    The quant argument is ignored: the adapter fixes its own quantifier semantics."""

    def evaluate(node: Node, env: Env, quant: str) -> float:
        """The adapter's own value: the adapter fixes its quantifier semantics, so `quant` is not used."""
        return adapter.evaluate(node, env)

    return evaluate


def cross_check(adapters: Sequence[Adapter], node: Node, env: Env) -> Optional[Violation]:
    """Law: every two adapters agree on phi in env within tol(node, node, widest dtype). Adapters raising Unsupported
    are left out; with fewer than two answers there is nothing to compare and the result is None.
    """
    answered = _answers(adapters, node, env)
    if len(answered) < 2:
        return None
    bound = tol(node, node, _widest_dtype([adapter.dtype_for(node) for adapter, _ in answered]))
    for (left, left_value), (right, right_value) in itertools.combinations(answered, 2):
        if abs(left_value - right_value) > bound:
            return Violation(
                lib="%s vs %s" % (left.name, right.name),
                rule="cross_check",
                path=(),
                before=node,
                after=node,
                env=env,
                got=(left_value, right_value),
                margin=abs(left_value - right_value),
            )
    return None


def minimise(v: Violation, adapter: Adapter) -> Violation:
    """Delta-debug `v.before`: swap subtrees for children or constants while the same check still fails; fixpoint."""
    current = v
    while True:
        smaller = _first_smaller_violation(current, adapter)
        if smaller is None:
            return current
        current = smaller


def _compare(
    adapter: Adapter,
    rule_id: str,
    path: Tuple[int, ...],
    before: Node,
    before_env: Env,
    after: Node,
    after_env: Env,
) -> Optional[Violation]:
    """Law: both sides agree within tol(before, after, dtype_for(before)); exceptions other than Unsupported fail."""
    try:
        got = (adapter.evaluate(before, before_env), adapter.evaluate(after, after_env))
    except Unsupported:
        raise
    except Exception:
        return _violation(adapter, rule_id, path, before, after, before_env, NO_VALUE, float("inf"))
    margin = abs(got[0] - got[1])
    if margin <= tol(before, after, adapter.dtype_for(before)):
        return None
    return _violation(adapter, rule_id, path, before, after, before_env, got, margin)


def _violation(
    adapter: Adapter,
    rule_id: str,
    path: Tuple[int, ...],
    before: Node,
    after: Node,
    env: Env,
    got: Tuple[float, float],
    margin: float,
) -> Violation:
    """Build the Violation for `adapter`; its name is the lib field."""
    return Violation(
        lib=adapter.name, rule=rule_id, path=path, before=before, after=after, env=env, got=got, margin=margin
    )


def _answers(adapters: Sequence[Adapter], node: Node, env: Env) -> List[Tuple[Adapter, float]]:
    """Each adapter's value for `node` in `env`, leaving out the adapters that raise Unsupported."""
    answered = []
    for adapter in adapters:
        try:
            answered.append((adapter, adapter.evaluate(node, env)))
        except Unsupported:
            pass
    return answered


def _widest_dtype(dtypes: List[str]) -> str:
    """The least precise of `dtypes`: the one with the largest machine epsilon."""
    return max(dtypes, key=lambda dtype: np.finfo(dtype).eps)


def _first_smaller_violation(v: Violation, adapter: Adapter) -> Optional[Violation]:
    """The first candidate smaller than v.before on which the same check still fails; None at a fixpoint."""
    for candidate in _shrinks(v.before):
        found = _recheck(adapter, v, candidate)
        if found is not None:
            return found
    return None


def _shrinks(node: Node) -> Iterator[Node]:
    """Candidate formulas: each subtree replaced by one of its children or by const 0 or 1, if strictly smaller."""
    for path in paths(node):
        for replacement in _replacements(get(node, path)):
            candidate = replace(node, path, replacement)
            if size(candidate) < size(node):
                yield candidate


def _replacements(target: Node) -> List[Node]:
    """The possible replacements of a subtree: its children, then the constants 0.0 and 1.0."""
    return list(target.kids) + [Node(op="const", value=0.0), Node(op="const", value=1.0)]


def _recheck(adapter: Adapter, v: Violation, candidate: Node) -> Optional[Violation]:
    """Re-run the check of `v` on `candidate` (same rule, path and env); None if it passes or cannot be evaluated."""
    try:
        return _rerun(adapter, v, candidate)
    except Unsupported:
        return None


def _rerun(adapter: Adapter, v: Violation, candidate: Node) -> Optional[Violation]:
    """Dispatch the check of `v`'s rule: env rules take no path, rewrites apply at `v.path`."""
    if v.rule in ENV_RULES:
        return check_env_rule(adapter, ENV_RULES[v.rule], candidate, v.env)
    rule = {**RULES, **QRULES}.get(v.rule)
    if rule is None:
        raise ValueError("minimise supports rewrite and env-rule violations, not %s" % v.rule)
    if v.path not in paths(candidate):
        return None
    return check_rewrite(adapter, rule, candidate, v.path, v.env)
