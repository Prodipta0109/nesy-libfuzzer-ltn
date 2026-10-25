"""Property relations and separating checks for the unified metamorphic oracle.
Implements spec Task 3: monotonicity A11, boolean snapping A12, piecewise-linear slope A13, and separation checks S1, S2.
"""
import random
from typing import Callable, Dict, List, Set

from mr_unified.formula import Env, Node

Evaluate = Callable[[Node, Env], float]

RAISE_LOW = 0.05
RAISE_HIGH = 0.5
SLOPE_STEP = 1.0 / 64.0
SLOPE_TOL = 1e-9
HALF = 0.5


def atom_names(n: Node) -> Set[str]:
    """Names of all propositional atoms occurring in `n`."""
    if n.op == "atom":
        return {n.name}
    names: Set[str] = set()
    for kid in n.kids:
        names |= atom_names(kid)
    return names


def polarities(n: Node, name: str, sign: int = 1) -> Set[int]:
    """Signs (+1/-1) of atom `name` in `n`: `not` and the antecedent of `imp` flip the sign."""
    if n.op == "atom":
        return {sign} if n.name == name else set()
    if n.op == "not":
        return polarities(n.kids[0], name, -sign)
    if n.op == "imp":
        return polarities(n.kids[0], name, -sign) | polarities(n.kids[1], name, sign)
    found: Set[int] = set()
    for kid in n.kids:
        found |= polarities(kid, name, sign)
    return found


def monotone_atoms(n: Node) -> List[str]:
    """Sorted atoms of `n` with one polarity only; atoms of mixed polarity are not checkable."""
    return [name for name in sorted(atom_names(n)) if len(polarities(n, name)) == 1]


def monotone_margin(evaluate: Evaluate, node: Node, env: Env, rng: random.Random) -> float:
    """Law A11: raising a with polarity +1 never lowers phi; with polarity -1 it never raises phi."""
    names = monotone_atoms(node)
    if not names:
        return 0.0
    name = rng.choice(names)
    (sign,) = polarities(node, name)
    raised = min(env.atoms[name] + rng.uniform(RAISE_LOW, RAISE_HIGH), 1.0)
    before = evaluate(node, env)
    after = evaluate(node, _with_atom(env, name, raised))
    return float(max(0.0, sign * (before - after)))


def boolean_margin(evaluate: Evaluate, node: Node, env: Env, rng: random.Random) -> float:
    """Law A12: on atoms in {0, 1} the formula equals its classical truth value (rng is unused)."""
    snapped = Env(
        atoms={key: _snap(value) for key, value in env.atoms.items()},
        preds=env.preds,
        domain=env.domain,
    )
    return float(abs(evaluate(node, snapped) - _classical(node, snapped.atoms)))


def slope_margin(evaluate: Evaluate, node: Node, env: Env, rng: random.Random) -> float:
    """Law A13: on one linear piece the slope in an atom is an integer; margin = distance to nearest integer."""
    names = sorted(atom_names(node))
    if not names:
        return 0.0
    name = rng.choice(names)
    start = min(env.atoms[name], 1.0 - 2.0 * SLOPE_STEP)
    values = [evaluate(node, _with_atom(env, name, start + k * SLOPE_STEP)) for k in range(3)]
    first = (values[1] - values[0]) / SLOPE_STEP
    second = (values[2] - values[1]) / SLOPE_STEP
    if abs(first - second) > SLOPE_TOL:
        return 0.0
    return float(abs(second - round(second)))


def separating_report(evaluate: Evaluate) -> List[str]:
    """Separation S1: or_w(a, not a) < 1 and S2: and_s(a, a) < 0.5 at a = 0.5; returns the failures."""
    env = Env(atoms={"a": HALF}, preds={}, domain=())
    a = Node(op="atom", name="a")
    report: List[str] = []
    excluded_middle = float(evaluate(Node(op="or_w", kids=(a, Node(op="not", kids=(a,)))), env))
    if not excluded_middle < 1.0:
        report.append("S1: or_w(a, not a) at a=0.5 is %r, expected < 1" % excluded_middle)
    contradiction = float(evaluate(Node(op="and_s", kids=(a, a)), env))
    if not contradiction < HALF:
        report.append("S2: and_s(a, a) at a=0.5 is %r, expected < 0.5" % contradiction)
    return report


def _snap(value: float) -> float:
    """Snap a truth value to 0.0 or 1.0 at the midpoint 0.5."""
    return 1.0 if value >= HALF else 0.0


def _classical(n: Node, atoms: Dict[str, float]) -> float:
    """Classical truth value of `n` over 0/1 atoms, combined with the laws in `_classical_op`."""
    if n.op == "atom":
        return atoms[n.name]
    if n.op == "const":
        return _snap(n.value)
    return _classical_op(n.op, [_classical(kid, atoms) for kid in n.kids])


def _classical_op(op: str, xs: List[float]) -> float:
    """Laws: not a = 1 - a; a and b = min; a or b = max; a -> b = (not a) or b."""
    if op == "not":
        return 1.0 - xs[0]
    if op in ("and_s", "and_w"):
        return min(xs[0], xs[1])
    if op in ("or_s", "or_w"):
        return max(xs[0], xs[1])
    if op == "imp":
        return max(1.0 - xs[0], xs[1])
    raise ValueError("unknown op: %s" % op)


def _with_atom(env: Env, name: str, value: float) -> Env:
    """Copy of `env` with atom `name` set to `value`; the other atoms are unchanged."""
    atoms = dict(env.atoms)
    atoms[name] = value
    return Env(atoms=atoms, preds=env.preds, domain=env.domain)
