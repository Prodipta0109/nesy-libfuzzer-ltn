"""Mutation study for the unified metamorphic oracle (spec Task 9): plant one bug at a time in an LTN or LNN
operator (in memory only, restored on exit) and report which oracle notices it: per_operator, full_reference or mr.
Run: python -m mr_unified.mutation_study --lib ltn|lnn|all [--cases N]
"""
import argparse
import contextlib
import random
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from mr_unified.adapters.base import Adapter, Unsupported
from mr_unified.cases import quantifier_rule_ids
from mr_unified.formula import QUANT_OPS, Env, Node, paths, ref_eval
from mr_unified.harness import make_adapters
from mr_unified.oracle import check_env_rule, check_property, check_rewrite, tol
from mr_unified.properties import (
    boolean_margin, instantiation_margin, monotone_margin, separating_report, slope_margin, split_margin,
)
from mr_unified.rewrites import (
    CONNECTIVE_OPS, ENV_RULES, PROP_ATOMS, QRULES, RULES, Rule, random_formula, random_quantified,
)

MAX_DEPTH = 4
DOMAIN = ("o0", "o1", "o2")
ORACLES = ("per_operator", "full_reference", "mr")
NOTE = "full_reference needs a reference evaluator; MR does not"
PROPOSITIONAL_PROPERTIES = ("monotone", "boolean", "slope")
QUANTIFIED_PROPERTIES = ("instantiation", "split")
KIND_LABELS = {
    (): "missed",
    ("per_operator",): "per-operator only",
    ("full_reference",): "full-reference only",
    ("mr",): "MR only",
    ("per_operator", "full_reference"): "per-operator + full-reference only",
    ("full_reference", "mr"): "MR + full-reference only",
    ("per_operator", "mr"): "MR + per-operator only",
    ("per_operator", "full_reference", "mr"): "all three",
}


@dataclass(frozen=True)
class Mutant:
    """A planted bug: the library it belongs to, what it does, and a context manager that installs it."""

    lib: str
    description: str
    patch: Callable[[], contextlib.AbstractContextManager]


class AdapterCrash(Exception):
    """A library operator raised an exception other than Unsupported: a wrong-behaviour signal, not a harness error."""


class GuardedAdapter:
    """Forwards to an adapter; any exception from evaluate other than Unsupported becomes AdapterCrash (a firing)."""

    def __init__(self, adapter: Adapter) -> None:
        self._adapter = adapter
        self.name = adapter.name
        self.dtype = adapter.dtype

    def evaluate(self, node: Node, env: Env) -> float:
        """The adapter's value; a library exception is re-raised as AdapterCrash, Unsupported passes through."""
        try:
            return self._adapter.evaluate(node, env)
        except Unsupported:
            raise
        except Exception as err:
            raise AdapterCrash(err) from err

    def declared_quantifiers(self) -> Dict[str, str]:
        """The wrapped adapter's declared quantifier semantics."""
        return self._adapter.declared_quantifiers()

    def dtype_for(self, node: Node) -> str:
        """The wrapped adapter's dtype for `node`."""
        return self._adapter.dtype_for(node)


@dataclass
class Tally:
    """Counts for one oracle on one mutant: checks run, checks skipped (Unsupported), and whether any check fired."""

    runs: int = 0
    skips: int = 0
    fired: bool = False

    def attempt(self, check: Callable[[], object]) -> None:
        """Run one check: Unsupported is a skip; a crash, or a truthy result (a Violation or True), is a firing."""
        try:
            outcome = check()
        except Unsupported:
            self.skips += 1
            return
        except AdapterCrash:
            outcome = True
        self.runs += 1
        self.fired = self.fired or bool(outcome)


# ----- patch plumbing ---------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _swapped(owner: object, name: str, replacement: Callable) -> Iterator[None]:
    """Replace owner.name for the block, and put the saved attribute back on exit, even after an error."""
    saved = vars(owner)[name]
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, saved)


def _fuzzy() -> object:
    """The ltn.fuzzy_ops module; imported on first use so that importing this module does not load ltn."""
    import ltn.fuzzy_ops

    return ltn.fuzzy_ops


def _lnn_method() -> type:
    """LNN's Lukasiewicz method class, whose upward methods the LNN mutants replace."""
    from lnn.neural.methods.lukasiewicztransparent import LukasiewiczTransparent

    return LukasiewiczTransparent


def _lnn_utils() -> object:
    """lnn._utils, which holds negate_bounds used by the LNN Not node."""
    import lnn._utils

    return lnn._utils


# ----- LTN mutants ------------------------------------------------------------------------------------------------

def _implies_without_clamp() -> contextlib.AbstractContextManager:
    """Law broken: a -> b == min(1 - a + b, 1), here without the min."""
    return _swapped(_fuzzy().ImpliesLuk, "__call__", lambda self, x, y: 1.0 - x + y)


def _andluk_is_min() -> contextlib.AbstractContextManager:
    """Law broken: a (x) b == max(a + b - 1, 0), here min(a, b)."""
    return _swapped(_fuzzy().AndLuk, "__call__", lambda self, x, y: torch.minimum(x, y))


def _orluk_without_clamp() -> contextlib.AbstractContextManager:
    """Law broken: a (+) b == min(a + b, 1), here without the min."""
    return _swapped(_fuzzy().OrLuk, "__call__", lambda self, x, y: x + y)


def _notstandard_clipped() -> contextlib.AbstractContextManager:
    """Law broken: not a == 1 - a, here clipped at 0.5 from above."""
    return _swapped(_fuzzy().NotStandard, "__call__", lambda self, x: torch.clamp(1.0 - x, max=0.5))


def _andluk_off_by_epsilon() -> contextlib.AbstractContextManager:
    """Law broken: a (x) b == max(a + b - 1, 0), here off by 1e-3 (a tiny bug, far above rounding)."""
    fz = _fuzzy()
    original = vars(fz.AndLuk)["__call__"]
    return _swapped(fz.AndLuk, "__call__", lambda self, x, y: original(self, x, y) + 1e-3)


def _aggregmin_ignores_last() -> contextlib.AbstractContextManager:
    """Law broken: forall over one axis is the min of all elements except the last (axis size >= 2)."""
    fz = _fuzzy()
    original = vars(fz.AggregMin)["__call__"]

    def call(self, xs, dim=None, keepdim=False, mask=None):
        """Forall over one axis takes the min without its last element; other calls are unchanged."""
        if dim is None or mask is not None or len(dim) != 1 or xs.shape[dim[0]] < 2:
            return original(self, xs, dim=dim, keepdim=keepdim, mask=mask)
        axis = dim[0]
        return torch.amin(xs.narrow(axis, 0, xs.shape[axis] - 1), dim=axis, keepdim=keepdim)

    return _swapped(fz.AggregMin, "__call__", call)


def _aggregmin_wrong_axis() -> contextlib.AbstractContextManager:
    """Law broken: a quantifier over one of two free variables (a 2-d grounding) reduces the other axis."""
    fz = _fuzzy()
    original = vars(fz.AggregMin)["__call__"]

    def call(self, xs, dim=None, keepdim=False, mask=None):
        """With two free variables and one axis to reduce, the min runs over the other axis."""
        if dim is not None and xs.dim() == 2 and len(dim) == 1:
            dim = (1 - dim[0],)
        return original(self, xs, dim=dim, keepdim=keepdim, mask=mask)

    return _swapped(fz.AggregMin, "__call__", call)


def _andluk_stale_cache() -> contextlib.AbstractContextManager:
    """Law broken: an operator object called again with an input of the same shape returns its previous result."""
    fz = _fuzzy()
    original = vars(fz.AndLuk)["__call__"]
    cache: Dict[int, Tuple[Tuple[int, ...], torch.Tensor]] = {}

    def call(self, x, y):
        """Cache per operator id: a hit when the same shape was seen for this id, so the old value comes back."""
        key = id(self)
        if key in cache and cache[key][0] == tuple(x.shape):
            return cache[key][1]
        out = original(self, x, y)
        cache[key] = (tuple(x.shape), out)
        return out

    return _swapped(fz.AndLuk, "__call__", call)


# ----- LNN mutants ------------------------------------------------------------------------------------------------

def _lnn_and_is_min() -> contextlib.AbstractContextManager:
    """Law broken: the strong And upward bound is the min over the operands, not max(sum - (n - 1), 0)."""
    return _swapped(_lnn_method(), "_and_upward", lambda self, bounds: bounds.amin(dim=-1))


def _lnn_implies_swapped() -> contextlib.AbstractContextManager:
    """Law broken: a -> b with the two operands swapped before the upward step."""
    method = _lnn_method()
    original = vars(method)["_implies_upward"]
    return _swapped(method, "_implies_upward", lambda self, bounds: original(self, bounds.flip(-1)))


def _lnn_not_is_identity() -> contextlib.AbstractContextManager:
    """Law broken: not a == 1 - a, here the operand bounds are returned unchanged."""
    return _swapped(_lnn_utils(), "negate_bounds", lambda bounds, dim=-1: bounds)


def _lnn_or_raised() -> contextlib.AbstractContextManager:
    """Law broken: the strong Or upward bound is raised by 0.1 (as in lnn/mutation_check.py)."""
    method = _lnn_method()
    original = vars(method)["_or_upward"]
    return _swapped(method, "_or_upward", lambda self, bounds: original(self, bounds) + 0.1)


MUTANTS: Dict[str, Mutant] = {
    "ImpliesLuk without min(.,1)": Mutant("ltn", "a -> b = 1 - a + b with no clamp", _implies_without_clamp),
    "AndLuk uses min": Mutant("ltn", "a (x) b = min(a, b)", _andluk_is_min),
    "OrLuk without clamp": Mutant("ltn", "a (+) b = a + b with no clamp", _orluk_without_clamp),
    "NotStandard clipped at 0.5": Mutant("ltn", "not a = min(1 - a, 0.5)", _notstandard_clipped),
    "AggregMin ignores last element": Mutant("ltn", "forall skips the last element", _aggregmin_ignores_last),
    "AndLuk off by 1e-3": Mutant("ltn", "a (x) b + 1e-3", _andluk_off_by_epsilon),
    "AggregMin wrong axis with two free variables": Mutant(
        "ltn", "quantifier reduces the other axis when two variables are free", _aggregmin_wrong_axis
    ),
    "AndLuk stale result per operator id": Mutant(
        "ltn", "returns the previous result for the same operator id and shape", _andluk_stale_cache
    ),
    "LNN And upward computed as min": Mutant("lnn", "strong And upward is the min of operands", _lnn_and_is_min),
    "LNN Implies arguments swapped": Mutant("lnn", "Implies upward with swapped operands", _lnn_implies_swapped),
    "LNN Not is identity": Mutant("lnn", "Not returns its operand bounds", _lnn_not_is_identity),
    "LNN Or upward raised by 0.1": Mutant("lnn", "strong Or upward bound plus 0.1", _lnn_or_raised),
}


# ----- oracles ----------------------------------------------------------------------------------------------------

def _truth(rng: random.Random) -> float:
    """A truth value: 0.0, 1.0, or a uniform draw (the same distribution as the rewrites' random env)."""
    return rng.choice([0.0, 1.0, rng.random()])


def _domain(rng: random.Random) -> Tuple[str, ...]:
    """A quantifier domain of one to three objects."""
    return DOMAIN[:rng.randint(1, len(DOMAIN))]


def _reference(adapter: Adapter, node: Node, env: Env) -> float:
    """ref_eval with the quantifier semantics the adapter declares: min/max for LTN, the strong fold for LNN."""
    quant = "minmax" if adapter.declared_quantifiers()["forall"] == "min" else "fold"
    return ref_eval(node, env, quant)


def _disagrees(adapter: Adapter, node: Node, env: Env) -> bool:
    """Law: the adapter's value equals the reference within tol(node, node, dtype); a crash propagates as firing."""
    got = adapter.evaluate(node, env)
    return abs(got - _reference(adapter, node, env)) > tol(node, node, adapter.dtype_for(node))


def _per_operator_thunks(adapter: Adapter, rng: random.Random) -> List[Callable[[], object]]:
    """One depth-1 formula: a single connective over propositional atoms, compared with the reference."""
    op = rng.choice(CONNECTIVE_OPS)
    arity = 1 if op == "not" else 2
    node = Node(op=op, kids=tuple(Node(op="atom", name=rng.choice(PROP_ATOMS)) for _ in range(arity)))
    env = Env(atoms={name: _truth(rng) for name in PROP_ATOMS}, preds={}, domain=())
    return [partial(_disagrees, adapter, node, env)]


def _full_reference_thunks(adapter: Adapter, rng: random.Random) -> List[Callable[[], object]]:
    """A random closed formula of depth <= MAX_DEPTH, with quantifiers, compared with the reference."""
    node, env = random_quantified(rng, rng.randint(1, MAX_DEPTH), _domain(rng))
    return [partial(_disagrees, adapter, node, env)]


def _rewrites_for(adapter: Adapter) -> Dict[str, Rule]:
    """Every propositional rule, plus the quantifier rules valid for the adapter's declared quantifiers."""
    quantified = {rid: QRULES[rid] for rid in quantifier_rule_ids([adapter])}
    return {**RULES, **quantified}


def _quantified_root(node: Node) -> Node:
    """`node` if it is a quantifier, else forall x over it (instantiation needs a quantifier at the root)."""
    if node.op in QUANT_OPS:
        return node
    return Node(op="forall", name="x", kids=(node,))


def _property_margin(name: str, adapter: Adapter, node: Node, env: Env, rng: random.Random) -> float:
    """How far `adapter` is from the law of property `name` (A11, A12, A13, B6, B8); 0 when the law holds."""
    if name == "monotone":
        return monotone_margin(adapter.evaluate, node, env, rng)
    if name == "boolean":
        return boolean_margin(adapter.evaluate, node, env, rng)
    if name == "slope":
        return slope_margin(adapter.evaluate, node, env, rng)
    quant = "minmax" if adapter.declared_quantifiers()["forall"] == "min" else "fold"
    if name == "instantiation":
        return instantiation_margin(_quant_evaluate(adapter), node, env, quant, rng)
    return split_margin(_quant_evaluate(adapter), node, env, quant, rng)


def _quant_evaluate(adapter: Adapter) -> Callable[[Node, Env, str], float]:
    """The (node, env, quant) shape of B6 and B8; `quant` is ignored because the adapter fixes its semantics."""

    def evaluate(node: Node, env: Env, quant: str) -> float:
        """The adapter's own value of `node` in `env`."""
        return adapter.evaluate(node, env)

    return evaluate


def _property_thunks(adapter: Adapter, rng: random.Random, node: Node, env: Env) -> List[Callable[[], object]]:
    """Each property check, on the formula it needs: propositional for A11-A13, quantified for B6 and B8."""
    thunks = []
    for name in PROPOSITIONAL_PROPERTIES:
        body = random_formula(rng, rng.randint(1, MAX_DEPTH), PROP_ATOMS)
        thunks.append(partial(check_property, adapter, partial(_property_margin, name), body, env, rng, name))
    thunks.append(partial(
        check_property, adapter, partial(_property_margin, "instantiation"), _quantified_root(node), env, rng,
        "instantiation",
    ))
    split_node = Node(op=rng.choice(QUANT_OPS), name="x", kids=(random_formula(rng, MAX_DEPTH, PROP_ATOMS),))
    thunks.append(partial(
        check_property, adapter, partial(_property_margin, "split"), split_node, env, rng, "split",
    ))
    return thunks


def _mr_thunks(adapter: Adapter, rng: random.Random) -> List[Callable[[], object]]:
    """Every rewrite rule at a random path, every env rule, every property, and the separating checks."""
    node, env = random_quantified(rng, rng.randint(1, MAX_DEPTH), _domain(rng))
    path = rng.choice(paths(node))
    thunks = [partial(check_rewrite, adapter, rule, node, path, env) for rule in _rewrites_for(adapter).values()]
    thunks += [partial(check_env_rule, adapter, rule, node, env) for rule in ENV_RULES.values()]
    thunks += _property_thunks(adapter, rng, node, env)
    thunks.append(partial(_separating_fired, adapter))
    return thunks


def _separating_fired(adapter: Adapter) -> bool:
    """Separation S1 and S2: whether the adapter fails either at a = 0.5."""
    return bool(separating_report(adapter.evaluate))


ORACLE_THUNKS: Dict[str, Callable[[Adapter, random.Random], List[Callable[[], object]]]] = {
    "per_operator": _per_operator_thunks,
    "full_reference": _full_reference_thunks,
    "mr": _mr_thunks,
}


def _sweep(oracle: str, adapter: Adapter, n_cases: int, seed: int) -> Tally:
    """Run `oracle` on n_cases seeded cases; stop at the first firing, since the study asks only whether it fires."""
    rng = random.Random(seed)
    tally = Tally()
    for _ in range(n_cases):
        for check in ORACLE_THUNKS[oracle](adapter, rng):
            tally.attempt(check)
        if tally.fired:
            break
    return tally


def _measure_one(lib: str, n_cases: int, seed: int, patch: Optional[Callable[[], object]]) -> Dict[str, Tally]:
    """Run all three oracles on `lib`, with `patch` active (or none for the baseline)."""
    adapter = GuardedAdapter(make_adapters(lib)[0])
    context = patch() if patch is not None else contextlib.nullcontext()
    with context:
        return {oracle: _sweep(oracle, adapter, n_cases, seed) for oracle in ORACLES}


def baseline(lib: str, n_cases: int = 2000, seed: int = 0) -> Dict[str, Tally]:
    """The three oracles on the unmutated library: every one should be silent."""
    return _measure_one(lib, n_cases, seed, None)


def measure(lib: str, n_cases: int = 2000, seed: int = 0) -> Dict[str, Dict[str, Tally]]:
    """For each mutant of `lib`, the tallies of the three oracles with that mutant active."""
    return {
        name: _measure_one(lib, n_cases, seed, mutant.patch)
        for name, mutant in MUTANTS.items() if mutant.lib == lib
    }


def run(lib: str, n_cases: int = 2000, seed: int = 0) -> Dict[str, Dict[str, bool]]:
    """For each mutant of `lib`: whether each oracle fired at least once on the seeded cases."""
    return {
        name: {oracle: tally.fired for oracle, tally in tallies.items()}
        for name, tallies in measure(lib, n_cases, seed).items()
    }


def kind(firing: Dict[str, bool]) -> str:
    """The label of the set of oracles that fired, e.g. "MR only" or "missed" when none did."""
    fired = tuple(oracle for oracle in ORACLES if firing[oracle])
    return KIND_LABELS[fired]


def format_table(results: Dict[str, Dict[str, bool]]) -> str:
    """The table of one row per mutant: whether each oracle fired, and the kind; the note goes under it."""
    width = max([len("mutant")] + [len(name) for name in results])
    header = "%-*s  %-12s  %-14s  %-4s  %s" % (width, "mutant", "per_operator", "full_reference", "mr", "kind")
    lines = [header]
    for name, firing in results.items():
        cells = ["yes" if firing[oracle] else "no" for oracle in ORACLES]
        lines.append("%-*s  %-12s  %-14s  %-4s  %s" % (width, name, cells[0], cells[1], cells[2], kind(firing)))
    lines.append(NOTE)
    return "\n".join(lines)


def _parse(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Command-line options: the library, the number of cases per oracle, and the seed."""
    parser = argparse.ArgumentParser(description="Plant bugs in LTN/LNN and report which oracle notices them.")
    parser.add_argument("--lib", default="all", choices=["ltn", "lnn", "all"])
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _summary(tallies: Dict[str, Tally]) -> str:
    """One line of run and skip counts for an oracle set, e.g. per_operator runs=3 skips=1."""
    return " ".join("%s runs=%d skips=%d" % (oracle, t.runs, t.skips) for oracle, t in tallies.items())


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the table of mutants and the note (skip counts and baseline go to stderr). 1 if the baseline fires."""
    args = _parse(argv)
    libs = ["ltn", "lnn"] if args.lib == "all" else [args.lib]
    baseline_silent = True
    results: Dict[str, Dict[str, bool]] = {}
    for lib in libs:
        base = baseline(lib, args.cases, args.seed)
        baseline_silent &= not any(t.fired for t in base.values())
        print("baseline %s (unmutated): %s" % (lib, _summary(base)), file=sys.stderr)
        for name, tallies in measure(lib, args.cases, args.seed).items():
            results[name] = {oracle: tally.fired for oracle, tally in tallies.items()}
            print("%s: %s" % (name, _summary(tallies)), file=sys.stderr)
    print(format_table(results))
    return 0 if baseline_silent else 1


if __name__ == "__main__":
    sys.exit(main())
