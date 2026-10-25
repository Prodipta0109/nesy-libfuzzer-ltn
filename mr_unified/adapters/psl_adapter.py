"""PSL adapter for the unified metamorphic oracle (spec Task 10): the PSL core of NeuPSL, not its neural predicate layer.
Every formula is compiled to Lukasiewicz conjunctions and negations (or, implies: negated conjunctions), one PSL predicate
per node, and read off as the MAP value of an output atom. Run: venv_psl/bin/python -m pytest mr_unified/tests/test_psl_adapter.py
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from mr_unified.adapters.base import Unsupported
from mr_unified.formula import QUANT_OPS, Env, Node

ROW = "c"
OUTPUT_PRIOR = 1.0
# ADMM stop tolerance 1e-8: at 1e-6 a 120-case batch was off by up to 1.2e-4 vs ref_eval; 1e-8 passes the tests.
ADMM_OPTIONS = {"admmreasoner.epsilonabs": "1e-8", "admmreasoner.epsilonrel": "1e-8", "admmreasoner.maxiterations": "100000"}
WEAK_OPS = ("and_w", "or_w")

Literal = Tuple[str, bool]


@dataclass
class _Build:
    """The PSL pieces of one batch: observed leaves, derived nodes with their prior depth, and the rules over them."""

    leaves: Dict[str, float] = field(default_factory=dict)
    priors: Dict[str, float] = field(default_factory=dict)
    rules: List[Tuple[str, Tuple[Literal, ...]]] = field(default_factory=list)
    count: int = 0
    outputs: int = 0
    depth: int = 0

    def fresh(self, prefix: str) -> str:
        """A predicate name no other node of this batch has."""
        self.count += 1
        return "%s%d" % (prefix, self.count)


class PSLAdapter:
    """PSL core of NeuPSL (not the neural predicate layer): Lukasiewicz connectives via conjunction rules, MAP by ADMM."""

    name = "psl"
    dtype = "float64"

    def __init__(self, scale: float = 1.0) -> None:
        """`scale` multiplies every rule and prior weight; the MAP value does not change with it."""
        self.scale = scale

    def declared_quantifiers(self) -> Dict[str, str]:
        """No quantifier is declared; PSL grounds universal rules to separate weighted rules, not one aggregate."""
        return {}

    def dtype_for(self, node: Node) -> str:
        """ADMM is accurate to about 1e-6, far from float64; float32 is the closest honest precision (see report)."""
        return "float32"

    def evaluate(self, node: Node, env: Env) -> float:
        """The truth value of `node` in `env`: one PSL run for one formula."""
        return self.evaluate_many([(node, env)])[0]

    def evaluate_many(self, cases: Sequence[Tuple[Node, Env]]) -> List[float]:
        """The truth values of many (node, env) pairs in ONE PSL run, one output atom per case."""
        build = _Build()
        for index, (node, env) in enumerate(cases):
            root = _compile(build, node, env, 1)
            build.rules.append(("O%d" % index, _with_positive(build, [root])))
            build.outputs += 1
        frame = _run(build, self.scale)
        return [frame["O%d" % index] for index in range(len(cases))]


def _atom_value(node: Node, env: Env) -> float:
    """Truth of a propositional atom, or of a ground predicate atom P(o1, ...) with every argument in the domain."""
    if not node.args:
        return env.atoms[node.name]
    unbound = [arg for arg in node.args if arg not in env.domain]
    if unbound:
        raise ValueError("formula has free variables: %s" % (unbound,))
    return env.preds[node.name][tuple(node.args)]


def _compile(build: _Build, node: Node, env: Env, depth: int) -> Literal:
    """Laws: not a == 1 - a; a (x) b == max(a + b - 1, 0); a (+) b == not(not a (x) not b); a -> b == not(a (x) not b)."""
    if node.op == "atom":
        return _leaf(build, _atom_value(node, env))
    if node.op == "const":
        return _leaf(build, node.value)
    if node.op == "not":
        name, sign = _compile(build, node.kids[0], env, depth + 1)
        return name, not sign
    if node.op in QUANT_OPS or node.op in WEAK_OPS:
        raise Unsupported("PSL fragment has no %s" % node.op)
    return _compile_connective(build, node, env, depth)


def _compile_connective(build: _Build, node: Node, env: Env, depth: int) -> Literal:
    """The binary strong connectives as conjunction nodes over literals of their kids."""
    lits = [_compile(build, kid, env, depth + 1) for kid in node.kids]
    if node.op == "and_s":
        return _conj(build, [lits[0], lits[1]], depth)
    if node.op == "or_s":
        return _negated(_conj(build, [_negated(lits[0]), _negated(lits[1])], depth))
    if node.op == "imp":
        return _negated(_conj(build, [lits[0], _negated(lits[1])], depth))
    raise ValueError("unknown op: %s" % node.op)


def _negated(lit: Literal) -> Literal:
    """The literal of 1 - value."""
    return lit[0], not lit[1]


def _leaf(build: _Build, value: float) -> Literal:
    """An observed predicate whose one atom has truth `value`."""
    name = build.fresh("L")
    build.leaves[name] = float(value)
    return name, True


def _conj(build: _Build, lits: List[Literal], depth: int) -> Literal:
    """Laws: the rule `a & b -> N` makes N >= max(0, a + b - 1); the prior ~N with weight `depth` pins N to it."""
    name = build.fresh("N")
    build.priors[name] = 2.0 ** depth
    build.depth = max(build.depth, depth)
    build.rules.append((name, _with_positive(build, lits)))
    return name, True


def _with_positive(build: _Build, lits: List[Literal]) -> Tuple[Literal, ...]:
    """Laws: 1 & a == a, so a body of only negated literals gets a constant-1 literal (PSL needs one positive literal)."""
    if any(sign for _, sign in lits):
        return tuple(lits)
    return (_leaf(build, 1.0),) + tuple(lits)


def _literal_text(lit: Literal) -> str:
    """PSL text of a literal over the row variable X."""
    name, sign = lit
    return "%s(X)" % name if sign else "!%s(X)" % name


def _rule_text(head: str, body: Sequence[Literal]) -> str:
    """PSL text of `body -> head`, with the body a conjunction of literals."""
    return "%s -> %s(X)" % (" & ".join(_literal_text(lit) for lit in body), head)


def _rule_weight(build: _Build) -> float:
    """Laws: the hard weight exceeds every prior a deviation could save (sum of 2^d over ancestors is below 2^(depth+1))."""
    return 2.0 ** (build.depth + 2)


def _run(build: _Build, scale: float) -> Dict[str, float]:
    """Solve the batch with PSL MAP inference and return the value of every output atom O<i>."""
    from pslpython.model import Model
    from pslpython.partition import Partition
    from pslpython.predicate import Predicate
    from pslpython.rule import Rule

    model = Model("psl_batch")
    preds: Dict[str, Predicate] = {}
    for name, value in build.leaves.items():
        preds[name] = Predicate(name, size=1)
        preds[name].add_data_row(Partition.OBSERVATIONS, [ROW], value)
    derived = list(build.priors) + ["O%d" % i for i in range(build.outputs)]
    for name in derived:
        preds[name] = Predicate(name, size=1)
        preds[name].add_data_row(Partition.TARGETS, [ROW])
    for pred in preds.values():
        model.add_predicate(pred)
    for head, body in build.rules:
        model.add_rule(Rule(_rule_text(head, body), weighted=True, weight=_rule_weight(build) * scale, squared=False))
    for name, weight in build.priors.items():
        model.add_rule(Rule("!%s(X)" % name, weighted=True, weight=weight * scale, squared=False))
    for index in range(build.outputs):
        model.add_rule(Rule("!O%d(X)" % index, weighted=True, weight=OUTPUT_PRIOR * scale, squared=False))
    frame = model.infer(psl_options=ADMM_OPTIONS)
    return _values(frame)


def _values(frame) -> Dict[str, float]:
    """Map each predicate name to its single inferred truth value (the 'truth' column)."""
    return {pred.name(): float(table["truth"].iloc[0]) for pred, table in frame.items()}
