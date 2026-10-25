"""PSL adapter for the unified metamorphic oracle (spec Task 10): the PSL core of NeuPSL, not its neural predicate layer.
Every formula is compiled to Lukasiewicz conjunctions and negations (or, implies: negated conjunctions); each conjunction is
one PSL predicate, and each formula is read off as the MAP value of its output atom. MAP is solved by ADMM, so evaluate and
evaluate_many agree with ref_eval within the "admm" precision of oracle.py, not exactly.
"""
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from mr_unified.adapters.base import Unsupported
from mr_unified.formula import QUANT_OPS, Env, Node

ROW = "c"
OUTPUT_PRIOR = 1.0
# Largest compile depth at which 100 random formulas agreed with ref_eval (random depth 7 reached compile depth 12); deeper: Unsupported.
MAX_VALIDATED_DEPTH = 12
ADMM_OPTIONS = {"admmreasoner.epsilonabs": "1e-8", "admmreasoner.epsilonrel": "1e-8", "admmreasoner.maxiterations": "100000"}
WEAK_OPS = ("and_w", "or_w")

Literal = Tuple[str, bool]


@dataclass(frozen=True)
class _Rule:
    """One PSL rule `body -> head` of case `case`; its weight is set by that case's depth."""

    head: str
    body: Tuple[Literal, ...]
    case: int


@dataclass
class _Build:
    """The PSL pieces of one batch: observed leaves, prior weights, rules, and the compile depth of every case."""

    leaves: Dict[str, float] = field(default_factory=dict)
    priors: Dict[str, float] = field(default_factory=dict)
    rules: List[_Rule] = field(default_factory=list)
    case_depths: List[int] = field(default_factory=list)
    count: int = 0
    case: int = 0
    depth: int = 0

    def fresh(self, prefix: str) -> str:
        """A predicate name no other node of this batch has."""
        self.count += 1
        return "%s%d" % (prefix, self.count)


class PSLAdapter:
    """PSL core of NeuPSL (not the neural predicate layer): Lukasiewicz connectives via conjunction rules, MAP by ADMM."""

    name = "psl"
    dtype = "admm"

    def __init__(self, scale: float = 1.0, rule_copies: int = 1) -> None:
        """`scale` multiplies every rule and prior weight; `rule_copies` emits each rule that many times (C-group tests only)."""
        self.scale = scale
        self.rule_copies = rule_copies

    def declared_quantifiers(self) -> Dict[str, str]:
        """Both quantifiers are "unsupported": PSL grounds universal rules to separate weighted rules, not one aggregate."""
        return {"forall": "unsupported", "exists": "unsupported"}

    def dtype_for(self, node: Node) -> str:
        """The "admm" precision: PSL's ADMM solver stops at a residual tolerance, not at machine precision."""
        return "admm"

    def evaluate(self, node: Node, env: Env) -> float:
        """The truth value of `node` in `env`: one PSL run for one formula."""
        return self.evaluate_many([(node, env)])[0]

    def evaluate_many(self, cases: Sequence[Tuple[Node, Env]]) -> List[float]:
        """The truth values of many (node, env) pairs in ONE PSL run; each equals `evaluate` within the ADMM tolerance."""
        if not cases:
            return []
        build = _Build()
        for index, (node, env) in enumerate(cases):
            _compile_case(build, index, node, env)
        frame = _run(build, self.scale, self.rule_copies)
        return [frame["O%d" % index] for index in range(len(cases))]


def _compile_case(build: _Build, index: int, node: Node, env: Env) -> None:
    """Law: the root literal is copied into the output atom O<index>, whose value is the case's truth value."""
    build.case = index
    build.depth = 0
    root = _compile(build, node, env, 1)
    build.rules.append(_Rule("O%d" % index, _with_positive(build, [root]), index))
    build.case_depths.append(build.depth)
    if build.depth > MAX_VALIDATED_DEPTH:
        raise Unsupported("formula depth %d exceeds the validated depth %d" % (build.depth, MAX_VALIDATED_DEPTH))


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
    """The binary strong connectives as conjunction nodes over the literals of their kids."""
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
    """Laws: the rule `a & b -> N` makes N >= max(0, a + b - 1). Its prior ~N has weight 2^depth, which exceeds the ancestors'
    priors 2^1 + ... + 2^(depth-1) plus the output prior 1, so raising N never pays and MAP pins N to that lower bound."""
    name = build.fresh("N")
    build.priors[name] = 2.0 ** depth
    build.depth = max(build.depth, depth)
    build.rules.append(_Rule(name, _with_positive(build, lits), build.case))
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


def _rule_weight(build: _Build, case: int) -> float:
    """Laws: the hard weight 2^(d+2) of a case of depth d exceeds any saving a deviation could make, since the case's priors
    sum to less than 2^(d+1) (see _conj), so every rule of the case is tight at the least solution."""
    return 2.0 ** (build.case_depths[case] + 2)


def _run(build: _Build, scale: float, rule_copies: int) -> Dict[str, float]:
    """Solve the batch in a fresh temp directory (the working directory while PSL runs), removed afterwards."""
    temp_dir = tempfile.mkdtemp(prefix="psl_run_")
    previous = os.getcwd()
    try:
        os.chdir(temp_dir)
        return _solve(build, scale, rule_copies)
    finally:
        os.chdir(previous)
        shutil.rmtree(temp_dir)


def _solve(build: _Build, scale: float, rule_copies: int) -> Dict[str, float]:
    """Build the PSL model for the batch, run MAP inference, and return every predicate's inferred value."""
    from pslpython.model import Model
    from pslpython.partition import Partition
    from pslpython.predicate import Predicate
    from pslpython.rule import Rule

    model = Model("psl_batch_%s" % uuid.uuid4().hex)
    preds: Dict[str, Predicate] = {}
    for name, value in build.leaves.items():
        preds[name] = Predicate(name, size=1)
        preds[name].add_data_row(Partition.OBSERVATIONS, [ROW], value)
    outputs = ["O%d" % i for i in range(len(build.case_depths))]
    for name in list(build.priors) + outputs:
        preds[name] = Predicate(name, size=1)
        preds[name].add_data_row(Partition.TARGETS, [ROW])
    for pred in preds.values():
        model.add_predicate(pred)
    for rule in build.rules:
        for _ in range(rule_copies):
            weight = _rule_weight(build, rule.case) * scale
            model.add_rule(Rule(_rule_text(rule.head, rule.body), weighted=True, weight=weight, squared=False))
    for name, weight in build.priors.items():
        model.add_rule(Rule("!%s(X)" % name, weighted=True, weight=weight * scale, squared=False))
    for name in outputs:
        model.add_rule(Rule("!%s(X)" % name, weighted=True, weight=OUTPUT_PRIOR * scale, squared=False))
    frame = model.infer(psl_options=ADMM_OPTIONS)
    return _values(frame)


def _values(frame) -> Dict[str, float]:
    """Map each predicate name to its single inferred truth value (the 'truth' column)."""
    return {pred.name(): float(table["truth"].iloc[0]) for pred, table in frame.items()}
