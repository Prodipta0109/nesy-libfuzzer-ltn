"""Byte-driven case generation (spec Task 8): fuzz bytes -> one Case (rewrite, quantifier rewrite, env rule, property or
cross check), plus the formula and env text used in findings. Only pure decoding lives here; running is in harness.py.
"""
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from mr_unified.adapters.base import Adapter
from mr_unified.formula import QUANT_OPS, Env, Node, paths
from mr_unified.rewrites import PROP_ATOMS, QRULES, RULES, Rule, apply_at, random_formula, random_quantified, valid_for

MAX_DEPTH = 4
DOMAIN = ("o0", "o1", "o2")
VAR = "x"
SUPPORTED_AGGREGATORS = ("min", "max", "fold")
B1 = "B1_permute_domain"
B7 = "B7_singleton"


class Bytes:
    """A cursor over fuzz bytes: each read consumes one byte, and reads past the end give 0."""

    def __init__(self, data: bytes) -> None:
        self.data = bytes(data)
        self.pos = 0

    def byte(self) -> int:
        """The next byte as 0..255, or 0 past the end."""
        value = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return value

    def unit(self) -> float:
        """A float in [0, 1) from the next byte."""
        return self.byte() / 256.0

    def rest(self) -> bytes:
        """The unread bytes without trailing zeros: zeros past the end read the same, so padding is invisible."""
        return self.data[self.pos:].rstrip(b"\x00")

    def int(self, lo: int, hi: int) -> int:
        """An integer in [lo, hi] from the next byte."""
        return lo + self.byte() % (hi - lo + 1)


class ByteRandom(random.Random):
    """A random.Random whose draws read the fuzz bytes, so one changed byte changes one draw."""

    def __init__(self, stream: Bytes) -> None:
        super().__init__(0)
        self.stream = stream

    def random(self) -> float:
        """A float in [0, 1) from the next byte."""
        return self.stream.unit()

    def getrandbits(self, k: int) -> int:
        """k random bits from the next ceil(k/8) bytes, big-endian, masked to k bits."""
        value = 0
        for _ in range((k + 7) // 8):
            value = (value << 8) | self.stream.byte()
        return value & ((1 << k) - 1)


@dataclass
class Case:
    """One decoded check: its kind, formula and env, the rule or property name, where it applies (None: nothing
    applies), and the unread bytes without trailing zeros (property checks draw their random numbers from them)."""

    kind: str
    node: Node
    env: Env
    name: str
    path: Optional[Tuple[int, ...]]
    tail: bytes = b""


def decode(data: bytes, adapters: Sequence[Adapter]) -> Case:
    """Decode fuzz bytes into one Case for `adapters`; the kind is drawn from the kinds those adapters allow."""
    stream = Bytes(data)
    rng = ByteRandom(stream)
    kinds = available_kinds(adapters)
    kind = kinds[stream.int(0, len(kinds) - 1)]
    case = _DECODERS[kind](stream, rng, adapters)
    case.tail = stream.rest()
    return case


def available_kinds(adapters: Sequence[Adapter]) -> List[str]:
    """Kinds that can run on `adapters`: qrewrite needs a rule valid for all of them, cross needs two or more."""
    kinds = ["rewrite", "env_rule", "property"]
    if quantifier_rule_ids(adapters):
        kinds.insert(1, "qrewrite")
    if len(adapters) > 1:
        kinds.append("cross")
    return kinds


def quantifier_rule_ids(adapters: Sequence[Adapter]) -> List[str]:
    """Ids of QRULES that hold for every adapter's declared quantifiers, in QRULES order."""
    return [
        rid for rid, rule in QRULES.items()
        if all(valid_for(rule, adapter.declared_quantifiers()) for adapter in adapters)
    ]


def same_quantifiers(adapters: Sequence[Adapter]) -> bool:
    """Law: quantified cross checks need one shared, supported declaration; empty or unsupported ones never match."""
    declared = {tuple(sorted(adapter.declared_quantifiers().items())) for adapter in adapters}
    if len(declared) != 1:
        return False
    (items,) = declared
    return bool(items) and all(value in SUPPORTED_AGGREGATORS for _, value in items)


def _domain(stream: Bytes) -> Tuple[str, ...]:
    """A domain of 1 to 3 objects, its size drawn from the next byte."""
    return DOMAIN[:stream.int(1, len(DOMAIN))]


def _random_env(rng: random.Random, domain: Tuple[str, ...]) -> Env:
    """A random env over `domain`; the formula drawn with it is discarded, depth 0 keeps that draw small."""
    return random_quantified(rng, 0, domain)[1]


def _wrap(op: str, body: Node) -> Node:
    """The formula `op VAR. body` with the bound variable VAR."""
    return Node(op=op, name=VAR, kids=(body,))


def _instances(node: Node, rules: Dict[str, Rule]) -> List[Tuple[str, Tuple[int, ...]]]:
    """Every (rule id, path) where a rule applies to `node`, in rule order and then path order."""
    return [
        (rid, path) for rid, rule in rules.items() for path in paths(node)
        if apply_at(node, rule, path) is not None
    ]


def _pick_instance(kind: str, node: Node, env: Env, rules: Dict[str, Rule], stream: Bytes) -> Case:
    """The case for the applicable (rule, path) that the next byte picks; path None when nothing applies."""
    options = _instances(node, rules)
    if not options:
        return Case(kind, node, env, "", None)
    name, path = options[stream.int(0, len(options) - 1)]
    return Case(kind, node, env, name, path)


def _decode_rewrite(stream: Bytes, rng: ByteRandom, adapters: Sequence[Adapter]) -> Case:
    """A propositional formula with a RULES instance picked by the next byte."""
    node = random_formula(rng, stream.int(1, MAX_DEPTH), PROP_ATOMS)
    return _pick_instance("rewrite", node, _random_env(rng, ()), RULES, stream)


def _decode_qrewrite(stream: Bytes, rng: ByteRandom, adapters: Sequence[Adapter]) -> Case:
    """A closed quantified formula with a QRULES instance that holds for every adapter."""
    node, env = random_quantified(rng, stream.int(1, MAX_DEPTH), _domain(stream))
    rules = {rid: QRULES[rid] for rid in quantifier_rule_ids(adapters)}
    return _pick_instance("qrewrite", node, env, rules, stream)


def _decode_env_rule(stream: Bytes, rng: ByteRandom, adapters: Sequence[Adapter]) -> Case:
    """An env rule on a quantified formula; B7 only on a one-object domain, B1 on any domain."""
    domain = _domain(stream)
    names = [B1] + ([B7] if len(domain) == 1 else [])
    name = names[stream.int(0, len(names) - 1)]
    node, env = random_quantified(rng, stream.int(1, MAX_DEPTH), domain)
    if name == B7 and node.op != "forall":
        node = _wrap("forall", node)
    return Case("env_rule", node, env, name, ())


PROPERTIES = ("monotone", "boolean", "slope", "separating", "instantiation", "split")


def _decode_property(stream: Bytes, rng: ByteRandom, adapters: Sequence[Adapter]) -> Case:
    """One property check, with the propositional or quantified formula and env that it needs."""
    name = PROPERTIES[stream.int(0, len(PROPERTIES) - 1)]
    if name == "separating":
        return Case("property", Node(op="atom", name="p0"), _random_env(rng, ()), name, ())
    if name in ("monotone", "boolean", "slope"):
        node = random_formula(rng, stream.int(1, MAX_DEPTH), PROP_ATOMS)
        return Case("property", node, _random_env(rng, ()), name, ())
    domain = _domain(stream)
    if name == "instantiation":
        node, env = random_quantified(rng, stream.int(1, MAX_DEPTH), domain)
        if node.op not in QUANT_OPS:
            node = _wrap("forall", node)
        return Case("property", node, env, name, ())
    op = QUANT_OPS[stream.int(0, len(QUANT_OPS) - 1)]
    body = random_formula(rng, stream.int(1, MAX_DEPTH), PROP_ATOMS)
    return Case("property", _wrap(op, body), _random_env(rng, domain), name, ())


def _decode_cross(stream: Bytes, rng: ByteRandom, adapters: Sequence[Adapter]) -> Case:
    """A formula and its env to compare across all adapters: quantified only if the adapters agree on quantifiers."""
    domain = _domain(stream)
    depth = stream.int(1, MAX_DEPTH)
    if same_quantifiers(adapters):
        node, env = random_quantified(rng, depth, domain)
        return Case("cross", node, env, "", ())
    return Case("cross", random_formula(rng, depth, PROP_ATOMS), _random_env(rng, ()), "", ())


_DECODERS: Dict[str, Callable[[Bytes, ByteRandom, Sequence[Adapter]], Case]] = {
    "rewrite": _decode_rewrite,
    "qrewrite": _decode_qrewrite,
    "env_rule": _decode_env_rule,
    "property": _decode_property,
    "cross": _decode_cross,
}


def format_node(n: Node) -> str:
    """Readable text for a formula, e.g. and_s(p0, not(p1)); a quantifier prints as `forall x. body`."""
    if n.op == "atom":
        return "%s(%s)" % (n.name, ", ".join(n.args)) if n.args else n.name
    if n.op == "const":
        return "%g" % n.value
    if n.op in QUANT_OPS:
        return "%s %s. %s" % (n.op, n.name, format_node(n.kids[0]))
    return "%s(%s)" % (n.op, ", ".join(format_node(kid) for kid in n.kids))


def format_env(env: Env) -> Dict[str, object]:
    """A JSON-ready copy of an env: atoms, predicate tables keyed by 'o0,o1', and the domain."""
    tables = {
        name: {",".join(key): value for key, value in table.items()} for name, table in env.preds.items()
    }
    return {"atoms": dict(env.atoms), "preds": tables, "domain": list(env.domain)}
