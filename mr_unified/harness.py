"""Byte-driven fuzz harness for the unified metamorphic oracle (spec Task 8).
Decodes fuzz bytes into one Case (rewrite, quantifier rewrite, env rule, property or cross check), runs it through
oracle.py on the selected adapters, counts every run in STATS, and writes the first violation of each (lib, rule).
Run: python -m mr_unified.harness --random N [--lib ltn|lnn|ref|all]; with Atheris installed, the same without --random.
"""
import argparse
import contextlib
import math
import random
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import atheris
except ImportError:
    atheris = None

# Instrument only mr_unified, so libFuzzer sees the oracle's edges and not torch's.
INSTRUMENT = atheris.instrument_imports(include=["mr_unified"]) if atheris else contextlib.nullcontext()

with INSTRUMENT:
    from mr_unified import findings_io
    from mr_unified.adapters.base import Adapter, Unsupported
    from mr_unified.adapters.ref_adapter import RefAdapter
    from mr_unified.formula import QUANT_OPS, Env, Node, paths
    from mr_unified.oracle import (
        ALL_REWRITES, Violation, as_quant_evaluate, check_env_rule, check_property, check_rewrite,
        cross_check, minimise, rewrite_gap,
    )
    from mr_unified.properties import (
        boolean_margin, instantiation_margin, monotone_margin, separating_report, slope_margin, split_margin,
    )
    from mr_unified.rewrites import (
        ENV_RULES, PROP_ATOMS, QRULES, RULES, Rule, apply_at, random_formula, random_quantified, valid_for,
    )

KINDS = ("rewrite", "qrewrite", "env_rule", "property", "cross")
PROPERTIES = ("monotone", "boolean", "slope", "separating", "instantiation", "split")
B1 = "B1_permute_domain"
B7 = "B7_singleton"
ALL_LIBS = ("ltn", "lnn", "ref")
DOMAIN = ("o0", "o1", "o2")
MAX_DEPTH = 4
INPUT_BYTES = 1024
GUIDANCE_SCALE = 1e-3
VAR = "x"
REGISTRY: Dict[str, Callable[[], Adapter]] = {"ref": RefAdapter}


def _make_ltn() -> Adapter:
    """The LTN adapter; imported only when selected, since it loads torch."""
    from mr_unified.adapters.ltn_adapter import LTNAdapter

    return LTNAdapter()


def _make_lnn() -> Adapter:
    """The LNN adapter; imported only when selected, since it needs lnn/src on the path."""
    from mr_unified.adapters.lnn_adapter import LNNAdapter

    return LNNAdapter()


REGISTRY["ltn"] = _make_ltn
REGISTRY["lnn"] = _make_lnn


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
    """One decoded check: its kind, formula and env, the rule or property name, where it applies, and the bytes
    left over (used by property checks that draw random numbers)."""

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
    """A closed quantified formula and its env, to compare across all adapters."""
    node, env = random_quantified(rng, stream.int(1, MAX_DEPTH), _domain(stream))
    return Case("cross", node, env, "", ())


_DECODERS: Dict[str, Callable[[Bytes, ByteRandom, Sequence[Adapter]], Case]] = {
    "rewrite": _decode_rewrite,
    "qrewrite": _decode_qrewrite,
    "env_rule": _decode_env_rule,
    "property": _decode_property,
    "cross": _decode_cross,
}


def bucket(margin: float) -> None:
    """Coverage hook for libFuzzer: each threshold passed is a new edge, so larger gaps are kept in the corpus."""
    # Copied from mr_fuzz/mrs.py bucket(): the same seven thresholds, one new edge per decade of margin.
    if margin > 1e-15: pass
    if margin > 1e-12: pass
    if margin > 1e-9: pass
    if margin > 1e-6: pass
    if margin > 1e-3: pass
    if margin > 1e-1: pass
    if margin > 5e-1: pass


def guidance_value(gap: float, bound: float) -> float:
    """Gap over tolerance, scaled by GUIDANCE_SCALE and capped at 1; a NaN gap counts as a failure (1)."""
    if math.isnan(gap):
        return 1.0
    return min(1.0, gap / bound * GUIDANCE_SCALE)


def _blank_stats() -> Dict[str, Dict[str, int]]:
    """Zeroed counters for every kind: run, skipped and violations."""
    return {kind: {"run": 0, "skipped": 0, "violations": 0} for kind in KINDS}


STATS: Dict[str, Dict[str, int]] = _blank_stats()
ACTIVE: List[Adapter] = []
SELECTION = ""
_SEEN = set()
_CRASH_SEEN = set()


def reset() -> None:
    """Clear STATS and the sets of (lib, rule) pairs and crashes already reported."""
    STATS.clear()
    STATS.update(_blank_stats())
    _SEEN.clear()
    _CRASH_SEEN.clear()


def make_adapters(lib: str) -> List[Adapter]:
    """Fresh adapters for `lib`: one registry name, or every name in ALL_LIBS for "all"."""
    names = ALL_LIBS if lib == "all" else (lib,)
    for name in names:
        if name not in REGISTRY:
            raise ValueError("unknown lib: %s" % lib)
    return [REGISTRY[name]() for name in names]


def configure(lib: str) -> None:
    """Make `lib` the active selection used by TestOneInput and by random mode."""
    global ACTIVE, SELECTION
    ACTIVE = make_adapters(lib)
    SELECTION = lib


def run_input(data: bytes, adapters: Sequence[Adapter]) -> List[Violation]:
    """Decode `data` for `adapters`, run the case, update STATS, and return the violations it produces."""
    case = decode(data, adapters)
    if case.kind == "cross":
        return _run_cross(case, adapters)
    if case.path is None:
        _count(case.kind, "skipped")
        return []
    found: List[Violation] = []
    for adapter in adapters:
        found.extend(_run_on(case, adapter))
    return found


def _count(kind: str, field_name: str) -> None:
    """Add one to STATS[kind][field_name]."""
    STATS[kind][field_name] += 1


def _run_on(case: Case, adapter: Adapter) -> List[Violation]:
    """Run one case on one adapter: Unsupported is a skip; a failed law or a crash comes back as a violation."""
    try:
        violation = _check(case, adapter)
    except Unsupported:
        _count(case.kind, "skipped")
        return []
    except Exception as err:
        violation = _crash(case, adapter.name, err)
    _count(case.kind, "run")
    if violation is None:
        return []
    _count(case.kind, "violations")
    return [violation]


def _crash(case: Case, lib: str, err: Exception) -> Violation:
    """An adapter exception other than Unsupported that the oracle does not catch: a violation with margin inf."""
    key = (lib, case.name, repr(err))
    if key not in _CRASH_SEEN:
        _CRASH_SEEN.add(key)
        print("[CRASH] %s %s: %r" % (lib, case.name, err), file=sys.stderr, flush=True)
    nan = float("nan")
    path = case.path if case.path is not None else ()
    return Violation(lib, case.name, path, case.node, case.node, case.env, (nan, nan), float("inf"))


def _check(case: Case, adapter: Adapter) -> Optional[Violation]:
    """The law of `case`, checked on `adapter`: None when it holds."""
    if case.kind in ("rewrite", "qrewrite"):
        return _check_rewrite_case(case, adapter)
    if case.kind == "env_rule":
        return check_env_rule(adapter, ENV_RULES[case.name], case.node, case.env)
    if case.name == "separating":
        return _check_separating(case, adapter)
    rng = ByteRandom(Bytes(case.tail))
    return check_property(adapter, partial(_margin, case.name), case.node, case.env, rng, case.name)


def _check_rewrite_case(case: Case, adapter: Adapter) -> Optional[Violation]:
    """The rewrite law at case.path; its gap is also the coverage guidance for libFuzzer."""
    rule = ALL_REWRITES[case.name]
    gap, bound = rewrite_gap(adapter, rule, case.node, case.path, case.env)
    bucket(guidance_value(gap, bound))
    if gap <= bound:
        return None
    return check_rewrite(adapter, rule, case.node, case.path, case.env)


def _margin(name: str, adapter: Adapter, node: Node, env: Env, rng: random.Random) -> float:
    """How far `adapter` is from the property `name` (laws A11, A12, A13, B6, B8); 0 means the law holds."""
    evaluate = adapter.evaluate
    if name == "monotone":
        return monotone_margin(evaluate, node, env, rng)
    if name == "boolean":
        return boolean_margin(evaluate, node, env, rng)
    if name == "slope":
        return slope_margin(evaluate, node, env, rng)
    quant = as_quant_evaluate(adapter)
    if name == "instantiation":
        return instantiation_margin(quant, node, env, "minmax", rng)
    if name == "split":
        return split_margin(quant, node, env, "minmax", rng)
    raise ValueError("unknown property: %s" % name)


def _check_separating(case: Case, adapter: Adapter) -> Optional[Violation]:
    """Separation S1, S2: the margin of a violation is the number of failed separation checks."""
    failures = separating_report(adapter.evaluate)
    if not failures:
        return None
    nan = float("nan")
    return Violation(adapter.name, "separating", (), case.node, case.node, case.env, (nan, nan), float(len(failures)))


def _run_cross(case: Case, adapters: Sequence[Adapter]) -> List[Violation]:
    """Cross-check the case on all adapters; if any adapter refuses the formula, the case is skipped."""
    lib = " vs ".join(adapter.name for adapter in adapters)
    try:
        for adapter in adapters:
            adapter.evaluate(case.node, case.env)
        violation = cross_check(adapters, case.node, case.env)
    except Unsupported:
        _count("cross", "skipped")
        return []
    except Exception as err:
        violation = _crash(case, lib, err)
    _count("cross", "run")
    if violation is None:
        return []
    _count("cross", "violations")
    return [violation]


def describe(v: Violation) -> str:
    """One line: library, rule, path, formula, margin and the two values."""
    return "%s %s path=%s formula=%s margin=%.3g got=%r" % (
        v.lib, v.rule, list(v.path), format_node(v.before), v.margin, v.got,
    )


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


def _meta(v: Violation, small: Violation, selection: str) -> Dict[str, object]:
    """The meta.json of a finding: the violation, its minimised formula, and how to replay it."""
    return {
        "lib": v.lib,
        "rule": v.rule,
        "selection": selection,
        "replay": "python -m mr_unified.replay input.bin --lib %s" % selection,
        "path": list(v.path),
        "formula": format_node(v.before),
        "after": format_node(v.after),
        "env": format_env(v.env),
        "got": list(v.got),
        "margin": v.margin,
        "minimised": format_node(small.before),
        "minimised_margin": small.margin,
    }


def _minimised(v: Violation, adapter: Optional[Adapter]) -> Violation:
    """The minimised violation for rewrite and env-rule findings; other violations are returned unchanged."""
    if adapter is None or (v.rule not in ALL_REWRITES and v.rule not in ENV_RULES):
        return v
    return minimise(v, adapter)


def report(data: bytes, adapters: Sequence[Adapter], violations: Sequence[Violation], selection: str) -> None:
    """Print each (lib, rule) seen for the first time and write its finding once."""
    by_name = {adapter.name: adapter for adapter in adapters}
    for v in violations:
        if (v.lib, v.rule) in _SEEN:
            continue
        _SEEN.add((v.lib, v.rule))
        print("[VIOLATION] " + describe(v), file=sys.stderr, flush=True)
        small = _minimised(v, by_name.get(v.lib))
        findings_io.write_finding(v.lib, v.rule, data, _meta(v, small, selection))


def TestOneInput(data: bytes) -> None:
    """Atheris entry point: run one input on the active selection and report what it finds."""
    data = bytes(data)
    report(data, ACTIVE, run_input(data, ACTIVE), SELECTION)


def random_input(rng: random.Random) -> bytes:
    """INPUT_BYTES random bytes: the input of one random-mode case."""
    return rng.randbytes(INPUT_BYTES)


def _random_mode(count: int, seed: int) -> int:
    """Run `count` random inputs on the active selection, print one line per kind, return 1 if any violation."""
    rng = random.Random(seed)
    for _ in range(count):
        data = random_input(rng)
        report(data, ACTIVE, run_input(data, ACTIVE), SELECTION)
    for kind in KINDS:
        entry = STATS[kind]
        print("%-11s run=%d skipped=%d violations=%d" % (kind, entry["run"], entry["skipped"], entry["violations"]))
    return 1 if any(STATS[kind]["violations"] for kind in KINDS) else 0


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    """Command-line options; unknown options are left alone so that libFuzzer flags pass through."""
    parser = argparse.ArgumentParser(description="Byte-driven harness for the unified metamorphic oracle.")
    parser.add_argument("--lib", default="ltn", help="ltn, lnn, ref or all")
    parser.add_argument("--random", type=int, default=None, help="run N random inputs instead of Atheris")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random inputs")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Random mode with --random N; otherwise Atheris, when it is importable. Returns the exit code."""
    args = _parse(argv)
    try:
        configure(args.lib)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    if args.random is not None:
        return _random_mode(args.random, args.seed)
    if atheris is None:
        print("atheris is not installed; use --random N", file=sys.stderr)
        return 2
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
    return 0


if __name__ == "__main__":
    sys.exit(main())
