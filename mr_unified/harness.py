"""Byte-driven fuzz harness for the unified metamorphic oracle (spec Task 8): runs decoded cases on the selected adapters,
counts them in STATS, and reports each (lib, rule) once with a finding. Case decoding is in cases.py, guidance in guidance.py.
Run: python -m mr_unified.harness --random N [--lib ltn|lnn|ref|all]; with Atheris installed, the same without --random.
"""
import argparse
import contextlib
import dataclasses
import math
import random
import sys
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence

try:
    import atheris
except ImportError:
    atheris = None

# Under Atheris, every mr_unified module must be imported inside instrument_imports, or libFuzzer sees none of its
# edges. That includes guidance.py, where bucket() is the coverage signal. include=["mr_unified"] leaves torch out.
INSTRUMENT = atheris.instrument_imports(include=["mr_unified"]) if atheris else contextlib.nullcontext()

with INSTRUMENT:
    from mr_unified import findings_io
    from mr_unified.adapters.base import Adapter, Unsupported
    from mr_unified.adapters.guard import AdapterCrash, GuardedAdapter
    from mr_unified.adapters.ref_adapter import RefAdapter
    from mr_unified.cases import Bytes, ByteRandom, Case, decode, format_env, format_node
    from mr_unified.guidance import bucket, guidance_value
    from mr_unified.oracle import (
        ALL_REWRITES, Violation, check_env_rule, check_property, check_rewrite, cross_check, cross_gap, env_rule_gap,
        minimise, property_gap, property_margin, rewrite_gap,
    )
    from mr_unified.properties import separating_report
    from mr_unified.rewrites import ENV_RULES

KINDS = ("rewrite", "qrewrite", "env_rule", "property", "cross")
ALL_LIBS = ("ltn", "lnn", "ref")
INPUT_BYTES = 1024


def _make_ltn() -> Adapter:
    """The LTN adapter; imported only when selected, since it loads torch."""
    from mr_unified.adapters.ltn_adapter import LTNAdapter

    return LTNAdapter()


def _make_lnn() -> Adapter:
    """The LNN adapter; imported only when selected, since it needs lnn/src on the path."""
    from mr_unified.adapters.lnn_adapter import LNNAdapter

    return LNNAdapter()


def _make_psl() -> Adapter:
    """The PSL adapter; imported only when selected, since it needs venv_psl (pslpython and the JVM)."""
    from mr_unified.adapters.psl_adapter import PSLAdapter

    return PSLAdapter()


REGISTRY: Dict[str, Callable[[], Adapter]] = {"ref": RefAdapter, "ltn": _make_ltn, "lnn": _make_lnn, "psl": _make_psl}


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
    """Decode `data` for `adapters`, run the case, update STATS, and return the violations it produces.
    Counting: with several adapters a non-cross case counts once per adapter (a case with no applicable rule instance
    counts once); a cross case counts once for all adapters together.
    """
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


def _label(case: Case) -> str:
    """The rule label of a case: its rule or property name, or "cross" for the cross kind (never empty)."""
    return "cross" if case.kind == "cross" else case.name


def _run_on(case: Case, adapter: Adapter) -> List[Violation]:
    """Run one case on one adapter: Unsupported is a skip; a failed law or an adapter crash is a violation."""
    try:
        violation = _check(case, GuardedAdapter(adapter))
    except Unsupported:
        _count(case.kind, "skipped")
        return []
    except AdapterCrash as crash:
        violation = _crash(case, adapter.name, crash)
    _count(case.kind, "run")
    if violation is None:
        return []
    _count(case.kind, "violations")
    return [violation]


def _crash(case: Case, lib: str, crash: AdapterCrash) -> Violation:
    """The violation of an adapter crash: rule label `<label>:crash`, margin inf. Printed once per distinct error.
    Crashes inside oracle.rewrite_gap or oracle._compare are converted there instead; _crash_mark labels those.
    """
    label = _label(case)
    key = (lib, label, repr(crash.original))
    if key not in _CRASH_SEEN:
        _CRASH_SEEN.add(key)
        print("[CRASH] %s %s: %r" % (lib, label, crash.original), file=sys.stderr, flush=True)
    nan = float("nan")
    path = case.path if case.path is not None else ()
    return Violation(lib, label + ":crash", path, case.node, case.node, case.env, (nan, nan), float("inf"))


def _crash_mark(violation: Optional[Violation]) -> Optional[Violation]:
    """A violation with margin inf came from an adapter exception inside the oracle: label it `<rule>:crash`."""
    if violation is None or not math.isinf(violation.margin):
        return violation
    return dataclasses.replace(violation, rule=violation.rule + ":crash")


def _check(case: Case, adapter: Adapter) -> Optional[Violation]:
    """The law of a non-cross case, checked on `adapter`: None when it holds."""
    if case.kind in ("rewrite", "qrewrite"):
        return _check_rewrite_case(case, adapter)
    if case.kind == "env_rule":
        return _check_env_rule_case(case, adapter)
    return _check_property_case(case, adapter)


def _check_rewrite_case(case: Case, adapter: Adapter) -> Optional[Violation]:
    """The rewrite law at case.path; its gap is also the coverage guidance for libFuzzer."""
    rule = ALL_REWRITES[case.name]
    gap, bound = rewrite_gap(adapter, rule, case.node, case.path, case.env)
    bucket(guidance_value(gap, bound))
    if gap <= bound:
        return None
    return _crash_mark(check_rewrite(adapter, rule, case.node, case.path, case.env))


def _check_env_rule_case(case: Case, adapter: Adapter) -> Optional[Violation]:
    """The env-rule law of case.name; its gap is the coverage guidance."""
    rule = ENV_RULES[case.name]
    gap, bound = env_rule_gap(adapter, rule, case.node, case.env)
    bucket(guidance_value(gap, bound))
    if gap <= bound:
        return None
    return _crash_mark(check_env_rule(adapter, rule, case.node, case.env))


def _tail_rng(case: Case) -> ByteRandom:
    """A fresh random source over the case's unread bytes: every adapter sees the same draws."""
    return ByteRandom(Bytes(case.tail))


def _check_property_case(case: Case, adapter: Adapter) -> Optional[Violation]:
    """The property law of case.name; its margin over tol(node, node) is the coverage guidance."""
    if case.name == "separating":
        return _check_separating(case, adapter)
    prop = partial(property_margin, case.name)
    margin, bound = property_gap(adapter, prop, case.node, case.env, _tail_rng(case))
    bucket(guidance_value(margin, bound))
    if margin <= bound:
        return None
    return _crash_mark(check_property(adapter, prop, case.node, case.env, _tail_rng(case), case.name))


def _check_separating(case: Case, adapter: Adapter) -> Optional[Violation]:
    """Separation S1, S2: the margin of a violation is the number of failed separation checks."""
    failures = separating_report(adapter.evaluate)
    bucket(guidance_value(float(len(failures)), 1.0))
    if not failures:
        return None
    nan = float("nan")
    return Violation(adapter.name, "separating", (), case.node, case.node, case.env, (nan, nan), float(len(failures)))


def _run_cross(case: Case, adapters: Sequence[Adapter]) -> List[Violation]:
    """Cross-check the case on all adapters; if any adapter refuses the formula, the case is skipped."""
    guarded = [GuardedAdapter(adapter) for adapter in adapters]
    lib = " vs ".join(adapter.name for adapter in adapters)
    try:
        violation = _cross_violation(case, guarded)
    except Unsupported:
        _count("cross", "skipped")
        return []
    except AdapterCrash as crash:
        violation = _crash(case, lib, crash)
    _count("cross", "run")
    if violation is None:
        return []
    _count("cross", "violations")
    return [violation]


def _cross_violation(case: Case, guarded: Sequence[Adapter]) -> Optional[Violation]:
    """The cross law on all adapters, labelled "cross"; its gap is the coverage guidance."""
    gap, bound = cross_gap(guarded, case.node, case.env)
    bucket(guidance_value(gap, bound))
    if gap <= bound:
        return None
    found = cross_check(guarded, case.node, case.env)
    return None if found is None else dataclasses.replace(found, rule=_label(case))


def describe(v: Violation) -> str:
    """One line: library, rule, path, formula, margin and the two values."""
    return "%s %s path=%s formula=%s margin=%.3g got=%r" % (
        v.lib, v.rule, list(v.path), format_node(v.before), v.margin, v.got,
    )


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
    """The minimised violation for rewrite and env-rule findings; crashes and other violations are returned unchanged."""
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
    parser.add_argument("--lib", default="ltn", help="ltn, lnn, ref, psl or all (all = ltn, lnn, ref; psl must be named)")
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
