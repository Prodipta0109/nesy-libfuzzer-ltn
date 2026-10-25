"""Tests for the byte-driven harness (spec Task 8): decoding, byte-backed randomness, STATS, findings and replay.
Every loop counts the cases it compares and asserts that the count is positive.
"""
import copy
import json
import math
import os
import random
import subprocess
import sys
from typing import List

import pytest

from mr_unified import findings_io, harness, replay
from mr_unified.adapters.base import Unsupported
from mr_unified.adapters.ref_adapter import RefAdapter
from mr_unified.cases import Bytes, ByteRandom, Case, decode, format_node
from mr_unified.formula import QUANT_OPS, Env, Node, paths, ref_eval
from mr_unified.guidance import guidance_value
from mr_unified.harness import run_input
from mr_unified.oracle import Violation, check_rewrite, rewrite_gap
from mr_unified.rewrites import PROP_ATOMS, QRULES, RULES, random_formula, valid_for

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEEDS = range(40)
REF = RefAdapter()


class FoldRef(RefAdapter):
    """Reference semantics, but declares fold quantifiers like LNN, so minmax-only rules must not be decoded."""

    name = "foldref"

    def declared_quantifiers(self):
        """forall and exists are both the strong fold."""
        return {"forall": "fold", "exists": "fold"}


class ShiftedRef(RefAdapter):
    """Planted disagreement: the reference value plus 0.1, so every cross check with it must fail."""

    name = "shifted"

    def evaluate(self, node: Node, env: Env) -> float:
        """The reference value plus 0.1."""
        return super().evaluate(node, env) + 0.1


class Refusing(RefAdapter):
    """Raises Unsupported on every formula, so every case must be counted as skipped."""

    name = "refusing"

    def evaluate(self, node: Node, env: Env) -> float:
        """Always refuses."""
        raise Unsupported("refuses every formula")


class NoClampAdapter:
    """Planted bug: implication is 1 - a + b with no min(., 1) clamp. Propositional formulas only."""

    name = "planted"
    dtype = "float64"

    def declared_quantifiers(self):
        """forall is min and exists is max, as in the reference."""
        return {"forall": "min", "exists": "max"}

    def dtype_for(self, node: Node) -> str:
        """Always float64."""
        return "float64"

    def evaluate(self, node: Node, env: Env) -> float:
        """Reference value, except that imp is not clamped."""
        if node.op in QUANT_OPS or (node.op == "atom" and node.args):
            raise Unsupported("planted adapter is propositional")
        if node.op == "atom":
            return env.atoms[node.name]
        if node.op == "const":
            return node.value
        kids = [self.evaluate(kid, env) for kid in node.kids]
        if node.op == "imp":
            return 1.0 - kids[0] + kids[1]
        return float(ref_eval(Node(op=node.op, kids=tuple(Node(op="const", value=v) for v in kids)), env))


def _inputs(count: int, seed: int = 0) -> List[bytes]:
    """`count` random inputs drawn from a generator seeded with `seed`."""
    rng = random.Random(seed)
    return [harness.random_input(rng) for _ in range(count)]


def _decode_all(adapters, count: int = 300) -> List:
    """Decoded cases for `count` random inputs."""
    return [decode(data, adapters) for data in _inputs(count)]


def _prop_env(rng: random.Random) -> Env:
    """A propositional environment with random truth values for p0 and p1."""
    return Env(atoms={name: rng.random() for name in PROP_ATOMS}, preds={}, domain=())


@pytest.fixture(autouse=True)
def restore_harness_state():
    """Each test starts and ends with the harness module state it found: main() and report() change it."""
    saved = (harness.ACTIVE, harness.SELECTION, set(harness._SEEN), set(harness._CRASH_SEEN), copy.deepcopy(harness.STATS))
    yield
    harness.ACTIVE, harness.SELECTION = saved[0], saved[1]
    harness._SEEN.clear()
    harness._SEEN.update(saved[2])
    harness._CRASH_SEEN.clear()
    harness._CRASH_SEEN.update(saved[3])
    harness.STATS.clear()
    harness.STATS.update(saved[4])


def _has_quantifier(node: Node) -> bool:
    """True if some node of `node` is a forall or exists."""
    return node.op in QUANT_OPS or any(_has_quantifier(kid) for kid in node.kids)


def _first_violation(adapters, limit: int = 5000):
    """The first (input, violation) that `adapters` produce on random inputs, or fail the test."""
    for seed in range(limit):
        data = harness.random_input(random.Random(seed))
        found = run_input(data, adapters)
        if found:
            return data, found[0]
    raise AssertionError("no violation found in %d inputs" % limit)


DECODE_SCRIPT = """
from mr_unified.adapters.ref_adapter import RefAdapter
from mr_unified.cases import decode, format_node
for i in range(30):
    data = bytes((i * 37 + j * 11) % 256 for j in range(64))
    case = decode(data, [RefAdapter()])
    print(case.kind, case.name, case.path, format_node(case.node))
"""


def _decode_output(hashseed: str) -> str:
    """Stdout of decoding 30 fixed inputs in a fresh process with the given PYTHONHASHSEED."""
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    done = subprocess.run(
        [sys.executable, "-c", DECODE_SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout


def test_decode_is_deterministic_across_processes():
    first = _decode_output("1")
    second = _decode_output("2")
    assert len(first.splitlines()) == 30
    assert first == second


def test_short_input_decodes_like_zero_padded_input():
    rng = random.Random(7)
    compared = 0
    for _ in SEEDS:
        short = bytes(rng.getrandbits(8) for _ in range(5))
        assert decode(short, [REF]) == decode(short + bytes(600), [REF])
        compared += 1
    assert compared == len(SEEDS)
    assert decode(b"", [REF]) == decode(bytes(1024), [REF])


def test_decode_respects_declared_quantifiers():
    cases = _decode_all([FoldRef()])
    quant = [case for case in cases if case.kind == "qrewrite" and case.path is not None]
    assert quant
    for case in quant:
        assert valid_for(QRULES[case.name], {"forall": "fold", "exists": "fold"})


def test_cross_kind_needs_two_adapters():
    assert all(case.kind != "cross" for case in _decode_all([REF]))
    cross = [case for case in _decode_all([REF, FoldRef()]) if case.kind == "cross"]
    assert cross


def test_b7_is_decoded_only_on_one_object_domain():
    b7 = [case for case in _decode_all([REF], 600) if case.name == "B7_singleton"]
    assert b7
    for case in b7:
        assert len(case.env.domain) == 1


def test_bytes_past_the_end_read_as_zero():
    stream = Bytes(b"")
    assert stream.int(2, 5) == 2
    assert stream.unit() == 0.0


def test_bytes_int_covers_its_whole_range():
    stream = Bytes(bytes(range(256)))
    values = {stream.int(2, 5) for _ in range(256)}
    assert values == {2, 3, 4, 5}


def _draws(rng: ByteRandom) -> list:
    """A fixed sequence of draws of each kind, used to compare two streams."""
    out = []
    for _ in range(15):
        out.append(rng.random())
        out.append(rng.getrandbits(7))
        out.append(rng.choice([10, 20, 30]))
        out.append(rng.randint(0, 9))
    return out


def test_byte_random_draws_are_reproducible():
    data = bytes(range(200))
    first = _draws(ByteRandom(Bytes(data)))
    second = _draws(ByteRandom(Bytes(data)))
    assert first == second
    assert len(first) == 60


def test_byte_random_getrandbits_stays_below_two_to_k():
    rng = ByteRandom(Bytes(harness.random_input(random.Random(3))))
    compared = 0
    for k in range(1, 17):
        for _ in range(50):
            assert 0 <= rng.getrandbits(k) < 2 ** k
            compared += 1
    assert compared == 16 * 50


def test_format_node_prints_connectives_readably():
    node = Node(op="and_s", kids=(Node(op="atom", name="p0"), Node(op="not", kids=(Node(op="atom", name="p1"),))))
    assert format_node(node) == "and_s(p0, not(p1))"


def test_guidance_value_scales_a_gap_equal_to_tolerance_to_one_thousandth():
    assert guidance_value(1e-6, 1e-6) == pytest.approx(1e-3)


def test_guidance_value_is_zero_for_a_zero_gap():
    assert guidance_value(0.0, 1e-6) == 0.0


def test_guidance_value_is_capped_at_one():
    assert guidance_value(1e9, 1e-6) == 1.0


def test_guidance_value_counts_nan_as_a_failure():
    assert guidance_value(float("nan"), 1e-6) == 1.0


@pytest.mark.parametrize("adapter", [REF, ShiftedRef()], ids=["ref", "shifted"])
def test_rewrite_gap_agrees_with_check_rewrite(adapter):
    compared = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        node = random_formula(rng, 4, PROP_ATOMS)
        env = _prop_env(rng)
        for rule in RULES.values():
            for path in paths(node):
                gap, bound = rewrite_gap(adapter, rule, node, path, env)
                if math.isnan(gap):
                    continue
                compared += 1
                assert (check_rewrite(adapter, rule, node, path, env) is None) == (gap <= bound)
    assert compared > 0


def test_every_input_is_counted_once_when_the_adapter_refuses():
    harness.reset()
    inputs = _inputs(200)
    for data in inputs:
        run_input(data, [Refusing()])
    counted = sum(entry["run"] + entry["skipped"] for entry in harness.STATS.values())
    skipped = sum(entry["skipped"] for entry in harness.STATS.values())
    violations = sum(entry["violations"] for entry in harness.STATS.values())
    assert counted == len(inputs)
    assert skipped > 0
    assert violations == 0


def test_cross_check_flags_the_shifted_adapter():
    harness.reset()
    found: List[Violation] = []
    for data in _inputs(300):
        found.extend(run_input(data, [REF, ShiftedRef()]))
    crosses = [v for v in found if v.rule == "cross"]
    assert crosses
    assert harness.STATS["cross"]["run"] > 0


class Crashing(RefAdapter):
    """Raises ValueError (not Unsupported) on every formula, like an adapter that rejects object constants."""

    name = "crashing"

    def evaluate(self, node: Node, env: Env) -> float:
        """Always raises ValueError."""
        raise ValueError("adapter crash")


def test_adapter_crash_is_recorded_as_violation_with_infinite_margin():
    harness.reset()
    found: List[Violation] = []
    for data in _inputs(300):
        found.extend(run_input(data, [Crashing()]))
    crashes = [v for v in found if math.isinf(v.margin)]
    assert crashes
    assert harness.STATS["property"]["violations"] > 0


def test_finding_is_written_once_per_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("MR_UNIFIED_FINDINGS", str(tmp_path))
    harness.reset()
    env = Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=())
    violation = Violation("planted", "separating", (), Node(op="atom", name="p0"), Node(op="atom", name="p0"),
                          env, (0.0, 1.0), 1.0)
    first, second = b"\x01\x02", b"\x03\x04"
    harness.report(first, [NoClampAdapter()], [violation], "planted")
    harness.report(second, [NoClampAdapter()], [violation], "planted")
    with open(os.path.join(str(tmp_path), "planted", "separating", "input.bin"), "rb") as handle:
        assert handle.read() == first


def test_finding_written_and_replays(tmp_path, monkeypatch):
    monkeypatch.setenv("MR_UNIFIED_FINDINGS", str(tmp_path))
    monkeypatch.setitem(harness.REGISTRY, "planted", NoClampAdapter)
    harness.reset()
    adapters = harness.make_adapters("planted")
    data, violation = _first_violation(adapters)
    harness.report(data, adapters, [violation], "planted")
    folder = os.path.join(str(tmp_path), "planted", violation.rule)
    with open(os.path.join(folder, "meta.json")) as handle:
        meta = json.load(handle)
    assert meta["rule"] == violation.rule
    assert meta["formula"] == format_node(violation.before)
    replayed = replay.replay(os.path.join(folder, "input.bin"), "planted")
    assert [v.rule for v in replayed] == [violation.rule]
    assert replayed[0].before == violation.before


def test_replay_prints_no_violation_for_a_clean_input(tmp_path, capsys):
    harness.reset()
    clean = next(data for data in _inputs(200) if not run_input(data, [REF]))
    path = os.path.join(str(tmp_path), "clean.bin")
    with open(path, "wb") as handle:
        handle.write(clean)
    assert replay.main([path, "--lib", "ref"]) == 0
    assert "no violation" in capsys.readouterr().out


def test_random_mode_runs_clean_on_ref_in_process():
    harness.reset()
    assert harness.main(["--random", "500", "--lib", "ref", "--seed", "0"]) == 0
    for kind in ("rewrite", "qrewrite", "env_rule", "property"):
        assert harness.STATS[kind]["run"] > 0


def test_random_mode_exits_zero_on_ref_in_a_subprocess():
    done = subprocess.run(
        [sys.executable, "-m", "mr_unified.harness", "--random", "500", "--lib", "ref"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "rewrite" in done.stdout


class FoldSemantics(RefAdapter):
    """A fake adapter with fold semantics: its values are ref_eval(fold) and it declares fold quantifiers."""

    name = "foldsem"

    def declared_quantifiers(self):
        """forall and exists are the strong fold."""
        return {"forall": "fold", "exists": "fold"}

    def evaluate(self, node: Node, env: Env) -> float:
        """ref_eval with the fold quantifiers."""
        return float(ref_eval(node, env, "fold"))


class Undeclared(RefAdapter):
    """An adapter that declares no quantifier semantics at all, so B6 and B8 cannot be checked on it."""

    name = "undeclared"

    def declared_quantifiers(self):
        """No declaration."""
        return {}


def test_split_margin_is_zero_for_fold_declared_adapter():
    rng = random.Random(5)
    compared = 0
    for _ in SEEDS:
        body = random_formula(rng, 3, PROP_ATOMS)
        node = Node(op="forall", name="x", kids=(body,))
        env = _prop_env(rng)
        env = Env(atoms=env.atoms, preds={}, domain=("o0", "o1", "o2"))
        assert harness._quant_of(FoldSemantics()) == "fold"
        assert harness._margin("split", FoldSemantics(), node, env, rng) <= 1e-12
        compared += 1
    assert compared == len(SEEDS)


def test_quant_of_minmax_adapter_is_minmax():
    assert harness._quant_of(REF) == "minmax"


def test_adapter_without_declared_quantifiers_is_skipped_for_split():
    harness.reset()
    node = Node(op="forall", name="x", kids=(Node(op="atom", name="p0"),))
    env = Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=("o0",))
    case = Case("property", node, env, "split", (), tail=bytes(64))
    assert harness._run_on(case, Undeclared()) == []
    assert harness.STATS["property"]["skipped"] == 1
    assert harness.STATS["property"]["run"] == 0


def test_cross_kind_on_ltn_and_lnn_generates_no_quantified_formula():
    from mr_unified.adapters.lnn_adapter import LNNAdapter
    from mr_unified.adapters.ltn_adapter import LTNAdapter

    cross = [case for case in _decode_all([LTNAdapter(), LNNAdapter()], 600) if case.kind == "cross"]
    assert cross
    assert not any(_has_quantifier(case.node) for case in cross)


def test_cross_kind_on_equal_quantifiers_generates_quantified_formula():
    cross = [case for case in _decode_all([REF, ShiftedRef()], 600) if case.kind == "cross"]
    assert any(_has_quantifier(case.node) for case in cross)


def test_bucket_receives_the_guidance_value_of_a_rewrite_gap(monkeypatch):
    seen = []
    monkeypatch.setattr(harness, "bucket", seen.append)
    case = next(c for c in _decode_all([REF]) if c.kind == "rewrite" and c.path is not None)
    rule = harness.ALL_REWRITES[case.name]
    expected = guidance_value(*rewrite_gap(REF, rule, case.node, case.path, case.env))
    harness._check(case, REF)
    assert seen == [pytest.approx(expected)]


def test_bucket_receives_the_guidance_value_of_a_property_margin(monkeypatch):
    seen = []
    monkeypatch.setattr(harness, "bucket", seen.append)
    node = Node(op="and_s", kids=(Node(op="atom", name="p0"), Node(op="atom", name="p1")))
    env = Env(atoms={"p0": 0.3, "p1": 0.6}, preds={}, domain=())
    case = Case("property", node, env, "monotone", (), tail=bytes(range(1, 60)))
    harness._check(case, REF)
    assert len(seen) == 1
    assert 0.0 <= seen[0] <= 1.0


def test_harness_internal_value_error_propagates(monkeypatch):
    def broken(name, adapter, node, env, rng):
        """A harness bug: raises a ValueError that is not an adapter exception."""
        raise ValueError("harness bug")

    monkeypatch.setattr(harness, "_margin", broken)
    node = Node(op="and_s", kids=(Node(op="atom", name="p0"), Node(op="atom", name="p1")))
    env = Env(atoms={"p0": 0.3, "p1": 0.6}, preds={}, domain=())
    case = Case("property", node, env, "monotone", (), tail=bytes(range(1, 60)))
    with pytest.raises(ValueError):
        harness._check(case, REF)


class Crashing(RefAdapter):
    """Raises ValueError (not Unsupported) on every formula, as an adapter that rejects an input would."""

    name = "crashing"

    def evaluate(self, node: Node, env: Env) -> float:
        """Always raises ValueError."""
        raise ValueError("adapter crash")


def test_adapter_value_error_is_recorded_as_crash_violation():
    harness.reset()
    node = Node(op="and_s", kids=(Node(op="atom", name="p0"), Node(op="atom", name="p1")))
    env = Env(atoms={"p0": 0.3, "p1": 0.6}, preds={}, domain=())
    case = Case("property", node, env, "monotone", (), tail=bytes(range(1, 60)))
    found = harness._run_on(case, Crashing())
    assert len(found) == 1
    assert found[0].rule == "monotone:crash"
    assert math.isinf(found[0].margin)


def test_crash_and_real_violation_of_same_rule_are_both_written(tmp_path, monkeypatch):
    monkeypatch.setenv("MR_UNIFIED_FINDINGS", str(tmp_path))
    harness.reset()
    env = Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=())
    node = Node(op="atom", name="p0")
    real = Violation("planted", "separating", (), node, node, env, (0.0, 1.0), 1.0)
    crash = Violation("planted", "separating:crash", (), node, node, env, (float("nan"),) * 2, float("inf"))
    harness.report(b"\x01", [NoClampAdapter()], [real, crash], "planted")
    assert os.path.isdir(os.path.join(str(tmp_path), "planted", "separating"))
    assert os.path.isdir(os.path.join(str(tmp_path), "planted", "separating:crash"))


def test_empty_rule_label_is_refused_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MR_UNIFIED_FINDINGS", str(tmp_path))
    with pytest.raises(ValueError):
        findings_io.write_finding("ltn", "", b"\x00", {})
    assert not os.path.exists(os.path.join(str(tmp_path), "ltn"))


def test_finding_meta_is_strict_json_with_infinity_as_string(tmp_path, monkeypatch):
    monkeypatch.setenv("MR_UNIFIED_FINDINGS", str(tmp_path))
    harness.reset()
    env = Env(atoms={"p0": 0.5, "p1": 0.5}, preds={}, domain=())
    node = Node(op="atom", name="p0")
    crash = Violation("planted", "separating:crash", (), node, node, env, (float("nan"),) * 2, float("inf"))
    harness.report(b"\x01", [NoClampAdapter()], [crash], "planted")
    path = os.path.join(str(tmp_path), "planted", "separating:crash", "meta.json")
    with open(path) as handle:
        meta = json.loads(handle.read(), parse_constant=_reject_constant)
    assert meta["margin"] == "inf"
    assert meta["got"] == ["nan", "nan"]


def _reject_constant(name: str):
    """json.loads hook: NaN, Infinity and -Infinity are not strict JSON, so they raise."""
    raise ValueError("non-strict JSON constant: %s" % name)
