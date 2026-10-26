"""Tests for the byte trace: the replay builds the same case as decode, and every byte read appears once, in order."""
import json
import random

from mr_unified import byte_trace, case_log, cases, harness
from mr_unified.adapters.ref_adapter import RefAdapter
from mr_unified.cases import ByteRandom, Bytes, decode

ADAPTERS = [RefAdapter()]
SEEDS = range(40)


def _data(seed):
    """1024 random bytes, as random mode draws them."""
    return random.Random(seed).randbytes(harness.INPUT_BYTES)


def test_traced_decode_builds_the_same_case_as_plain_decode():
    compared = 0
    for seed in SEEDS:
        data = _data(seed)
        plain = decode(data, ADAPTERS)
        traced, _, _ = byte_trace.trace_decode(data, ADAPTERS)
        assert (traced.kind, traced.node, traced.env, traced.name, traced.path) == (
            plain.kind, plain.node, plain.env, plain.name, plain.path)
        compared += 1
    assert compared == len(SEEDS)


def test_every_byte_read_is_one_draw_in_order():
    checked = 0
    for seed in SEEDS:
        _, draws, used = byte_trace.trace_decode(_data(seed), ADAPTERS)
        assert [draw.pos for draw in draws] == list(range(used)) and used > 0
        checked += 1
    assert checked == len(SEEDS)


def test_first_draw_is_the_kind_chosen_in_decode():
    _, draws, _ = byte_trace.trace_decode(_data(0), ADAPTERS)
    assert draws[0].who.startswith("decode ") and draws[0].how == "int" and "range 0.." in draws[0].result


def test_formula_draws_name_the_generator_functions_that_made_them():
    seen = set()
    for seed in SEEDS:
        _, draws, _ = byte_trace.trace_decode(_data(seed), ADAPTERS)
        seen.update(draw.who.split()[0] for draw in draws)
    assert {"decode", "random_formula", "_random_leaf"} <= seen


def test_choice_draws_carry_the_picked_item_and_rejected_redraws():
    results = []
    for seed in SEEDS:
        _, draws, _ = byte_trace.trace_decode(_data(seed), ADAPTERS)
        results.extend(draw.result for draw in draws if draw.how == "choice")
    assert results and all(results)
    assert any("(1 of" in result for result in results)


def test_cases_module_is_restored_after_a_replay():
    byte_trace.trace_decode(_data(1), ADAPTERS)
    assert cases.Bytes is Bytes and cases.ByteRandom is ByteRandom


def test_cases_module_is_restored_when_decode_raises():
    try:
        with byte_trace.tracing():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert cases.Bytes is Bytes and cases.ByteRandom is ByteRandom


def test_logged_input_replays_to_the_logged_formula(tmp_path):
    path = str(tmp_path / "cases.jsonl")
    harness.reset()
    harness.configure("ref")
    assert harness.main(["--lib", "ref", "--random", "30", "--log", path]) == 0
    with open(path) as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 30
    for rec in records:
        case, _, _ = byte_trace.trace_decode(bytes.fromhex(rec["input_hex"]), ADAPTERS)
        assert cases.format_node(case.node) == rec["formula"] and rec["lib"] == "ref"


def test_cli_prints_the_table_and_the_decoded_case(tmp_path, capsys):
    path = str(tmp_path / "cases.jsonl")
    harness.reset()
    harness.configure("ref")
    harness.main(["--lib", "ref", "--random", "5", "--log", path])
    capsys.readouterr()
    assert byte_trace.main(["--log", path, "--case", "2"]) == 0
    out = capsys.readouterr().out
    assert "decision" in out and "decoded case:" in out and "formula:" in out


def test_reader_shows_the_input_hex(tmp_path, capsys):
    path = str(tmp_path / "cases.jsonl")
    harness.reset()
    harness.configure("ref")
    harness.main(["--lib", "ref", "--random", "3", "--log", path])
    capsys.readouterr()
    case_log.main([path])
    assert "input:   1024 bytes, hex " in capsys.readouterr().out
