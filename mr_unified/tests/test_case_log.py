"""Tests for the case log: --log writes one parseable line per case, and the reader prints formula, rewrite and env."""
import json

from mr_unified import case_log, harness

COUNT = 60


def _run_logged(tmp_path, count=COUNT):
    """Run `count` random inputs on ref with the log on; return the parsed records and the log path."""
    path = str(tmp_path / "cases.jsonl")
    harness.reset()
    harness.configure("ref")
    assert harness.main(["--lib", "ref", "--random", str(count), "--log", path]) == 0
    with open(path) as handle:
        return [json.loads(line) for line in handle], path


def test_log_has_one_record_per_input(tmp_path):
    records, _ = _run_logged(tmp_path)
    assert len(records) == COUNT
    assert [rec["n"] for rec in records] == list(range(COUNT))


def test_every_record_has_formula_and_env(tmp_path):
    records, _ = _run_logged(tmp_path)
    checked = 0
    for rec in records:
        assert rec["formula"]
        assert set(rec["env"]) == {"atoms", "preds", "domain"}
        checked += 1
    assert checked > 0


def test_rewrite_records_show_the_rewritten_formula(tmp_path):
    records, _ = _run_logged(tmp_path)
    rewrites = [rec for rec in records if rec["kind"] in ("rewrite", "qrewrite") and rec["path"] is not None]
    assert rewrites
    for rec in rewrites:
        assert rec["after"] and rec["after"] != rec["formula"]


def test_skipped_case_has_no_path_and_counts_as_skipped(tmp_path):
    records, _ = _run_logged(tmp_path, 200)
    unmatched = [rec for rec in records if rec["kind"] in ("rewrite", "qrewrite") and rec["path"] is None]
    assert unmatched
    for rec in unmatched:
        assert rec["skipped"] == 1 and rec["run"] == 0 and rec["after"] is None


def test_reader_prints_formula_rewrite_and_env(tmp_path, capsys):
    records, path = _run_logged(tmp_path)
    case_log.main([path, "--limit", "5"])
    out = capsys.readouterr().out
    assert "formula:" in out and "atoms:" in out and "domain:" in out
    assert out.strip().splitlines()[-1].startswith("cases: ")


def test_reader_filters_by_kind(tmp_path, capsys):
    _, path = _run_logged(tmp_path)
    case_log.main([path, "--kind", "property"])
    out = capsys.readouterr().out
    assert " property " in out and " rewrite " not in out


def test_no_log_file_is_written_without_the_flag(tmp_path):
    harness.reset()
    harness.configure("ref")
    assert harness.main(["--lib", "ref", "--random", "5"]) == 0
    assert not case_log.is_open()
    assert list(tmp_path.iterdir()) == []


def test_rendering_an_env_with_predicate_tables():
    env = {"atoms": {"p0": 0.5}, "preds": {"P": {"o0": 0.25}, "R": {}}, "domain": ["o0"]}
    lines = case_log._env_lines(env)
    assert lines[0] == "atoms:   p0=0.5" and lines[1] == "domain:  o0"
    assert lines[2].endswith("P(o0)=0.25") and len(lines) == 3

