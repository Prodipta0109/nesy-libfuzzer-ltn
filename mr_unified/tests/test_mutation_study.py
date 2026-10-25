"""Tests for the mutation study (spec Task 9): the unmutated libraries are silent, every LTN mutant is caught by some
oracle, a patch puts the original operator back on exit (also after an error), and the table has the expected columns.
Case counts are small (BASELINE_CASES, CATCH_CASES) so that this file runs in well under 60 s.
"""
import pytest
import torch

from mr_unified.mutation_study import MUTANTS, format_table, kind, run, baseline

SEED = 0
BASELINE_CASES = 500
CATCH_CASES = 300
LTN_NAMES = [name for name, mutant in MUTANTS.items() if mutant.lib == "ltn"]


def _operator_probe():
    """Values of the LTN connectives and of AggregMin on a fixed grid, to see whether a patch is still active."""
    import ltn.fuzzy_ops as fz

    x = torch.tensor([0.0, 0.3, 0.6, 1.0], dtype=torch.float64)
    y = torch.tensor([1.0, 0.5, 0.2, 0.0], dtype=torch.float64)
    return [
        fz.AndLuk()(x, y),
        fz.OrLuk()(x, y),
        fz.ImpliesLuk()(x, y),
        fz.NotStandard()(x),
        fz.AggregMin()(x.reshape(2, 2), dim=(1,)),
    ]


def _same(left, right):
    """Whether two probe results hold equal tensors, position by position."""
    return len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))


def test_unmutated_libs_have_no_oracle_firing_on_ltn():
    pytest.importorskip("ltn")
    tallies = baseline("ltn", n_cases=BASELINE_CASES, seed=SEED)
    assert set(tallies) == {"per_operator", "full_reference", "mr"}
    for oracle, tally in tallies.items():
        assert tally.runs > 0, oracle
        assert not tally.fired, oracle


def test_unmutated_libs_have_no_oracle_firing_on_lnn():
    pytest.importorskip("lnn")
    tallies = baseline("lnn", n_cases=BASELINE_CASES, seed=SEED)
    assert set(tallies) == {"per_operator", "full_reference", "mr"}
    for oracle, tally in tallies.items():
        assert tally.runs > 0, oracle
        assert not tally.fired, oracle


def test_every_ltn_mutant_is_caught_by_something():
    pytest.importorskip("ltn")
    assert LTN_NAMES
    results = run("ltn", n_cases=CATCH_CASES, seed=SEED)
    assert set(results) == set(LTN_NAMES)
    missed = [name for name, fired in results.items() if not any(fired.values())]
    assert missed == []


@pytest.mark.parametrize("name", LTN_NAMES)
def test_patch_restores_original(name):
    pytest.importorskip("ltn")
    before = _operator_probe()
    with MUTANTS[name].patch():
        _operator_probe()
    assert _same(before, _operator_probe())


def test_patch_restores_original_after_an_error():
    pytest.importorskip("ltn")
    before = _operator_probe()
    with pytest.raises(RuntimeError):
        with MUTANTS["AndLuk uses min"].patch():
            raise RuntimeError("boom")
    assert _same(before, _operator_probe())


def test_patch_changes_andluk_only_while_active():
    import ltn.fuzzy_ops as fz

    pytest.importorskip("ltn")
    x = torch.tensor([0.6], dtype=torch.float64)
    y = torch.tensor([0.9], dtype=torch.float64)
    with MUTANTS["AndLuk uses min"].patch():
        assert fz.AndLuk()(x, y).item() == pytest.approx(0.6)
    assert fz.AndLuk()(x, y).item() == pytest.approx(0.5)


def test_table_has_expected_columns():
    results = {
        "demo": {"per_operator": True, "full_reference": True, "mr": True},
        "quiet": {"per_operator": False, "full_reference": False, "mr": False},
    }
    table = format_table(results)
    lines = table.splitlines()
    assert lines[0].split() == ["mutant", "per_operator", "full_reference", "mr", "kind"]
    assert "all three" in lines[1]
    assert "missed" in lines[2]
    assert "full_reference needs a reference evaluator; MR does not" in table


@pytest.mark.parametrize(
    "firing, label",
    [
        ({"per_operator": True, "full_reference": True, "mr": True}, "all three"),
        ({"per_operator": False, "full_reference": True, "mr": True}, "MR + full-reference only"),
        ({"per_operator": False, "full_reference": False, "mr": True}, "MR only"),
        ({"per_operator": True, "full_reference": False, "mr": False}, "per-operator only"),
        ({"per_operator": False, "full_reference": True, "mr": False}, "full-reference only"),
        ({"per_operator": False, "full_reference": False, "mr": False}, "missed"),
    ],
)
def test_kind_labels_name_the_firing_oracles(firing, label):
    assert kind(firing) == label
