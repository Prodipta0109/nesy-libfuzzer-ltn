"""Tests for the LTN-only extension checks (spec Task 11, van Krieken et al. Section 6 and Section 7).
Clean run: every check compares cases and reports nothing. Mutant runs: each check has a patched operator
that must make that check report a violation. Tie and endpoint handling: skipped cases are counted, not dropped.
"""
import random
import warnings
from unittest import mock

import pytest
import torch

# ltn emits DeprecationWarnings for invalid escape sequences in its own docstrings at import time.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="invalid escape sequence", category=DeprecationWarning)
    pytest.importorskip("ltn")
    from ltn import fuzzy_ops as fz
    from mr_unified import ltn_extras
    from mr_unified.ltn_extras import ENDPOINT_EVERY, run_extras

MUTANT_CASES = 100
N_CASES = 200
EXPECTED_CHECKS = (
    "AndLuk gradient",
    "OrLuk gradient",
    "AndMin gradient",
    "OrMax gradient",
    "AggregMin gradient",
    "AggregPMeanError gradient",
    "duality Lukasiewicz",
    "duality min/max",
    "duality product/probsum",
    "Godel invariance AndMin",
    "Godel invariance OrMax",
    "Godel invariance ImpliesGodel",
    "Godel invariance NotGodel",
    "AggregPMean p=1 equals AggregMean",
    "AggregPMeanError p=1 equals AggregMean",
    "AggregPMeanError non-increasing in p",
    "AggregPMean non-decreasing in p",
)


class _ZeroGradAndLuk(torch.autograd.Function):
    """Lukasiewicz conjunction whose backward returns zeros, so its gradient is wrong everywhere."""

    @staticmethod
    def forward(ctx, x, y):
        return torch.clamp(x + y - 1.0, min=0.0)

    @staticmethod
    def backward(ctx, grad_out):
        return torch.zeros_like(grad_out), torch.zeros_like(grad_out)


def _violated_checks(patch) -> set:
    """Names of the checks that report a violation when run_extras runs with `patch` active."""
    with patch:
        result = run_extras(n_cases=MUTANT_CASES, seed=0)
    return {violation.name for violation in result.violations}


def _pmean_error_with_p_plus_one(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
    """Mutant: pMeanError with p+1 in the inner power, so the gradient differs from the closed form."""
    power = self.p if p is None else p
    inner = torch.mean(torch.pow(1.0 - xs, power + 1), dim=dim, keepdim=keepdim)
    return 1.0 - torch.pow(inner, 1.0 / power)


def _luk_implies(self, x, y):
    """Mutant: Godel implication replaced by the Lukasiewicz formula min(1, 1 - a + b)."""
    return torch.minimum(1.0 - x + y, torch.ones_like(x))


def _half_at_zero_not(self, x):
    """Mutant: Godel negation returning 0.5 at 0 and 0 elsewhere (a squaring-invariance violation at 0)."""
    return torch.where(torch.eq(x, 0.0), torch.full_like(x, 0.5), torch.zeros_like(x))


def _scaled_by_p(original):
    """Mutant factory: an AggregPMeanError call whose result is multiplied by p."""

    def scaled(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
        return self.p * original(self, xs, dim=dim, keepdim=keepdim, mask=mask, p=p, stable=stable)

    return scaled


def _divided_by_p(original):
    """Mutant factory: an AggregPMean call whose result is divided by p."""

    def divided(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
        return original(self, xs, dim=dim, keepdim=keepdim, mask=mask, p=p, stable=stable) / self.p

    return divided


def _forced_tie(rng, index, size_range):
    """Stand-in for the case draw: always the tie [0.5, 0.5], so a pair check must skip every case."""
    return [0.5, 0.5]


# ----- clean run -------------------------------------------------------------------------------------------------


def test_extras_clean_on_unmutated_ltn():
    assert run_extras().violations == []


def test_extras_reports_the_expected_check_names():
    result = run_extras(n_cases=N_CASES)
    assert set(result.case_counts) == set(EXPECTED_CHECKS)
    assert set(result.skip_counts) == set(EXPECTED_CHECKS)


def test_extras_compares_at_least_one_case_for_every_check():
    result = run_extras(n_cases=N_CASES)
    assert all(result.case_counts[name] > 0 for name in EXPECTED_CHECKS)


def test_case_counts_plus_skips_equal_n_cases():
    result = run_extras(n_cases=N_CASES)
    for name in EXPECTED_CHECKS:
        total = result.case_counts[name] + result.skip_counts[name]
        assert total == N_CASES, name


def test_extras_skips_bounded_by_endpoint_cases():
    result = run_extras(n_cases=N_CASES)
    endpoint_cases = N_CASES // ENDPOINT_EVERY
    assert all(result.skip_counts[name] <= endpoint_cases for name in EXPECTED_CHECKS)


def test_forced_tie_is_skipped_and_counted():
    with mock.patch.object(ltn_extras, "_draw_case", _forced_tie):
        outcome = ltn_extras.check_andmin_gradient(random.Random(0), 7)
    assert outcome.compared == 0
    assert outcome.skipped == 7
    assert outcome.violations == []


# ----- mutant runs, one per check --------------------------------------------------------------------------------


def test_extras_catch_wrong_gradient_mutant():
    patch = mock.patch.object(fz.AndLuk, "__call__", lambda self, x, y: _ZeroGradAndLuk.apply(x, y))
    assert "AndLuk gradient" in _violated_checks(patch)


def test_extras_orluk_without_clamp_fails_orluk_gradient():
    patch = mock.patch.object(fz.OrLuk, "__call__", lambda self, x, y: x + y)
    assert "OrLuk gradient" in _violated_checks(patch)


def test_extras_andmin_as_max_fails_andmin_gradient():
    patch = mock.patch.object(fz.AndMin, "__call__", lambda self, x, y: torch.maximum(x, y))
    assert "AndMin gradient" in _violated_checks(patch)


def test_extras_ormax_as_min_fails_ormax_gradient():
    patch = mock.patch.object(fz.OrMax, "__call__", lambda self, x, y: torch.minimum(x, y))
    assert "OrMax gradient" in _violated_checks(patch)


def test_extras_aggregmin_as_mean_fails_aggregmin_gradient():
    patch = mock.patch.object(
        fz.AggregMin, "__call__", lambda self, xs, dim=None, keepdim=False, mask=None: torch.mean(xs, dim=dim)
    )
    assert "AggregMin gradient" in _violated_checks(patch)


def test_extras_pmeanerror_p_plus_one_fails_pmeanerror_gradient():
    patch = mock.patch.object(fz.AggregPMeanError, "__call__", _pmean_error_with_p_plus_one)
    assert "AggregPMeanError gradient" in _violated_checks(patch)


def test_extras_orluk_as_max_fails_lukasiewicz_duality():
    patch = mock.patch.object(fz.OrLuk, "__call__", lambda self, x, y: torch.maximum(x, y))
    assert "duality Lukasiewicz" in _violated_checks(patch)


def test_extras_ormax_as_min_fails_minmax_duality():
    patch = mock.patch.object(fz.OrMax, "__call__", lambda self, x, y: torch.minimum(x, y))
    assert "duality min/max" in _violated_checks(patch)


def test_extras_probsum_with_wrong_cross_term_fails_product_duality():
    patch = mock.patch.object(fz.OrProbSum, "__call__", lambda self, x, y, stable=None: x + y - 2.0 * x * y)
    assert "duality product/probsum" in _violated_checks(patch)


def test_extras_andmin_as_mean_fails_godel_invariance_andmin():
    patch = mock.patch.object(fz.AndMin, "__call__", lambda self, x, y: 0.5 * (x + y))
    assert "Godel invariance AndMin" in _violated_checks(patch)


def test_extras_ormax_as_probsum_fails_godel_invariance_ormax():
    patch = mock.patch.object(fz.OrMax, "__call__", lambda self, x, y: x + y - x * y)
    assert "Godel invariance OrMax" in _violated_checks(patch)


def test_extras_godel_implies_as_luk_fails_godel_invariance_implies():
    patch = mock.patch.object(fz.ImpliesGodel, "__call__", _luk_implies)
    assert "Godel invariance ImpliesGodel" in _violated_checks(patch)


def test_extras_notgodel_half_at_zero_fails_godel_invariance_notgodel():
    patch = mock.patch.object(fz.NotGodel, "__call__", _half_at_zero_not)
    assert "Godel invariance NotGodel" in _violated_checks(patch)


def test_extras_pmean_squared_fails_pmean_p1_equals_mean():
    patch = mock.patch.object(
        fz.AggregPMean,
        "__call__",
        lambda self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None: torch.pow(
            torch.mean(xs, dim=dim, keepdim=keepdim), 2.0
        ),
    )
    assert "AggregPMean p=1 equals AggregMean" in _violated_checks(patch)


def test_extras_pmeanerror_one_minus_mean_fails_pmeanerror_p1_equals_mean():
    patch = mock.patch.object(
        fz.AggregPMeanError,
        "__call__",
        lambda self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None: 1.0 - torch.mean(xs, dim=dim),
    )
    assert "AggregPMeanError p=1 equals AggregMean" in _violated_checks(patch)


def test_extras_pmeanerror_scaled_by_p_fails_non_increasing_in_p():
    patch = mock.patch.object(fz.AggregPMeanError, "__call__", _scaled_by_p(fz.AggregPMeanError.__call__))
    assert "AggregPMeanError non-increasing in p" in _violated_checks(patch)


def test_extras_pmean_divided_by_p_fails_non_decreasing_in_p():
    patch = mock.patch.object(fz.AggregPMean, "__call__", _divided_by_p(fz.AggregPMean.__call__))
    assert "AggregPMean non-decreasing in p" in _violated_checks(patch)
