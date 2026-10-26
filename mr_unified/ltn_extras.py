"""LTN-only extension checks (spec Task 11, van Krieken et al. Section 6 and Section 7).
Each check draws cases (random in (0, 1), with every ENDPOINT_EVERY-th case on the endpoints {0, 1}), evaluates an
ltn.fuzzy_ops operator (stable=False where it exists) and compares it with a closed form or a law. Cases where the
closed form has a kink or is undefined are skipped and counted, never silently dropped.
"""
import itertools
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from ltn import fuzzy_ops as fz

# Float64 closed-form comparisons: rounding costs a few ulps (about 2e-16 relative) of the largest intermediate value.
# GRAD_TOL and VALUE_TOL sit many orders above that, so rounding alone never fires.
GRAD_TOL = 1e-10
VALUE_TOL = 1e-12
# Kink margin: autograd picks one side of a kink arbitrarily when two candidates tie, so cases closer than TIE_GAP to
# a kink are skipped and counted, not compared.
TIE_GAP = 1e-9
PS = (1, 2, 3, 6)
EXPONENTS = (2, 3)
ENDPOINT_EVERY = 5
ENDPOINTS = (0.0, 1.0)


@dataclass
class ExtraViolation:
    """A failed extra check: its name, the inputs, the value got and the value the law expects."""

    name: str
    inputs: list
    got: float
    expected: float


@dataclass
class CheckOutcome:
    """Result of one check: its name, cases compared, cases skipped (kink or undefined), and violations."""

    name: str
    compared: int
    skipped: int
    violations: List[ExtraViolation]


@dataclass
class ExtraResult:
    """Result of run_extras: every violation, and per check the number of compared and skipped cases."""

    violations: List[ExtraViolation]
    case_counts: Dict[str, int]
    skip_counts: Dict[str, int]


# ----- case plumbing ---------------------------------------------------------------------------------------------


def _draw_case(rng: random.Random, index: int, size_range: Tuple[int, int]) -> List[float]:
    """Draw a vector of length in size_range: entries in (0, 1), or from {0, 1} on every ENDPOINT_EVERY-th case."""
    size = rng.randint(*size_range)
    if index % ENDPOINT_EVERY == ENDPOINT_EVERY - 1:
        return [rng.choice(ENDPOINTS) for _ in range(size)]
    return [rng.random() for _ in range(size)]


def _never_tie(values: Sequence[float]) -> bool:
    """No skip: the closed form is defined and smooth on every case of this check."""
    return False


def _pair_sum_tie(values: Sequence[float]) -> bool:
    """Kink of max(a + b - 1, 0) and min(a + b, 1): the tie |a + b - 1| < TIE_GAP is skipped."""
    return abs(values[0] + values[1] - 1.0) < TIE_GAP


def _close_pair(values: Sequence[float]) -> bool:
    """Kink of min and max: some two entries are closer than TIE_GAP, so the argmin or argmax is a tie."""
    return any(abs(a - b) < TIE_GAP for a, b in itertools.combinations(values, 2))


def _all_ones(values: Sequence[float]) -> bool:
    """Undefined closed form: if every entry is 1, the power mean of (1 - x) is 0 and its power 1/p - 1 is undefined."""
    return all(v == 1.0 for v in values)


def _check_cases(
    name: str,
    rng: random.Random,
    n_cases: int,
    size_range: Tuple[int, int],
    is_tie: Callable[[Sequence[float]], bool],
    check_one: Callable[[List[float]], List[ExtraViolation]],
) -> CheckOutcome:
    """Law-free driver: run check_one on the non-skipped cases, and count skipped cases instead of dropping them."""
    outcome = CheckOutcome(name, 0, 0, [])
    for index in range(n_cases):
        values = _draw_case(rng, index, size_range)
        if is_tie(values):
            outcome.skipped += 1
            continue
        outcome.compared += 1
        outcome.violations.extend(check_one(values))
    return outcome


# ----- tensor evaluation -----------------------------------------------------------------------------------------


def _vector_gradient(op: Callable, values: Sequence[float]) -> List[float]:
    """Autograd gradient, in float64, of the scalar op(xs) with respect to each entry of xs."""
    xs = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(op(xs), xs)
    return grad.tolist()


def _vector_value(op: Callable, values: Sequence[float]) -> float:
    """Value of op(xs) for the float64 tensor xs built from `values`."""
    return float(op(torch.tensor(values, dtype=torch.float64)))


# ----- operators under test --------------------------------------------------------------------------------------


def _and_luk(xs: torch.Tensor) -> torch.Tensor:
    """Lukasiewicz conjunction of the pair: max(a + b - 1, 0)."""
    return fz.AndLuk()(xs[0], xs[1])


def _or_luk(xs: torch.Tensor) -> torch.Tensor:
    """Lukasiewicz disjunction of the pair: min(a + b, 1)."""
    return fz.OrLuk()(xs[0], xs[1])


def _and_min(xs: torch.Tensor) -> torch.Tensor:
    """Godel conjunction of the pair: min(a, b)."""
    return fz.AndMin()(xs[0], xs[1])


def _or_max(xs: torch.Tensor) -> torch.Tensor:
    """Godel disjunction of the pair: max(a, b)."""
    return fz.OrMax()(xs[0], xs[1])


def _agg_min(xs: torch.Tensor) -> torch.Tensor:
    """Min aggregation of the vector: min over all entries."""
    return fz.AggregMin()(xs, dim=0)


def _product(xs: torch.Tensor) -> torch.Tensor:
    """Goguen conjunction of the pair: a * b, unstable."""
    return fz.AndProd(stable=False)(xs[0], xs[1])


def _probsum(xs: torch.Tensor) -> torch.Tensor:
    """Goguen disjunction of the pair: a + b - a * b, unstable."""
    return fz.OrProbSum(stable=False)(xs[0], xs[1])


def _implies_godel(xs: torch.Tensor) -> torch.Tensor:
    """Godel implication of the pair: 1 if a <= b else b."""
    return fz.ImpliesGodel()(xs[0], xs[1])


def _not_godel(xs: torch.Tensor) -> torch.Tensor:
    """Godel negation of the single entry: 1 if a == 0 else 0."""
    return fz.NotGodel()(xs[0])


def _pmean_op(p: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Operator on a vector: AggregPMean with the given p over dimension 0, unstable."""
    return lambda xs: fz.AggregPMean(p=p, stable=False)(xs, dim=0)


def _pmean_error_op(p: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Operator on a vector: AggregPMeanError with the given p over dimension 0, unstable."""
    return lambda xs: fz.AggregPMeanError(p=p, stable=False)(xs, dim=0)


def _mean_op(xs: torch.Tensor) -> torch.Tensor:
    """Mean aggregation of the vector: (1/n) sum of entries."""
    return fz.AggregMean()(xs, dim=0)


# ----- closed-form gradients -------------------------------------------------------------------------------------


def _andluk_grad(v: List[float]) -> List[float]:
    """Law (eq. 33): d(a (x) b)/da = d/db = 1 if a + b > 1 else 0."""
    value = 1.0 if v[0] + v[1] > 1 else 0.0
    return [value, value]


def _orluk_grad(v: List[float]) -> List[float]:
    """Law (eq. 33 dual): d(a (+) b)/da = d/db = 1 if a + b < 1 else 0."""
    value = 1.0 if v[0] + v[1] < 1 else 0.0
    return [value, value]


def _andmin_grad(v: List[float]) -> List[float]:
    """Law (eq. 32): d min(a, b)/da = 1 if a < b else 0, and symmetrically for b."""
    return [1.0 if v[0] < v[1] else 0.0, 1.0 if v[1] < v[0] else 0.0]


def _ormax_grad(v: List[float]) -> List[float]:
    """Law (eq. 32 dual): d max(a, b)/da = 1 if a > b else 0, and symmetrically for b."""
    return [1.0 if v[0] > v[1] else 0.0, 1.0 if v[1] > v[0] else 0.0]


def _aggmin_grad(v: List[float]) -> List[float]:
    """Law (eq. 14): d min(x)/dx_i = 1 at the argmin of x and 0 elsewhere."""
    argmin = v.index(min(v))
    return [1.0 if i == argmin else 0.0 for i in range(len(v))]


def _pmean_error_grad(p: int, v: List[float]) -> List[float]:
    """Law (eq. 19): d/dx_i = (1-x_i)^(p-1) (1/n)^(1/p) (sum_j (1-x_j)^p)^(1/p - 1) for pMeanError."""
    total = sum((1.0 - x) ** p for x in v)
    size = len(v)
    return [(1.0 - x) ** (p - 1) * (1.0 / size) ** (1.0 / p) * total ** (1.0 / p - 1.0) for x in v]


# ----- generic law comparisons -----------------------------------------------------------------------------------


def _gradient_mismatches(name: str, op: Callable, closed_form: Callable, values: List[float]) -> List[ExtraViolation]:
    """Law: the autograd gradient of op at values equals closed_form(values), entry by entry."""
    got = _vector_gradient(op, values)
    expected = closed_form(values)
    return [ExtraViolation(name, list(values), g, e) for g, e in zip(got, expected) if abs(g - e) > GRAD_TOL]


def _duality_mismatches(name: str, conj: Callable, disj: Callable, values: List[float]) -> List[ExtraViolation]:
    """Law (Section 7): d conj(1-a, 1-b)/d(1-a) == d disj(a, b)/da."""
    a, b = values
    dual = _vector_gradient(conj, [1.0 - a, 1.0 - b])[0]
    direct = _vector_gradient(disj, [a, b])[0]
    if abs(dual - direct) > GRAD_TOL:
        return [ExtraViolation(name, [a, b], dual, direct)]
    return []


def _invariance_mismatches(name: str, op: Callable, values: List[float]) -> List[ExtraViolation]:
    """Law: op(f(x)) == f(op(x)) for f(t) = t**k, k in {2, 3} (Godel order-invariance)."""
    found: List[ExtraViolation] = []
    for k in EXPONENTS:
        got = _vector_value(op, [v ** k for v in values])
        expected = _vector_value(op, values) ** k
        if abs(got - expected) > VALUE_TOL:
            found.append(ExtraViolation(name, list(values), got, expected))
    return found


def _equal_mismatches(name: str, got_op: Callable, expected_op: Callable, values: List[float]) -> List[ExtraViolation]:
    """Law: got_op(x) == expected_op(x) for the vector x, up to VALUE_TOL."""
    got = _vector_value(got_op, values)
    expected = _vector_value(expected_op, values)
    if abs(got - expected) > VALUE_TOL:
        return [ExtraViolation(name, list(values), got, expected)]
    return []


def _order_mismatches(
    name: str, values: List[float], series: List[float], non_increasing: bool
) -> List[ExtraViolation]:
    """Law: along increasing p the series never moves the wrong way (non-increasing or non-decreasing)."""
    for earlier, later in zip(series, series[1:]):
        wrong_way = later > earlier + VALUE_TOL if non_increasing else earlier > later + VALUE_TOL
        if wrong_way:
            return [ExtraViolation(name, list(values), later, earlier)]
    return []


def _pmean_error_series(values: List[float]) -> List[float]:
    """AggregPMeanError of the vector at each p in PS, in increasing order of p."""
    return [_vector_value(_pmean_error_op(p), values) for p in PS]


def _pmean_series(values: List[float]) -> List[float]:
    """AggregPMean of the vector at each p in PS, in increasing order of p."""
    return [_vector_value(_pmean_op(p), values) for p in PS]


def _pmean_error_gradient_all_p(name: str, values: List[float]) -> List[ExtraViolation]:
    """Law (eq. 19) for p in {1, 2, 3}: the autograd gradient equals the closed form."""
    found: List[ExtraViolation] = []
    for p in (1, 2, 3):
        found += _gradient_mismatches(name, _pmean_error_op(p), partial(_pmean_error_grad, p), values)
    return found


def _pmean_error_monotone(name: str, values: List[float]) -> List[ExtraViolation]:
    """Law: AggregPMeanError is non-increasing in p, on the vector."""
    return _order_mismatches(name, values, _pmean_error_series(values), non_increasing=True)


def _pmean_monotone(name: str, values: List[float]) -> List[ExtraViolation]:
    """Law: AggregPMean is non-decreasing in p, on the vector."""
    return _order_mismatches(name, values, _pmean_series(values), non_increasing=False)


# ----- one function per check ------------------------------------------------------------------------------------


def check_andluk_gradient(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (eq. 33): the autograd gradient of AndLuk equals 1 if a + b > 1 else 0."""
    name = "AndLuk gradient"
    check_one = partial(_gradient_mismatches, name, _and_luk, _andluk_grad)
    return _check_cases(name, rng, n_cases, (2, 2), _pair_sum_tie, check_one)


def check_orluk_gradient(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (eq. 33 dual): the autograd gradient of OrLuk equals 1 if a + b < 1 else 0."""
    name = "OrLuk gradient"
    check_one = partial(_gradient_mismatches, name, _or_luk, _orluk_grad)
    return _check_cases(name, rng, n_cases, (2, 2), _pair_sum_tie, check_one)


def check_andmin_gradient(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (eq. 32): the autograd gradient of AndMin is 1 for the smaller argument and 0 for the other."""
    name = "AndMin gradient"
    check_one = partial(_gradient_mismatches, name, _and_min, _andmin_grad)
    return _check_cases(name, rng, n_cases, (2, 2), _close_pair, check_one)


def check_ormax_gradient(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (eq. 32 dual): the autograd gradient of OrMax is 1 for the larger argument and 0 for the other."""
    name = "OrMax gradient"
    check_one = partial(_gradient_mismatches, name, _or_max, _ormax_grad)
    return _check_cases(name, rng, n_cases, (2, 2), _close_pair, check_one)


def check_aggregmin_gradient(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (eq. 14): the autograd gradient of AggregMin is 1 at the argmin entry and 0 elsewhere."""
    name = "AggregMin gradient"
    check_one = partial(_gradient_mismatches, name, _agg_min, _aggmin_grad)
    return _check_cases(name, rng, n_cases, (1, 5), _close_pair, check_one)


def check_pmean_error_gradient(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (eq. 19): the autograd gradient of AggregPMeanError matches its closed form for p in {1, 2, 3}."""
    name = "AggregPMeanError gradient"
    check_one = partial(_pmean_error_gradient_all_p, name)
    return _check_cases(name, rng, n_cases, (1, 5), _all_ones, check_one)


def check_duality_luk(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (Section 7): d AndLuk(1-a, 1-b)/d(1-a) == d OrLuk(a, b)/da."""
    name = "duality Lukasiewicz"
    check_one = partial(_duality_mismatches, name, _and_luk, _or_luk)
    return _check_cases(name, rng, n_cases, (2, 2), _pair_sum_tie, check_one)


def check_duality_minmax(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (Section 7): d AndMin(1-a, 1-b)/d(1-a) == d OrMax(a, b)/da."""
    name = "duality min/max"
    check_one = partial(_duality_mismatches, name, _and_min, _or_max)
    return _check_cases(name, rng, n_cases, (2, 2), _close_pair, check_one)


def check_duality_product(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law (Section 7): d AndProd(1-a, 1-b)/d(1-a) == d OrProbSum(a, b)/da."""
    name = "duality product/probsum"
    check_one = partial(_duality_mismatches, name, _product, _probsum)
    return _check_cases(name, rng, n_cases, (2, 2), _never_tie, check_one)


def check_godel_invariance_andmin(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: min(a^k, b^k) == min(a, b)^k for k in {2, 3} (order-invariance of Godel AndMin)."""
    name = "Godel invariance AndMin"
    check_one = partial(_invariance_mismatches, name, _and_min)
    return _check_cases(name, rng, n_cases, (2, 2), _close_pair, check_one)


def check_godel_invariance_ormax(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: max(a^k, b^k) == max(a, b)^k for k in {2, 3} (order-invariance of Godel OrMax)."""
    name = "Godel invariance OrMax"
    check_one = partial(_invariance_mismatches, name, _or_max)
    return _check_cases(name, rng, n_cases, (2, 2), _close_pair, check_one)


def check_godel_invariance_implies(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: ImpliesGodel(a^k, b^k) == ImpliesGodel(a, b)^k for k in {2, 3} (order-invariance)."""
    name = "Godel invariance ImpliesGodel"
    check_one = partial(_invariance_mismatches, name, _implies_godel)
    return _check_cases(name, rng, n_cases, (2, 2), _close_pair, check_one)


def check_godel_invariance_not(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: NotGodel(a^k) == NotGodel(a)^k for k in {2, 3} (order-invariance of Godel negation)."""
    name = "Godel invariance NotGodel"
    check_one = partial(_invariance_mismatches, name, _not_godel)
    return _check_cases(name, rng, n_cases, (1, 1), _never_tie, check_one)


def check_pmean_p1_is_mean(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: AggregPMean with p = 1 equals AggregMean."""
    name = "AggregPMean p=1 equals AggregMean"
    check_one = partial(_equal_mismatches, name, _pmean_op(1), _mean_op)
    return _check_cases(name, rng, n_cases, (1, 5), _never_tie, check_one)


def check_pmean_error_p1_is_mean(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: AggregPMeanError with p = 1 equals AggregMean."""
    name = "AggregPMeanError p=1 equals AggregMean"
    check_one = partial(_equal_mismatches, name, _pmean_error_op(1), _mean_op)
    return _check_cases(name, rng, n_cases, (1, 5), _never_tie, check_one)


def check_pmean_error_monotone(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: AggregPMeanError is non-increasing in p over p in {1, 2, 3, 6}."""
    name = "AggregPMeanError non-increasing in p"
    check_one = partial(_pmean_error_monotone, name)
    return _check_cases(name, rng, n_cases, (1, 5), _never_tie, check_one)


def check_pmean_monotone(rng: random.Random, n_cases: int) -> CheckOutcome:
    """Law: AggregPMean is non-decreasing in p over p in {1, 2, 3, 6}."""
    name = "AggregPMean non-decreasing in p"
    check_one = partial(_pmean_monotone, name)
    return _check_cases(name, rng, n_cases, (1, 5), _never_tie, check_one)


CHECKS = (
    check_andluk_gradient,
    check_orluk_gradient,
    check_andmin_gradient,
    check_ormax_gradient,
    check_aggregmin_gradient,
    check_pmean_error_gradient,
    check_duality_luk,
    check_duality_minmax,
    check_duality_product,
    check_godel_invariance_andmin,
    check_godel_invariance_ormax,
    check_godel_invariance_implies,
    check_godel_invariance_not,
    check_pmean_p1_is_mean,
    check_pmean_error_p1_is_mean,
    check_pmean_error_monotone,
    check_pmean_monotone,
)


def run_extras(n_cases: int = 500, seed: int = 0) -> ExtraResult:
    """Run every extra check on n_cases seeded cases; return all violations and per-check compared/skipped counts."""
    rng = random.Random(seed)
    outcomes = [check(rng, n_cases) for check in CHECKS]
    return ExtraResult(
        violations=[v for outcome in outcomes for v in outcome.violations],
        case_counts={o.name: o.compared for o in outcomes},
        skip_counts={o.name: o.skipped for o in outcomes},
    )
