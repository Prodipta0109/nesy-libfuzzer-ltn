"""LTN-only extension checks (spec Task 11, van Krieken et al. Section 6 and Section 7).
Each check draws random cases in (0, 1), evaluates an ltn.fuzzy_ops operator (stable=False where it exists) and
compares it with a closed form or a law. Mismatches are returned as ExtraViolation records.
"""
import itertools
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Sequence

import torch
from ltn import fuzzy_ops as fz

GRAD_TOL = 1e-10
VALUE_TOL = 1e-12
TIE_GAP = 1e-9
PS = (1, 2, 3, 6)
CASE_COUNTS: Dict[str, int] = {}
SKIP_COUNTS: Dict[str, int] = {}


@dataclass
class ExtraViolation:
    """A failed extra check: its name, the inputs, the value got and the value the law expects."""

    name: str
    inputs: list
    got: float
    expected: float


# ----- case plumbing ---------------------------------------------------------------------------------------------


def _tally(name: str, compared: bool) -> None:
    """Count one case of check `name` as compared, or as skipped for a near-tie."""
    counts = CASE_COUNTS if compared else SKIP_COUNTS
    counts[name] = counts.get(name, 0) + 1


def _separated(values: Sequence[float]) -> bool:
    """Law: every two values are at least TIE_GAP apart (closer pairs are ties and are skipped)."""
    return all(abs(a - b) >= TIE_GAP for a, b in itertools.combinations(values, 2))


def _draw_values(rng: random.Random, low: int, high: int) -> List[float]:
    """A vector of random values in (0, 1) whose length is between low and high."""
    size = rng.randint(low, high)
    return [rng.random() for _ in range(size)]


def _check_cases(
    name: str,
    rng: random.Random,
    n_cases: int,
    low: int,
    high: int,
    check_one: Callable[[List[float]], List[ExtraViolation]],
) -> List[ExtraViolation]:
    """Law-free driver: run check_one on n_cases random vectors, skipping and counting near-ties."""
    found: List[ExtraViolation] = []
    for _ in range(n_cases):
        values = _draw_values(rng, low, high)
        compared = _separated(values)
        _tally(name, compared)
        if compared:
            found.extend(check_one(values))
    return found


# ----- tensor evaluation -----------------------------------------------------------------------------------------


def _vector_gradient(op: Callable, values: Sequence[float]) -> List[float]:
    """Autograd gradient, in float64, of the scalar op(xs) with respect to each entry of xs."""
    xs = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(op(xs), xs)
    return grad.tolist()


def _vector_value(op: Callable, values: Sequence[float]) -> float:
    """Value of op(xs) for the float64 tensor xs built from `values`."""
    return float(op(torch.tensor(values, dtype=torch.float64)))


# ----- operators under test (all with stable=False where the operator has that flag) ------------------------------


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
    """Law: op(f(x)) == f(op(x)) for the squaring map f(t) = t**2 (Godel order-invariance)."""
    got = _vector_value(op, [v * v for v in values])
    expected = _vector_value(op, values) ** 2
    if abs(got - expected) > VALUE_TOL:
        return [ExtraViolation(name, list(values), got, expected)]
    return []


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


def check_andluk_gradient(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (eq. 33): the autograd gradient of AndLuk equals 1 if a + b > 1 else 0."""
    name = "AndLuk gradient"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_gradient_mismatches, name, _and_luk, _andluk_grad))


def check_orluk_gradient(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (eq. 33 dual): the autograd gradient of OrLuk equals 1 if a + b < 1 else 0."""
    name = "OrLuk gradient"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_gradient_mismatches, name, _or_luk, _orluk_grad))


def check_andmin_gradient(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (eq. 32): the autograd gradient of AndMin is 1 for the smaller argument and 0 for the other."""
    name = "AndMin gradient"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_gradient_mismatches, name, _and_min, _andmin_grad))


def check_ormax_gradient(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (eq. 32 dual): the autograd gradient of OrMax is 1 for the larger argument and 0 for the other."""
    name = "OrMax gradient"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_gradient_mismatches, name, _or_max, _ormax_grad))


def check_aggregmin_gradient(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (eq. 14): the autograd gradient of AggregMin is 1 at the argmin entry and 0 elsewhere."""
    name = "AggregMin gradient"
    return _check_cases(name, rng, n_cases, 2, 5, partial(_gradient_mismatches, name, _agg_min, _aggmin_grad))


def check_pmean_error_gradient(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (eq. 19): the autograd gradient of AggregPMeanError matches its closed form for p in {1, 2, 3}."""
    name = "AggregPMeanError gradient"
    return _check_cases(name, rng, n_cases, 2, 5, partial(_pmean_error_gradient_all_p, name))


def check_duality_luk(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (Section 7): d AndLuk(1-a, 1-b)/d(1-a) == d OrLuk(a, b)/da."""
    name = "duality Lukasiewicz"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_duality_mismatches, name, _and_luk, _or_luk))


def check_duality_minmax(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (Section 7): d AndMin(1-a, 1-b)/d(1-a) == d OrMax(a, b)/da."""
    name = "duality min/max"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_duality_mismatches, name, _and_min, _or_max))


def check_duality_product(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law (Section 7): d AndProd(1-a, 1-b)/d(1-a) == d OrProbSum(a, b)/da."""
    name = "duality product/probsum"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_duality_mismatches, name, _product, _probsum))


def check_godel_invariance_andmin(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: min(a**2, b**2) == min(a, b)**2 (order-invariance of Godel AndMin under t -> t**2)."""
    name = "Godel invariance AndMin"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_invariance_mismatches, name, _and_min))


def check_godel_invariance_ormax(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: max(a**2, b**2) == max(a, b)**2 (order-invariance of Godel OrMax under t -> t**2)."""
    name = "Godel invariance OrMax"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_invariance_mismatches, name, _or_max))


def check_godel_invariance_implies(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: ImpliesGodel(a**2, b**2) == ImpliesGodel(a, b)**2 (order-invariance under t -> t**2)."""
    name = "Godel invariance ImpliesGodel"
    return _check_cases(name, rng, n_cases, 2, 2, partial(_invariance_mismatches, name, _implies_godel))


def check_godel_invariance_not(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: NotGodel(a**2) == NotGodel(a)**2 (order-invariance of Godel negation under t -> t**2)."""
    name = "Godel invariance NotGodel"
    return _check_cases(name, rng, n_cases, 1, 1, partial(_invariance_mismatches, name, _not_godel))


def check_pmean_p1_is_mean(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: AggregPMean with p = 1 equals AggregMean."""
    name = "AggregPMean p=1 equals AggregMean"
    return _check_cases(name, rng, n_cases, 2, 5, partial(_equal_mismatches, name, _pmean_op(1), _mean_op))


def check_pmean_error_p1_is_mean(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: AggregPMeanError with p = 1 equals AggregMean."""
    name = "AggregPMeanError p=1 equals AggregMean"
    return _check_cases(name, rng, n_cases, 2, 5, partial(_equal_mismatches, name, _pmean_error_op(1), _mean_op))


def check_pmean_error_monotone(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: AggregPMeanError is non-increasing in p over p in {1, 2, 3, 6}."""
    name = "AggregPMeanError non-increasing in p"
    return _check_cases(name, rng, n_cases, 2, 5, partial(_pmean_error_monotone, name))


def check_pmean_monotone(rng: random.Random, n_cases: int) -> List[ExtraViolation]:
    """Law: AggregPMean is non-decreasing in p over p in {1, 2, 3, 6}."""
    name = "AggregPMean non-decreasing in p"
    return _check_cases(name, rng, n_cases, 2, 5, partial(_pmean_monotone, name))


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


def run_extras(n_cases: int = 500, seed: int = 0) -> List[ExtraViolation]:
    """Run every extra check on n_cases seeded random cases and return all violations; counts go to CASE_COUNTS."""
    CASE_COUNTS.clear()
    SKIP_COUNTS.clear()
    rng = random.Random(seed)
    found: List[ExtraViolation] = []
    for check in CHECKS:
        found.extend(check(rng, n_cases))
    return found
