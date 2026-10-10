"""Closed-form reference semantics for LTNtorch's fuzzy operators (float64, plain Python/NumPy).

Every formula is the one printed in the docstring of the corresponding class in ltn/fuzzy_ops.py. The "stable" variants
apply the documented input projections pi_0(x) = (1-eps) x + eps and pi_1(x) = (1-eps) x with eps = 1e-4
(frameworks/LTNtorch/docsrc/source/stableconf.rst).
"""
import numpy as np

EPS = 1e-4


def pi0(x):
    return (1 - EPS) * x + EPS


def pi1(x):
    return (1 - EPS) * x


# ----------------------------------------------------------------- connectives: name -> (arity, stable-aware fn)
def not_standard(x):
    return 1.0 - x


def not_godel(x):
    return np.where(x == 0.0, 1.0, 0.0)


def and_min(x, y):
    return np.minimum(x, y)


def and_prod(x, y, stable=True):
    if stable:
        x, y = pi0(x), pi0(y)
    return x * y


def and_luk(x, y):
    return np.maximum(x + y - 1.0, 0.0)


def or_max(x, y):
    return np.maximum(x, y)


def or_probsum(x, y, stable=True):
    if stable:
        x, y = pi1(x), pi1(y)
    return x + y - x * y


def or_luk(x, y):
    return np.minimum(x + y, 1.0)


def imp_kd(x, y):
    return np.maximum(1.0 - x, y)


def imp_godel(x, y):
    return np.where(x <= y, 1.0, y)


def imp_reichenbach(x, y, stable=True):
    if stable:
        x, y = pi0(x), pi1(y)
    return 1.0 - x + x * y


def imp_goguen(x, y, stable=True):
    if stable:
        x = pi0(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x <= y, 1.0, y / x)


def imp_luk(x, y):
    return np.minimum(1.0 - x + y, 1.0)


# name -> (ltn class name, ctor kwargs, oracle(x, y) or oracle(x), arity)
def connective_table():
    t = {
        "NotStandard": ("NotStandard", {}, not_standard, 1),
        "NotGodel": ("NotGodel", {}, not_godel, 1),
        "AndMin": ("AndMin", {}, and_min, 2),
        "AndLuk": ("AndLuk", {}, and_luk, 2),
        "OrMax": ("OrMax", {}, or_max, 2),
        "OrLuk": ("OrLuk", {}, or_luk, 2),
        "ImpliesKleeneDienes": ("ImpliesKleeneDienes", {}, imp_kd, 2),
        "ImpliesGodel": ("ImpliesGodel", {}, imp_godel, 2),
        "ImpliesLuk": ("ImpliesLuk", {}, imp_luk, 2),
    }
    for stable in (True, False):
        s = "S" if stable else "U"
        t["AndProd" + s] = ("AndProd", {"stable": stable}, lambda x, y, st=stable: and_prod(x, y, st), 2)
        t["OrProbSum" + s] = ("OrProbSum", {"stable": stable}, lambda x, y, st=stable: or_probsum(x, y, st), 2)
        t["ImpliesReichenbach" + s] = ("ImpliesReichenbach", {"stable": stable},
                                       lambda x, y, st=stable: imp_reichenbach(x, y, st), 2)
        t["ImpliesGoguen" + s] = ("ImpliesGoguen", {"stable": stable}, lambda x, y, st=stable: imp_goguen(x, y, st), 2)
    return t


# ----------------------------------------------------------------- aggregators over a 1-D array (optionally masked)
def _select(xs, mask):
    xs = np.asarray(xs, dtype=np.float64)
    return xs if mask is None else xs[np.asarray(mask, dtype=bool)]


def agg_min(xs, mask=None, **_):
    s = _select(xs, mask)
    return 1.0 if s.size == 0 else float(s.min())


def agg_mean(xs, mask=None, **_):
    s = _select(xs, mask)
    return float("nan") if s.size == 0 else float(s.mean())


def agg_pmean(xs, mask=None, p=2, stable=True):
    s = _select(xs, mask)
    if s.size == 0:
        return float("nan")
    if stable:
        s = pi0(s)
    return float(np.mean(s ** p) ** (1.0 / p))


def agg_pmean_error(xs, mask=None, p=2, stable=True):
    s = _select(xs, mask)
    if s.size == 0:
        return float("nan")
    if stable:
        s = pi1(s)
    return float(1.0 - np.mean((1.0 - s) ** p) ** (1.0 / p))


def aggregator_table():
    return {
        "AggregMin": ("AggregMin", {}, agg_min),
        "AggregMean": ("AggregMean", {}, agg_mean),
        "AggregPMeanS": ("AggregPMean", {"stable": True, "p": 2}, lambda xs, mask=None, p=2: agg_pmean(xs, mask, p, True)),
        "AggregPMeanU": ("AggregPMean", {"stable": False, "p": 2}, lambda xs, mask=None, p=2: agg_pmean(xs, mask, p, False)),
        "AggregPMeanErrorS": ("AggregPMeanError", {"stable": True, "p": 2},
                              lambda xs, mask=None, p=2: agg_pmean_error(xs, mask, p, True)),
        "AggregPMeanErrorU": ("AggregPMeanError", {"stable": False, "p": 2},
                              lambda xs, mask=None, p=2: agg_pmean_error(xs, mask, p, False)),
    }


def agg_axes(fn, arr, dims, keepdim=False, mask=None):
    """Apply a 1-D oracle aggregator over the axes `dims` of an N-D array (optionally masked), like torch's dim/keepdim."""
    arr = np.asarray(arr, dtype=np.float64)
    nd = arr.ndim
    dims = tuple(sorted(d % nd for d in (dims if isinstance(dims, (list, tuple)) else [dims])))
    rest = [a for a in range(nd) if a not in dims]
    perm = rest + list(dims)
    flat = np.transpose(arr, perm).reshape([arr.shape[a] for a in rest] + [-1])
    mflat = None
    if mask is not None:
        mflat = np.transpose(np.asarray(mask, dtype=bool), perm).reshape([arr.shape[a] for a in rest] + [-1])
    out = np.empty(flat.shape[:-1])
    for idx in np.ndindex(*flat.shape[:-1]):
        out[idx] = fn(flat[idx], None if mflat is None else mflat[idx])
    if keepdim:
        shape = [1 if a in dims else arr.shape[a] for a in range(nd)]
        return out.reshape(shape)
    return out
