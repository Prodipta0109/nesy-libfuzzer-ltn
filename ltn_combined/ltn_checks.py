"""
Oracle layer for the combined LTN fuzzer.

Every fuzzy operator / aggregator used by the fuzzer is wrapped by `checked(op)`. The wrapper calls the real LTN
operator and then judges the call with the oracles ported from the manual fuzzers in neys_fuzz
(branch fuzz-ltn-lnn-probfuzz-adaptation, commit 9512567):

  O1 formula      LTN run in float64 must equal the documented closed formula (ltn_oracle.py, copied unchanged)
  O2 metamorphic  swap operands of commutative connectives; permute along an aggregated axis;
                  flip the `stable` flag (documented bound 3e-4)          [fuzzing/ltn/fuzz_swap|permute|stable.py]
  O3 dtype        output dtype must equal input dtype                      [fuzz/ltn_fuzz.py agg_dtype / F26]
  P  precision    float32 result far from the float64 formula although the float64 run is right
                  (reported separately: a float32 limit, not a formula bug)  [fuzzing/ltn/fuzz_float64.py]

Raising one of the Violation classes below makes libFuzzer treat the input as a failure.
Guidance is NOT changed: libFuzzer still keeps inputs only for new code coverage.
"""
import math

import numpy as np
import torch

import ltn_oracle as R

TOL_FORMULA = 1e-7      # float64 LTN vs float64 formula
TOL_PRECISION = 1e-4    # float32 LTN vs float64 formula
TOL_SWAP = 3e-6         # fuzzing/ltn/fuzz_swap.py
TOL_PERMUTE = 1e-5      # fuzzing/ltn/fuzz_permute.py
TOL_STABLE = 3e-4       # fuzzing/ltn/fuzz_stable.py (documented projection bound)


class Violation(Exception):
    pass


class FormulaViolation(Violation):
    pass


class MetamorphicViolation(Violation):
    pass


class DtypeViolation(Violation):
    pass


class StructureViolation(Violation):
    pass


class PrecisionNote(Violation):
    pass


ENABLED = {"formula", "metamorphic", "dtype", "precision"}

TOL_BOOL = 3e-4         # classical boundary: on {0,1} inputs a connective must follow Boolean logic
                        # (up to the documented projection bound)      [fuzz/ltn_fuzz.py conn_* boundary laws]

BOOLEAN = {
    "NotStandard": lambda x: 1 - x, "NotGodel": lambda x: 1 - x,
    "AndMin": lambda x, y: x * y, "AndProd": lambda x, y: x * y, "AndLuk": lambda x, y: x * y,
    "OrMax": lambda x, y: np.maximum(x, y), "OrProbSum": lambda x, y: np.maximum(x, y),
    "OrLuk": lambda x, y: np.maximum(x, y),
    "ImpliesKleeneDienes": lambda x, y: np.maximum(1 - x, y), "ImpliesGodel": lambda x, y: np.maximum(1 - x, y),
    "ImpliesReichenbach": lambda x, y: np.maximum(1 - x, y), "ImpliesGoguen": lambda x, y: np.maximum(1 - x, y),
    "ImpliesLuk": lambda x, y: np.maximum(1 - x, y),
    "Equiv": lambda x, y: (x == y).astype(float),
}

COMMUTATIVE = {"AndMin", "AndProd", "AndLuk", "OrMax", "OrProbSum", "OrLuk"}


def op_name(op):
    n = type(op).__name__
    if n.startswith("Checked"):
        n = n[len("Checked"):]
    return n


def label(op):
    n = op_name(op)
    if hasattr(op, "p"):
        return f"{n}(p={op.p}, stable={op.stable})"
    if hasattr(op, "stable"):
        return f"{n}(stable={op.stable})"
    if n == "Equiv":
        return f"Equiv({label(op.and_op)}, {label(op.implies_op)})"
    return f"{n}()"


# ------------------------------------------------------------------ reference formulas
def conn_ref(op, *xs):
    """Documented formula for a connective, in float64 numpy."""
    n = op_name(op)
    st = getattr(op, "stable", True)
    a = [np.asarray(x, dtype=np.float64) for x in xs]
    table = {
        "NotStandard": lambda: R.not_standard(a[0]),
        "NotGodel": lambda: R.not_godel(a[0]),
        "AndMin": lambda: R.and_min(*a),
        "AndProd": lambda: R.and_prod(*a, stable=st),
        "AndLuk": lambda: R.and_luk(*a),
        "OrMax": lambda: R.or_max(*a),
        "OrProbSum": lambda: R.or_probsum(*a, stable=st),
        "OrLuk": lambda: R.or_luk(*a),
        "ImpliesKleeneDienes": lambda: R.imp_kd(*a),
        "ImpliesGodel": lambda: R.imp_godel(*a),
        "ImpliesReichenbach": lambda: R.imp_reichenbach(*a, stable=st),
        "ImpliesGoguen": lambda: R.imp_goguen(*a, stable=st),
        "ImpliesLuk": lambda: R.imp_luk(*a),
    }
    if n == "Equiv":
        i1 = conn_ref(op.implies_op, a[0], a[1])
        i2 = conn_ref(op.implies_op, a[1], a[0])
        return conn_ref(op.and_op, i1, i2)
    return table[n]()


def agg_ref(op, xs, dims, keepdim, mask, p=None, stable=None):
    n = op_name(op)
    p = getattr(op, "p", 2) if p is None else p
    st = getattr(op, "stable", True) if stable is None else stable
    fn = {
        "AggregMin": lambda v, m: R.agg_min(v, m),
        "AggregMean": lambda v, m: R.agg_mean(v, m),
        "AggregPMean": lambda v, m: R.agg_pmean(v, m, p, st),
        "AggregPMeanError": lambda v, m: R.agg_pmean_error(v, m, p, st),
    }[n]
    arr = xs.detach().double().numpy()
    if dims is None:
        dims = tuple(range(arr.ndim))
    mk = None if mask is None else mask.detach().numpy()
    return R.agg_axes(fn, arr, dims, keepdim=keepdim, mask=mk)


def _maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        b = b.reshape(a.shape) if a.size == b.size else b
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0.0, np.abs(a - b))
    d = np.where(np.isnan(d), np.inf, d)
    return float(d.max()) if d.size else 0.0


def _finite_unit(t):
    return t.numel() == 0 or bool(torch.isfinite(t).all() and (t >= 0).all() and (t <= 1).all())


# ------------------------------------------------------------------ the checked call
def _check_connective(base, self, args, out):
    name = label(self)
    xs = [a.detach() for a in args]
    if not all(_finite_unit(x) for x in xs):
        return
    if "dtype" in ENABLED and out.dtype != xs[0].dtype:
        raise DtypeViolation(f"{name}: output {out.dtype} for input {xs[0].dtype}")
    x64 = [x.double() for x in xs]
    out64 = base.__call__(self, *x64)
    if "dtype" in ENABLED and out64.dtype != torch.float64:
        raise DtypeViolation(f"{name}: output {out64.dtype} for input torch.float64")
    ref = conn_ref(self, *x64)
    if "formula" in ENABLED:
        d = _maxdiff(out64.detach().numpy(), ref)
        if d > TOL_FORMULA:
            i = int(np.nanargmax(np.abs(np.asarray(out64.detach().numpy(), dtype=float) - ref)))
            pts = tuple(float(x.reshape(-1)[i]) if x.numel() > 1 else float(x) for x in x64)
            raise FormulaViolation(f"{name}: at inputs {pts} LTN gives "
                                   f"{float(out64.reshape(-1)[i])} but the documented formula gives "
                                   f"{float(np.asarray(ref).reshape(-1)[i])}")
    if "formula" in ENABLED:
        a = [x.numpy() for x in x64]
        a = np.broadcast_arrays(*a)
        crisp = np.ones(a[0].shape, dtype=bool)
        for v in a:
            crisp &= (v == 0) | (v == 1)
        if crisp.any():
            want = BOOLEAN[op_name(self)](*a)
            got = np.broadcast_to(out64.detach().numpy(), want.shape)
            diff = np.where(crisp, np.abs(got - want), 0.0)
            if diff.max() > TOL_BOOL:
                i = int(diff.argmax())
                pts = tuple(float(v.reshape(-1)[i]) for v in a)
                raise FormulaViolation(f"{name}: on Boolean inputs {pts} LTN gives {float(got.reshape(-1)[i])}, "
                                       f"classical logic gives {float(want.reshape(-1)[i])}")
    if "metamorphic" in ENABLED:
        if op_name(self) in COMMUTATIVE and len(xs) == 2:
            sw = base.__call__(self, xs[1], xs[0])
            d = _maxdiff(sw.detach().numpy(), out.detach().numpy())
            if d > TOL_SWAP:
                raise MetamorphicViolation(f"{name}: op(x,y) != op(y,x), max diff {d:.3g}")
        if hasattr(self, "stable"):
            flipped = base.__call__(self, *x64, stable=not self.stable)
            d = _maxdiff(flipped.detach().numpy(), out64.detach().numpy())
            if d > TOL_STABLE:
                raise MetamorphicViolation(f"{name}: stable and unstable versions differ by {d:.3g} "
                                           f"(documented bound {TOL_STABLE})")
    if "precision" in ENABLED:
        d = _maxdiff(out.detach().numpy(), ref)
        if d > TOL_PRECISION:
            raise PrecisionNote(f"{name}: float32 result off the formula by {d:.3g} (float64 is right)")


def _agg_args(args, kwargs):
    xs = args[0]
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    mask = kwargs.get("mask")
    if isinstance(dim, int):
        dim = (dim,)
    if dim is not None:
        dim = tuple(dim)
    return xs, dim, keepdim, mask


def _check_aggregator(base, self, args, kwargs, out):
    name = label(self)
    xs, dim, keepdim, mask = _agg_args(args, kwargs)
    p = kwargs.get("p", getattr(self, "p", None))
    xs = xs.detach()
    if not _finite_unit(xs) or xs.numel() == 0 or dim == ():
        return
    if p is not None and not (isinstance(p, (int, float)) and p >= 1 and math.isfinite(p)):
        return  # documented domain is p >= 1; outside it the crash oracle decides
    if "dtype" in ENABLED and out.dtype != xs.dtype:
        raise DtypeViolation(f"{name}: output {out.dtype} for input {xs.dtype}"
                             f"{' (masked)' if mask is not None else ''}")
    kw = dict(kwargs)
    kw.pop("mask", None)
    kw.pop("dim", None)
    kw.pop("keepdim", None)
    out64 = base.__call__(self, xs.double(), dim=dim, keepdim=keepdim, mask=mask, **kw)
    ref = agg_ref(self, xs, dim, keepdim, mask, p=kwargs.get("p"), stable=kwargs.get("stable"))
    if "formula" in ENABLED:
        d = _maxdiff(out64.detach().numpy(), ref)
        if d > TOL_FORMULA:
            raise FormulaViolation(f"{name}: dim={dim} keepdim={keepdim} mask={'yes' if mask is not None else 'no'}: "
                                   f"LTN {np.round(out64.detach().numpy(), 6).tolist()} vs formula "
                                   f"{np.round(ref, 6).tolist()}")
    if "metamorphic" in ENABLED:
        dims = dim if dim is not None else tuple(range(xs.dim()))
        ax = dims[0]
        if xs.shape[ax] > 1:
            perm = torch.arange(xs.shape[ax] - 1, -1, -1)
            pm = None if mask is None else mask.index_select(ax, perm)
            o2 = base.__call__(self, xs.index_select(ax, perm), dim=dim, keepdim=keepdim, mask=pm, **kw)
            d = _maxdiff(o2.detach().numpy(), out.detach().numpy())
            if d > TOL_PERMUTE:
                raise MetamorphicViolation(f"{name}: result changes when individuals are reordered, diff {d:.3g}")
        if hasattr(self, "stable"):
            f2 = base.__call__(self, xs.double(), dim=dim, keepdim=keepdim, mask=mask,
                               **dict(kw, stable=not self.stable))
            d = _maxdiff(f2.detach().numpy(), out64.detach().numpy())
            if d > TOL_STABLE:
                raise MetamorphicViolation(f"{name}: stable and unstable versions differ by {d:.3g} "
                                           f"(documented bound {TOL_STABLE})")
    if "precision" in ENABLED:
        d = _maxdiff(out.detach().numpy(), ref)
        if d > TOL_PRECISION:
            raise PrecisionNote(f"{name}: float32 result off the formula by {d:.3g} (float64 is right)")


_CLASS_CACHE = {}


def checked(op):
    """Return a copy of `op` whose calls are judged by the oracles. It stays an instance of the
    original class, so LTN's own isinstance checks still accept it."""
    base = type(op)
    if base.__name__.startswith("Checked"):
        return op
    if base not in _CLASS_CACHE:
        import ltn.fuzzy_ops as fo
        is_agg = issubclass(base, fo.AggregationOperator)

        def __call__(self, *args, **kwargs):
            out = base.__call__(self, *args, **kwargs)
            with torch.no_grad():
                if is_agg:
                    _check_aggregator(base, self, args, kwargs, out)
                else:
                    _check_connective(base, self, args, out)
            return out

        _CLASS_CACHE[base] = type("Checked" + base.__name__, (base,), {"__call__": __call__,
                                                                          "__repr__": base.__repr__})
    obj = _CLASS_CACHE[base].__new__(_CLASS_CACHE[base])
    obj.__dict__.update(op.__dict__)
    if base.__name__ == "Equiv":
        obj.and_op = checked(op.and_op)
        obj.implies_op = checked(op.implies_op)
    return obj
