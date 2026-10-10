#!/usr/bin/env python3
"""
Minimal, harness-free reproducers for the LTNtorch findings.
Each case uses only the public LTN API, so it can go into a bug report as is.
Run:  python confirm_findings.py
"""
import sys
import traceback

import torch

import os  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ltn_torch"))
import ltn  # noqa: E402
import ltn.fuzzy_ops as fo  # noqa: E402

ltn.device = torch.device("cpu")
P = ltn.Predicate(func=lambda x: torch.sigmoid(x.sum(dim=1)))
P2 = ltn.Predicate(func=lambda x, y: torch.sigmoid(x.sum(dim=1) + y.sum(dim=1)))


def case(name, fn):
    print(f"\n### {name}")
    try:
        out = fn()
        print("result:", out)
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"raised {type(e).__name__}: {e}")
        print(f"  at {tb.filename.split('/ltn/')[-1]}:{tb.lineno}  `{tb.line}`")


# ---- F1: empty domain (variable with 0 individuals) -----------------------
empty = lambda: ltn.Variable("x", torch.empty(0, 2))
for agg in [fo.AggregMin(), fo.AggregMean(), fo.AggregPMean(p=2), fo.AggregPMeanError(p=2)]:
    for q in ["f", "e"]:
        case(f"F1 empty domain: Quantifier({agg!r}, '{q}')(x, P(x))",
             lambda agg=agg, q=q: ltn.Quantifier(agg, q)(x := empty(), P(x)).value.item())

# Same semantics through guarded quantification: LTN *does* define it there
x = ltn.Variable("x", torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
case("F1 contrast: guarded Forall whose mask selects nobody (LTN maps NaN -> 1.0)",
     lambda: ltn.Quantifier(fo.AggregPMeanError(), "f")(
         x, P(x), cond_vars=[x], cond_fn=lambda v: v.value.sum(dim=1) < -100).value.item())
case("F1 contrast: guarded Forall + AggregMin, mask selects nobody",
     lambda: ltn.Quantifier(fo.AggregMin(), "f")(
         x, P(x), cond_vars=[x], cond_fn=lambda v: v.value.sum(dim=1) < -100).value.item())

# ---- F2: p = 0 is not validated --------------------------------------------
for cls in [fo.AggregPMean, fo.AggregPMeanError]:
    case(f"F2 {cls.__name__}(p=0) unguarded",
         lambda cls=cls: ltn.Quantifier(cls(p=0), "f")(x, P(x)).value.item())
    case(f"F2 {cls.__name__}(p=0) guarded",
         lambda cls=cls: ltn.Quantifier(cls(p=0), "f")(
             x, P(x), cond_vars=[x], cond_fn=lambda v: v.value.sum(dim=1) < 10).value.item())

# ---- F3: quantifying over a variable that is not free in the formula --------
y = ltn.Variable("y", torch.tensor([[0.5, 0.5], [1.5, 1.5], [3.0, 3.0]]))
case("F3 vacuous quantification: Forall(y, P(x))",
     lambda: ltn.Quantifier(fo.AggregPMeanError(), "f")(y, P(x)).value.tolist())
case("F3 vacuous quantification, guarded: Forall(y, P(x), cond on y)",
     lambda: ltn.Quantifier(fo.AggregPMeanError(), "f")(
         y, P(x), cond_vars=[y], cond_fn=lambda v: v.value.sum(dim=1) < 2).value.tolist())
case("F3 vacuous quantification, guarded on a different variable: Forall(y, P(x), cond on x)",
     lambda: ltn.Quantifier(fo.AggregPMeanError(), "f")(
         y, P(x), cond_vars=[x], cond_fn=lambda v: v.value.sum(dim=1) < 3).value.tolist())
