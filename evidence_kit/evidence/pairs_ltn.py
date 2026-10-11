"""LTN: for every dimension, a control input and the fuzzer-chosen input that differ ONLY in that dimension.
  python3 pairs_ltn.py"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("LTN_SRC", "/tmp/rv/ltn_torch"))
from evidence_core import run_pairs
import torch, ltn
fo = ltn.fuzzy_ops
P = ltn.Predicate(func=lambda a: torch.sigmoid(a.sum(dim=1)))

def forall_n(n, agg=fo.AggregPMeanError()):
    def f():
        x = ltn.Variable("x", torch.rand(n, 2)); out = ltn.Quantifier(agg, "f")(x, P(x)).value
        assert not torch.isnan(out), "result is NaN for %d individuals" % n
        return "%.3f" % out.item()
    return f
def implies(a, b):
    def f():
        r = fo.ImpliesGoguen(stable=True)(torch.tensor([a]), torch.tensor([b])).item()
        assert a > b or abs(r - 1.0) < 1e-3, "%s -> %s gave %.4f, expected about 1.0" % (a, b, r)
        return r
    return f
def pmean(p):
    def f():
        x = ltn.Variable("x", torch.rand(3, 2)); ltn.Quantifier(fo.AggregPMean(p=p), "f")(x, P(x))
    return f
def grad(z):
    def f():
        xs = torch.full((3,), z, requires_grad=True); fo.AggregPMean(p=2, stable=False)(xs, dim=0).backward()
        assert torch.isfinite(xs.grad).all(), "gradient %s" % xs.grad.tolist()
    return f
def satagg(n):
    def f():
        x = ltn.Variable("x", torch.rand(3, 2)); fs = [ltn.Quantifier(fo.AggregMin(), "f")(x, P(x)) for _ in range(n)]
        fo.SatAgg()(*fs)
    return f
def var_tensor(t):
    def f():
        v = ltn.Variable("s", t); ltn.Predicate(func=lambda a: torch.sigmoid(a.sum(dim=-1)))(v)
    return f
def dtype(dt):
    def f():
        out = fo.NotGodel()(torch.tensor([0.3, 0.7], dtype=dt)); assert out.dtype == dt, "float %s in -> %s out" % (dt, out.dtype)
    return f
PAIRS = [
 ("D1 data values", "F24", "ImpliesGoguen(stable=True) on (0.2, 0.9) -> (0, 0)", implies(0.2, 0.9), implies(0.0, 0.0)),
 ("D2 data shape", "F1", "3 individuals -> 0 individuals", forall_n(3), forall_n(0)),
 ("D3 predicate / dtype", "F26", "float32 input -> float64 input to NotGodel", dtype(torch.float32), dtype(torch.float64)),
 ("D4 formula structure", "F31", "SatAgg of 1 formula -> of no formulas", satagg(1), satagg(0)),
 ("D5 quantifier setup (p)", "F2", "AggregPMean p=2 -> p=0", pmean(2), pmean(0)),
 ("D6 satisfaction + backward", "F27", "gradient of unstable PMean at 0.5 -> at 0", grad(0.5), grad(0.0)),
 ("D7 API boundary", "F33", "Variable from a 2-D tensor -> 0-d scalar tensor", var_tensor(torch.rand(3, 2)), var_tensor(torch.tensor(0.5))),
]
if __name__ == "__main__": run_pairs("PAIRS_ltn", PAIRS)
