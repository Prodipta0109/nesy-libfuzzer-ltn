"""Reproduce every LTN bug found in this project, without the fuzzer.

LTNtorch 1.0.2, commit d1bd981.
Run from ~/nesy_libfuzzer_ltn:   PYTHONPATH=ltn_torch python3 check_all_bugs.py
"""
import warnings

warnings.filterwarnings("ignore")  # hide LTN's docstring SyntaxWarnings

import torch
import ltn

fo = ltn.fuzzy_ops
P = ltn.Predicate(func=lambda a: torch.sigmoid(a.sum(dim=1)))   # always in [0, 1]
x3 = ltn.Variable("x", torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))


def show(label, fn):
    try:
        out = fn()
        if hasattr(out, "value"):
            out = out.value
        if isinstance(out, torch.Tensor) and out.numel() == 1:
            out = out.item()
        print(f"  {label:48} -> {out}")
    except Exception as e:
        print(f"  {label:48} -> {type(e).__name__}: {e}")


print("=== NEW bugs (found by the libFuzzer harness) ===\n")

print("F1  Empty domain: 'for all x, P(x)' with 0 individuals")
print("    expected: one consistent answer (logically 1.0)")
x0 = ltn.Variable("x", torch.empty(0, 2))
for name, agg in [("Min", fo.AggregMin()), ("Mean", fo.AggregMean()),
                  ("pMean", fo.AggregPMean()), ("pMeanError", fo.AggregPMeanError())]:
    Q = ltn.Quantifier(agg, "f")
    show(f"Forall, {name}", lambda: Q(x0, P(x0)))
never = lambda v: v.value.sum(dim=1) > 100
show("Forall, Min, guard selects nobody",
     lambda: ltn.Quantifier(fo.AggregMin(), "f")(x3, P(x3), cond_vars=[x3], cond_fn=never))
print()

print("F2  p = 0 accepted by the constructor, crashes when used")
print("    expected: ValueError from the constructor")
show("AggregPMean(p=0) in a Forall",
     lambda: ltn.Quantifier(fo.AggregPMean(p=0), "f")(x3, P(x3)))
show("AggregPMeanError(p=0) in a Forall",
     lambda: ltn.Quantifier(fo.AggregPMeanError(p=0), "f")(x3, P(x3)))
print()

print("F3  Quantifying a variable that is not in the formula")
print("    expected: same as P(x) (y is irrelevant), or a clear error")
y = ltn.Variable("y", torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
show("Forall y, P(x)", lambda: ltn.Quantifier(fo.AggregPMeanError(), "f")(y, P(x3)).value.tolist())
print()

print("=== REDISCOVERED bugs (first found by neys_fuzz) ===\n")

print("F24 ImpliesGoguen(stable=True): false -> false")
print("    expected: 1.0 (false implies false is true)")
z = torch.tensor([0.0])
show("ImpliesGoguen(stable=True)(0, 0)", lambda: fo.ImpliesGoguen(stable=True)(z, z))
show("ImpliesGoguen(stable=False)(0, 0)", lambda: fo.ImpliesGoguen(stable=False)(z, z))
print()

print("F25 Quantifier([]) returns a scalar that still claims free variables")
print("    expected: shape () with no free variables, or an error")
q = ltn.Quantifier(fo.AggregPMeanError(), "f")([], P(x3))
print(f"  {'Forall [], P(x)':48} -> shape {tuple(q.value.shape)}, free variables {q.free_vars}")
print()

print("F26 dtype changes")
print("    expected: output dtype = input dtype (float64 in, float64 out; float32 in, float32 out)")
x64 = torch.tensor([0.3, 0.7], dtype=torch.float64)
show("NotGodel(float64 input).dtype", lambda: fo.NotGodel()(x64).dtype)
show("AggregMin(float32 input, mask).dtype",
     lambda: fo.AggregMin()(torch.rand(3), dim=0, mask=torch.tensor([True, False, True])).dtype)
show("guarded Forall (float32 data).dtype",
     lambda: ltn.Quantifier(fo.AggregPMeanError(), "f")(
         x3, P(x3), cond_vars=[x3], cond_fn=lambda v: v.value.sum(dim=1) > -1).value.dtype)
print()

print("F27 NaN gradient with unstable operators (low severity: stable=True is the documented fix)")
print("    expected: finite gradient")
for stable in (False, True):
    xs = torch.zeros(3, requires_grad=True)
    fo.AggregPMean(p=2, stable=stable)(xs, dim=0).backward()
    print(f"  {'AggregPMean(p=2, stable=' + str(stable) + ') at 0, 0, 0':48} -> grad {xs.grad.tolist()}")
print()

print("=== Found by code reading, NOT by the fuzzer ===\n")

print("F28 SatAgg given a non-scalar plain tensor")
print("    expected: LTN's own ValueError ('... containing scalars ...')")
show("SatAgg(tensor(0.5), tensor([0.3, 0.7]))",
     lambda: fo.SatAgg()(torch.tensor(0.5), torch.tensor([0.3, 0.7])))
