"""Manual check of F1, F2, F3 in LTNtorch (commit d1bd981).

Each test builds the smallest LTN program that shows the bug and prints
what LTN returns. No fuzzer involved: this is plain Python.

Run:  python check_f1_f2_f3.py
"""
import warnings

warnings.filterwarnings("ignore")  # hide LTN's docstring SyntaxWarnings

import torch
import ltn

# A simple predicate: score = sigmoid(sum of features), always in [0, 1].
P = ltn.Predicate(func=lambda a: torch.sigmoid(a.sum(dim=1)))


def run(label, fn):
    try:
        out = fn()
        print(f"  {label:42} -> {out.value.item() if hasattr(out, 'value') else out}")
    except Exception as e:
        print(f"  {label:42} -> {type(e).__name__}: {e}")


# ---------------------------------------------------------------- F1 ----
print("F1  Empty domain: 'for all x, P(x)' when x has ZERO individuals")
print("    Expected (logic): vacuously true = 1.0, or a clear error. Should be consistent.")
x_empty = ltn.Variable("x", torch.empty(0, 2))   # 0 individuals, 2 features each

for name, agg in [("Min", ltn.fuzzy_ops.AggregMin()),
                  ("Mean", ltn.fuzzy_ops.AggregMean()),
                  ("pMean (Exists-style)", ltn.fuzzy_ops.AggregPMean()),
                  ("pMeanError", ltn.fuzzy_ops.AggregPMeanError())]:
    Forall = ltn.Quantifier(agg, quantifier="f")
    run(f"Forall with {name}", lambda: Forall(x_empty, P(x_empty)))

# Same idea through a guard: x has 3 individuals but the guard selects none of them.
x3 = ltn.Variable("x", torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
Forall_min = ltn.Quantifier(ltn.fuzzy_ops.AggregMin(), quantifier="f")
Forall_mean = ltn.Quantifier(ltn.fuzzy_ops.AggregMean(), quantifier="f")
never = lambda v: v.value.sum(dim=1) > 100        # selects no individual
run("Forall Min, guard selects nobody", lambda: Forall_min(x3, P(x3), cond_vars=[x3], cond_fn=never))
run("Forall Mean, guard selects nobody", lambda: Forall_mean(x3, P(x3), cond_vars=[x3], cond_fn=never))
print()

# ---------------------------------------------------------------- F2 ----
print("F2  p = 0 accepted by the constructor, crashes when used")
print("    Expected: constructor rejects p = 0 with a clear ValueError.")
for name, cls in [("AggregPMean", ltn.fuzzy_ops.AggregPMean),
                  ("AggregPMeanError", ltn.fuzzy_ops.AggregPMeanError)]:
    run(f"{name}(p=0) constructor", lambda: cls(p=0) and "accepted (no error)")
    Q = ltn.Quantifier(cls(p=0), quantifier="f")
    run(f"Forall with {name}(p=0)", lambda: Q(x3, P(x3)))
print()

# ---------------------------------------------------------------- F3 ----
print("F3  Quantifying a variable that does not appear in the formula")
print("    Expected (logic): 'for all y, P(x)' = P(x), since y is not used.")
y = ltn.Variable("y", torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier="f")
run("Forall y, P(x)   [x has 3 individuals]", lambda: Forall(y, P(x3)).value.tolist())
run("Forall [x, y], P(x)", lambda: Forall([x3, y], P(x3)))
