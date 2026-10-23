"""
Sanity check of the MRs themselves: plant a bug in one operator ("mutant") and check that
at least one MR notices it. Run:  python selftest.py
"""
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ltn_torch"))
import mrs

ONE_OR_ZERO = lambda c: torch.where(c, 1.0, 0.0)

# name -> (table holding the operator, key, buggy replacement)
MUTANTS = {
    "AndLuk forgets the -1":        (mrs.AND, "luk", lambda x, y: torch.clamp(x + y, 0, 1)),
    "AndProd is x*x":               (mrs.AND, "prod", lambda x, y: x * x),
    "AndMin is max":                (mrs.AND, "min", lambda x, y: torch.maximum(x, y)),
    "OrMax is min":                 (mrs.OR, "max", lambda x, y: torch.minimum(x, y)),
    "OrProbSum without -x*y":       (mrs.OR, "probsum", lambda x, y: torch.clamp(x + y, 0, 1)),
    "ImpliesGodel returns 0":       (mrs.IMP, "godel", lambda x, y: ONE_OR_ZERO(x <= y)),
    "ImpliesLuk is 1-x+y (no min)": (mrs.IMP, "luk", lambda x, y: 1 - x + y),
    "AggregMean ignores last":      (mrs.AGG, "mean", lambda xs: xs[:-1].mean()),
}

torch.manual_seed(0)
inputs = [(torch.rand(4), torch.rand(4), torch.rand(4)) for _ in range(300)]


def violated(mr, x, y, z):
    try:
        m = mr(x, y, z)
    except Exception:
        return True
    return m != m or m > mrs.TOL


all_caught = True
for name, (table, key, bug) in MUTANTS.items():
    good, table[key] = table[key], bug
    caught = [mr.__name__ for mr in mrs.MRS if any(violated(mr, *i) for i in inputs)]
    table[key] = good
    print(f"{'CAUGHT' if caught else 'MISSED':7s}{name:32s}{', '.join(caught[:4])}")
    all_caught &= bool(caught)

clean = [mr.__name__ for mr in mrs.MRS if any(violated(mr, *i) for i in inputs)]
print(f"\nunmodified LTN violates: {clean or 'nothing'}")
sys.exit(0 if all_caught else 1)
