"""Do the oracles detect wrong answers?  Plant a bug in LNN on purpose (in memory only) and see if the harness notices.
  python3 mutation_check.py
"""
import os, sys, random, warnings
warnings.filterwarnings("ignore")
os.environ["LNN_FINDINGS"] = "/tmp/lnn_mut"
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("h", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lnn_fuzz.py"))
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
import atheris, torch
from lnn.neural.methods.lukasiewicztransparent import LukasiewiczTransparent as L

orig_and, orig_or, orig_imp = L._and_upward, L._or_upward, L._implies_upward
MUTANTS = {
    "And: upward result lowered by 0.05": ("_and_upward", lambda self, b: orig_and(self, b) - 0.05),
    "Or: upward result raised by 0.1": ("_or_upward", lambda self, b: orig_or(self, b) + 0.1),
    "Implies: upward result scaled by 0.9": ("_implies_upward", lambda self, b: orig_imp(self, b) * 0.9),
}
FAMILIES = [h.t_upward, h.t_classical, h.t_invariants, h.t_weights]

def run(n=150):
    rng = random.Random(7); caught = {f.__name__: 0 for f in FAMILIES}
    for f in FAMILIES:
        for _ in range(n):
            data = bytes(rng.getrandbits(8) for _ in range(160))
            try: f(atheris.FuzzedDataProvider(data))
            except ModuleNotFoundError: pass          # Goedel module missing (bug L3), not a wrong value
            except Exception: caught[f.__name__] += 1
    return caught

base = run(); print("unmodified LNN (hits are bug L5, formulas built twice):", base)
for name, (attr, fn) in MUTANTS.items():
    setattr(L, attr, fn)
    try: print("mutant '%s':" % name, run())
    finally: setattr(L, attr, {"_and_upward": orig_and, "_or_upward": orig_or, "_implies_upward": orig_imp}[attr])
