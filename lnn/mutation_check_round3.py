"""Do the round-2/3 oracles detect wrong answers?  Plant bugs in LNN in memory and run the new test families.
  python3 mutation_check_round3.py"""
import os, sys, random, warnings
warnings.filterwarnings("ignore")
os.environ["LNN_FINDINGS"] = "/tmp/lnn_mut3"
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lnn_round3 as h
import atheris
from lnn.neural.methods.lukasiewicztransparent import LukasiewiczTransparent as L

orig_and, orig_or, orig_imp = L._and_upward, L._or_upward, L._implies_upward
MUTANTS = {
    "And: upward result lowered by 0.05": ("_and_upward", lambda self, b: orig_and(self, b) - 0.05),
    "Or: upward result raised by 0.1": ("_or_upward", lambda self, b: orig_or(self, b) + 0.1),
    "Implies: upward result scaled by 0.9": ("_implies_upward", lambda self, b: orig_imp(self, b) * 0.9),
}
FAMILIES = [h.t_meta_prop, h.t_meta_model, h.t_contradiction]

def run(n=120):
    rng = random.Random(11); caught = {f.__name__: 0 for f in FAMILIES}
    for f in FAMILIES:
        for _ in range(n):
            data = bytes(rng.getrandbits(8) for _ in range(160))
            try: f(atheris.FuzzedDataProvider(data))
            except Exception: caught[f.__name__] += 1
    return caught

print("unmodified LNN:", run())
for name, (attr, fn) in MUTANTS.items():
    setattr(L, attr, fn)
    try: print("mutant '%s':" % name, run())
    finally: setattr(L, attr, {"_and_upward": orig_and, "_or_upward": orig_or, "_implies_upward": orig_imp}[attr])
