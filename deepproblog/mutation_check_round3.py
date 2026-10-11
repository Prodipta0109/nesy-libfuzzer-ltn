"""Do the round-2/3 oracles detect wrong answers?  Plant bugs in DeepProbLog in memory and run the new families.
  python3 mutation_check_round3.py"""
import os, sys, random, warnings
warnings.filterwarnings("ignore")
os.environ["DPL_OUT"] = "/tmp/dpl_mut3"
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpl_round3 as h
import atheris
from deepproblog.semiring.graph_semiring import GraphSemiring as G

MUTANTS = {
    "negate returns 0.99 - a instead of 1 - a": ("negate", lambda self, a: 0.99 - a),
    "plus drops its second argument when it is a plain float": ("plus", lambda self, a, b: a if type(b) is float else a + b),
    "times multiplies by 1.01": ("times", lambda self, a, b: a * b * 1.01),
}
FAMILIES = [h.t_meta_prog, h.t_fo, h.t_fo_ad, h.t_batch]

def run(n=40):
    rng = random.Random(7); caught = {f.__name__: 0 for f in FAMILIES}
    for f in FAMILIES:
        for _ in range(n):
            data = bytes(rng.getrandbits(8) for _ in range(160))
            try: f(atheris.FuzzedDataProvider(data))
            except Exception: caught[f.__name__] += 1
    return caught

print("unmodified DeepProbLog:", run())
for name, (attr, fn) in MUTANTS.items():
    orig = getattr(G, attr); setattr(G, attr, fn)
    try: print(f"mutant '{name}':", run())
    finally: setattr(G, attr, orig)
