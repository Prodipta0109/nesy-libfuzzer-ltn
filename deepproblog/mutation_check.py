"""Do the oracles actually detect wrong answers?  Plant a bug in DeepProbLog on purpose and see whether the
harness notices.  Each mutant is applied in memory only; the installed library is not changed.

  python3 mutation_check.py
"""
import os, sys, random, warnings
warnings.filterwarnings("ignore")
os.environ["DPL_OUT"] = "/tmp/dpl_mut"
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("h", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpl_fuzz.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
import atheris
from deepproblog.semiring.graph_semiring import GraphSemiring as G

MUTANTS = {
    "negate returns 0.99 - a instead of 1 - a": ("negate", lambda self, a: 0.99 - a),
    "plus drops its second argument when it is a plain float": ("plus", lambda self, a, b: a if type(b) is float else a + b),
    "times multiplies by 1.01": ("times", lambda self, a, b: a * b * 1.01),
}
FAMILIES = [h.t_prop, h.t_graph, h.t_neural]


def run(n=60):
    rng = random.Random(7)
    caught = {f.__name__: 0 for f in FAMILIES}
    for f in FAMILIES:
        for _ in range(n):
            data = bytes(rng.getrandbits(8) for _ in range(160))
            try:
                f(atheris.FuzzedDataProvider(data))
            except h.Bug:
                caught[f.__name__] += 1
            except Exception:
                caught[f.__name__] += 1
    return caught


base = run()
print("unmodified DeepProbLog:", base)
for name, (attr, fn) in MUTANTS.items():
    orig = getattr(G, attr)
    setattr(G, attr, fn)
    try:
        print(f"mutant '{name}':", run())
    finally:
        setattr(G, attr, orig)
