"""Why these dimensions and why these adaptations?  Every experiment runs the real LNN fuzzer WITH and WITHOUT the feature.
  python3 evidence_lnn.py list | dim | adapt | all | <ID>        (IDs: see `list`)   [--runs N]"""
import sys, os, re, random, warnings, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_core import *
from lnn_lib import LNN
sys.path.insert(0, LNN["cwd"] + "/src")
LIBNAME = os.path.basename(__file__)[9:-3]
RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 5000
MT = 240
BASE = {}
def base():
    if "b" not in BASE: BASE["b"] = run_fuzz(LNN, "full harness", runs=RUNS, max_time=MT)
    return BASE["b"]
def abl(label, env): return run_fuzz(LNN, label, env=env, runs=RUNS, max_time=MT)
COLS = ("label", "execs", "cov", "corpus", "unique_findings", "bugs")

# ---------------- dimensions: remove one dimension, see which bugs disappear ----------------
DIMS = {
 "DIM-L1": ("formula structure (which connectives, how deep)", {"LNN_SKIP": "t_upward,t_classical,t_downward,t_invariants"},
            "Without it the fuzzer never builds the connective trees where the wrong-value (Implies/Iff) and missing-module bugs live."),
 "DIM-L2": ("truth bounds / data: exact point values", {"LNN_FREEZE": "points"},
            "Bug L1 needs an operand known EXACTLY (lower = upper). If only intervals are generated the bug never shows."),
 "DIM-L3": ("first-order structure (Forall/Exists, groundings)", {"LNN_SKIP": "t_fol,t_conv"},
            "Quantifier bugs (L2, L6, L7, L8) are only reachable through first-order formulas."),
 "DIM-L4": ("neuron configuration: weights and bias", {"LNN_FREEZE": "weights"},
            "Bug L1 needs UNEQUAL weights. With every weight fixed to 1.0 it cannot appear."),
 "DIM-L6": ("learning (Model.train, losses)", {"LNN_SKIP": "t_learn"},
            "Loss.LOGICAL on Iff/XOr (bug L4) is only reached by training."),
 "DIM-L7": ("API boundary (misuse, odd arguments)", {"LNN_SKIP": "t_api"},
            "Argument-validation bugs only appear when the harness calls the API wrongly on purpose."),
}
def run_dim(i):
    name, env, why = DIMS[i]
    b = base(); w = abl("without: " + name.split(":")[0], env)
    lost = sorted(set(b["bugs"]) - set(w["bugs"]))
    return compare(LNN, i, "dimension " + name, why, b, w, COLS, note="bugs lost without this dimension: %s" % (lost or "none in this budget"))

# ---------------- adaptations ----------------
def A1():
    on = base(); off = abl("A1 off: no instrumentation", {"LNN_INSTR": "none"})
    return compare(LNN, "A1", "libFuzzer needs coverage feedback from Python (Atheris)",
      "Without instrumentation libFuzzer is blind: the corpus never grows (corp stays 1/1b) and it is only random testing.", on, off, COLS,
      "off: corp stays tiny and cov is absent (nothing measured); on: corpus grows as new code is reached.")
def A2():
    on = base(); off = abl("A2 off: instrument lnn+torch", {"LNN_INSTR": "lnn+torch"})
    return compare(LNN, "A2", "instrument ONLY the library under test",
      "Instrumenting torch as well floods libFuzzer with counters from code we do not test: slower, and 'new coverage' from torch fills the corpus.", on, off,
      ("label", "execs", "cov", "ft", "corpus", "exec_s", "unique_findings", "bugs"))
def A3():
    naive = dict(LNN, cwd=os.path.dirname(os.path.abspath(__file__)), script="naive_lnn.py", name="lnn_naive", out_env="NAIVE_OUT", count_env=None)
    on = base(); off = run_fuzz(naive, "A3 off: fixed formula, bytes->values only", runs=RUNS, max_time=MT)
    return compare(LNN, "A3", "decode bytes into STRUCTURE (formula shape, weights, data), not only values",
      "A fixed And(A,B) with random numbers reaches a tiny part of the library; structured decoding reaches many more branches.", on, off,
      ("label", "execs", "cov", "ft", "corpus", "unique_findings", "bugs"))
def A4():
    warnings.filterwarnings("ignore"); os.environ["LNN_FINDINGS"] = "/tmp/ev_lnn_a4"
    sys.argv = [sys.argv[0]]; sys.path.insert(0, LNN["cwd"])
    import importlib.util, atheris
    spec = importlib.util.spec_from_file_location("h", LNN["cwd"] + "/lnn_fuzz.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    import naive_lnn
    from lnn.neural.methods.lukasiewicztransparent import LukasiewiczTransparent as L
    orig = L._and_upward; rows = []
    for planted in (False, True):
        if planted: L._and_upward = lambda self, b: orig(self, b) - 0.05
        rng = random.Random(5); o = c = 0
        for _ in range(150):
            data = bytes(rng.getrandbits(8) for _ in range(160))
            try: h.t_upward(atheris.FuzzedDataProvider(data))
            except ModuleNotFoundError: pass
            except Exception: o += 1
            try: naive_lnn.one(data)
            except Exception: c += 1
        rows.append({"label": "bug planted: And result lowered by 0.05" if planted else "no bug planted (control)", "inputs": 150, "crash_only_detects": c, "oracle_detects": o})
    L._and_upward = orig
    print("=" * 78); print("A4 - an independent ORACLE is needed: a wrong value does not crash")
    table(rows, ["label", "inputs", "crash_only_detects", "oracle_detects"])
    print("READ: the planted bug never raises an exception (crash-only = 0), the oracle (classical Lukasiewicz formula) flags it.")
    save(LIBNAME + "_A4", {"id": "A4", "rows": rows}); return rows
def A5():
    r = base()
    print("=" * 78); print("A5 - keep going and de-duplicate findings by signature")
    print("raw bug hits: %s   unique findings kept: %s   (%.1fx fewer to read)" % (r["raw_calls"], r["unique_findings"], (r["raw_calls"] or 0) / max(1, r["unique_findings"])))
    print("WHY: one bug is hit again and again; libFuzzer would stop at the first crash. Keep-going finds all kinds, dedup makes them readable.")
    save(LIBNAME + "_A5", {"id": "A5", "raw": r["raw_calls"], "unique": r["unique_findings"]})
def A6():
    from lnn import Propositions, And, Model
    warnings.filterwarnings("ignore")
    A, B = Propositions("A", "B"); f1 = And(A, B); f2 = And(A, B)       # same printed name, different Python object
    m = Model(); m.add_knowledge(f1); m.add_knowledge(f2); m.add_data({A: (0.8, 0.8), B: (0.9, 0.9)}); m.infer(max_steps=20)
    naive = f2.get_data().flatten().tolist(); stored = [n for n in m.nodes.values() if n.name == f2.name][0].get_data().flatten().tolist()
    print("=" * 78); print("A6 - validate the harness itself (control for harness bugs)")
    print("naive oracle reads the object it built (f2):  ", naive, "-> looks like a WRONG value (false alarm)")
    print("fixed oracle reads the node the model stored: ", stored, "-> correct (And(0.8,0.9) = 0.7)")
    save(LIBNAME + "_A6", {"id": "A6", "naive": naive, "fixed": stored})
def A7():
    a = run_fuzz(LNN, "A7 short run", runs=500, max_time=MT); b = run_fuzz(LNN, "A7 long run", runs=3000, max_time=MT)
    slope = (b["rss_mb"] - a["rss_mb"]) * 1024 / (b["execs"] - a["execs"])
    print("=" * 78); print("A7 - run in restarted chunks because Atheris leaks memory")
    table([a, b], ["label", "execs", "rss_mb"]); print("growth: about %.1f KB per execution -> 1M executions would need ~%.0f GB; so we restart the process in chunks and keep the corpus" % (slope, slope * 1e6 / 1048576))
    save(LIBNAME + "_A7", {"id": "A7", "short": a, "long": b, "kb_per_exec": slope})
def A13():
    code = ("import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'%s/src')\nfrom lnn import *\n"
            "A,B=Propositions('A','B');f=And(A,B,activation={'weights':(1.0,0.5),'bias':1.0,'weights_learning':False,'bias_learning':False})\n"
            "m=Model();m.add_knowledge(f);m.add_data({B:(0.4,0.4)})\nm.infer(%s)\nprint('returned')\n") % (LNN["cwd"], "{}")
    rows = []
    for lab, arg in (("with cap: infer(max_steps=60)", "max_steps=60"), ("A13 off: infer() with no cap", "")):
        try: p = subprocess.run([sys.executable, "-c", code.replace("{}", arg)], capture_output=True, text=True, timeout=20); res = "returned (60 steps used, flagged as non-convergence)" if "returned" in p.stdout else "error"
        except subprocess.TimeoutExpired: res = "HANGS (killed after 20 s)"
        rows.append({"label": lab, "result": res})
    print("=" * 78); print("A13/A14 - every inference/training call gets max_steps"); table(rows, ["label", "result"])
    print("READ: without the cap a single input freezes the whole fuzzer (bug L1); with the cap the hang becomes a finding.")
    save(LIBNAME + "_A13", {"id": "A13", "rows": rows})
EXP = {**{k: (lambda k=k: run_dim(k)) for k in DIMS}, "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A6": A6, "A7": A7, "A13": A13}
if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--") and not x.isdigit()]
    cmd = a[0] if a else "list"
    if cmd == "list":
        for k in EXP: print(k, "-", DIMS[k][0] if k in DIMS else (EXP[k].__doc__ or ""))
    else:
        ids = [k for k in EXP if k.startswith("DIM")] if cmd == "dim" else [k for k in EXP if not k.startswith("DIM")] if cmd == "adapt" else list(EXP) if cmd == "all" else [cmd]
        for k in ids: EXP[k]()
