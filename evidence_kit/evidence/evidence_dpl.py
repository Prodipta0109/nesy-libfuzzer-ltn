"""Why these dimensions and adaptations?  Each experiment runs the real DeepProbLog fuzzer WITH and WITHOUT the feature.
  python3 evidence_dpl.py list | dim | adapt | all | <ID>   [--runs N]"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_core import *
from dpl_lib import DPL
LIBNAME = os.path.basename(__file__)[9:-3]
RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 3000
MT = 240; BASE = {}
def base():
    if "b" not in BASE: BASE["b"] = run_fuzz(DPL, "full harness", runs=RUNS, max_time=MT)
    return BASE["b"]
def abl(label, env, **k): return run_fuzz(DPL, label, env=env, runs=RUNS, max_time=MT, **k)
COLS = ("label", "execs", "cov", "corpus", "unique_findings", "bugs")
DIMS = {
 "DIM-P1": ("program: facts, rules, negation, graphs (t_prop, t_graph)", {"DPL_SKIP": "t_prop,t_graph"}, "Program-shape bugs and the exact-probability oracle live here."),
 "DIM-P3": ("neural predicates (t_neural)", {"DPL_SKIP": "t_neural"}, "Network outputs (one-hot, zeros) reach the top-k and engine code (D7)."),
 "DIM-P6": ("learning: optimizer, train loop (t_learn, t_train)", {"DPL_SKIP": "t_learn,t_train"}, "Optimizer/logging bugs (D5, D6) only occur during training."),
 "DIM-P7a": ("API boundary (t_api)", {"DPL_SKIP": "t_api"}, "Bad arguments: DataLoader(batch_size=0), unhashable query substitutions, empty file names."),
 "DIM-P7b": ("data / datasets (t_data)", {"DPL_SKIP": "t_data"}, "Dataset round-trip, cache, stop-condition, ConfusionMatrix, mutator bugs."),
 "DIM-P1b": ("names and terms (t_names)", {"DPL_SKIP": "t_names"}, "Builtin predicates called with unusual term types (embedding, stack, dot)."),
 "DIM-FORM": ("file-name FORM: bare 'x.pl' vs './x.pl' vs 'sub/x.pl'", {"DPL_FREEZE": "form"}, "Bug D4 needs a BARE file name (no directory). If every name has a directory, the bug disappears."),
}
def run_dim(i):
    name, env, why = DIMS[i]; b = base(); w = abl("without: " + name.split(" (")[0].split(":")[0], env)
    lost = sorted(set(b["bugs"]) - set(w["bugs"]))
    return compare(DPL, i, "dimension " + name, why, b, w, COLS, note="bugs lost without this dimension: %s" % (lost or "none in this budget"))
def A1():
    on = base(); off = abl("A1 off: no instrumentation", {"DPL_INSTR": "none"})
    return compare(DPL, "A1", "libFuzzer needs coverage feedback from Python", "No instrumentation: corpus does not grow, search is blind random testing.", on, off, COLS)
def A9():
    on = base(); off = abl("A9 off: deepproblog only", {"DPL_INSTR": "deepproblog"})
    return compare(DPL, "A9", "instrument deepproblog AND problog (the solver does the real work)",
      "Probabilistic inference runs inside problog. Instrumenting only deepproblog hides the solver, so the fuzzer gets no feedback from it.", on, off, ("label", "execs", "cov", "ft", "corpus", "exec_s", "unique_findings", "bugs"))
def A3():
    naive = dict(DPL, cwd=os.path.dirname(os.path.abspath(__file__)), script="naive_dpl.py", name="dpl_naive", out_env="NAIVE_OUT", count_env=None)
    on = base(); off = run_fuzz(naive, "A3 off: one fixed program", runs=RUNS, max_time=MT)
    return compare(DPL, "A3", "decode bytes into program STRUCTURE (rules, queries, networks, datasets)", "A fixed two-fact program reaches very little; structured decoding reaches the whole API.", on, off, ("label", "execs", "cov", "ft", "corpus", "unique_findings", "bugs"))
def A4():
    b = base(); ora = [k for k in b["finding_kinds"] if k.startswith("oracle")]; exc = [k for k in b["finding_kinds"] if k.startswith("exception")]
    print("=" * 78); print("A4 - independent ORACLES are needed (crash-only fuzzing is blind to wrong answers)")
    print("findings in the full run: %d   crash-visible (exceptions): %d   oracle-only (no crash at all): %d" % (len(b["finding_kinds"]), len(exc), len(ora)))
    for k in ora: print("   oracle-only:", k)
    print("READ: the oracle-only ones (wrong round trip, probability mass > 1, endless DataLoader) would be missed by a crash-only fuzzer.")
    save(LIBNAME + "_A4", {"id": "A4", "exceptions": exc, "oracle_only": ora})
def A5():
    r = base(); print("=" * 78); print("A5 - keep going + de-duplicate findings by signature")
    print("raw bug hits: %s   unique findings kept: %s" % (r["raw_calls"], r["unique_findings"])); save(LIBNAME + "_A5", {"id": "A5", "raw": r["raw_calls"], "unique": r["unique_findings"]})
def A6():
    import warnings; warnings.filterwarnings("ignore")
    from problog.program import PrologString; from problog import get_evaluatable
    out = []
    for lab, body in (("rule order: \\+ f(X), g(X)  (negation first)", "q(X) :- \\+ f(X), g(X)."), ("rule order: g(X), \\+ f(X)  (generator first)", "q(X) :- g(X), \\+ f(X).")):
        prog = "0.3::f(1). g(1). g(2). %s query(q(X))." % body
        try: r = get_evaluatable().create_from(PrologString(prog)).evaluate(); res = {str(k): round(v, 3) for k, v in r.items()}
        except Exception as e: res = "error: " + str(e).split("\n")[0][:70]
        out.append({"label": lab, "result": res})
    print("=" * 78); print("A6 - validate the harness itself: a rule written in the wrong order is NOT a library bug (Prolog evaluates left to right)")
    table(out, ["label", "result"]); print("READ: with negation first, q(2) comes out 0.7 (wrong: f(2) does not exist, so q(2) should be 1.0); written generator-first it is right. This is Prolog semantics, not a library bug, so the oracle was changed to write rules generator-first.")
    save(LIBNAME + "_A6", {"id": "A6", "rows": out})
def A7():
    a = run_fuzz(DPL, "A7 short run", runs=300, max_time=MT); b = run_fuzz(DPL, "A7 long run", runs=2000, max_time=MT)
    slope = (b["rss_mb"] - a["rss_mb"]) * 1024 / max(1, b["execs"] - a["execs"])
    print("=" * 78); print("A7 - run in restarted chunks because Atheris leaks memory"); table([a, b], ["label", "execs", "rss_mb"])
    print("growth: about %.1f KB per execution -> restart the process in chunks and keep the corpus" % slope); save(LIBNAME + "_A7", {"id": "A7", "short": a, "long": b, "kb_per_exec": slope})
def A10():
    on = base(); off = run_fuzz(DPL, "A10 off: libFuzzer default length control", runs=RUNS, max_time=MT, flags=[])
    return compare(DPL, "A10", "-len_control=0 -max_len=200", "Default settings grow inputs slowly from tiny; fixed small max length gives full structured inputs from the start.", on, off, ("label", "execs", "cov", "corpus", "exec_s", "unique_findings", "bugs"))
def A11():
    on = base(); off = abl("A11 off: no loop guards, 10 s timeout", {"DPL_NOGUARD": "1"}, flags=["-len_control=0", "-max_len=200", "-timeout=10"]) if False else None
    off = run_fuzz(DPL, "A11 off: no loop guards", env={"DPL_NOGUARD": "1"}, runs=RUNS, max_time=MT, flags=["-len_control=0", "-max_len=200", "-timeout=10"])
    return compare(DPL, "A11", "-timeout and loop counters", "A hang (DataLoader(batch_size=0) never stops) would freeze the campaign. Counters turn it into a finding.", on, off, ("label", "execs", "timeout", "unique_findings", "bugs"))
def A12():
    on = base(); off = abl("A12 off: no temp-dir isolation", {"DPL_NOTMP": "1"})
    return compare(DPL, "A12", "run each input in a temporary directory", "The library reads/writes files by name. Without isolation, stray files pile up and runs influence each other (stale file names).", on, off, ("label", "execs", "stray_files", "unique_findings", "bugs"))
EXP = {**{k: (lambda k=k: run_dim(k)) for k in DIMS}, "A1": A1, "A3": A3, "A4": A4, "A5": A5, "A6": A6, "A7": A7, "A9": A9, "A10": A10, "A11": A11, "A12": A12}
if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--") and not x.isdigit()]; cmd = a[0] if a else "list"
    if cmd == "list":
        for k in EXP: print(k, "-", DIMS[k][0] if k in DIMS else "")
    else:
        ids = [k for k in EXP if k.startswith("DIM")] if cmd == "dim" else [k for k in EXP if not k.startswith("DIM")] if cmd == "adapt" else list(EXP) if cmd == "all" else [cmd]
        for k in ids: EXP[k]()
