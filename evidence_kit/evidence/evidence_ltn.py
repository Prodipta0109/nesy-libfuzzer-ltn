"""Why these dimensions and adaptations?  Each experiment runs the real LTN fuzzer WITH and WITHOUT the feature.
  python3 evidence_ltn.py list | dim | adapt | all | <ID>   [--runs N]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_core import *
from ltn_lib import LTN
LIBNAME = os.path.basename(__file__)[9:-3]
RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 20000
MT = 240; BASE = {}
FULL = {"LTNFUZZ_MISUSE": "1"}
def base():
    if "b" not in BASE: BASE["b"] = run_fuzz(LTN, "full harness (D1-D7)", env=FULL, runs=RUNS, max_time=MT)
    return BASE["b"]
def abl(label, env, **k): return run_fuzz(LTN, label, env={**FULL, **env}, runs=RUNS, max_time=MT, **k)
COLS = ("label", "execs", "cov", "corpus", "unique_findings", "bugs")
DIMS = {
 "DIM-D1": ("D1 data values: 0, 1, eps, NaN, inf, huge", {"LTNFUZZ_FREEZE": "values"}, "Without extreme numbers only ordinary values flow in; NaN/inf-dependent behaviour is never exercised."),
 "DIM-D2": ("D2 data shape: 0 individuals, 1-D tensors", {"LTNFUZZ_FREEZE": "shape"}, "An EMPTY domain (zero individuals) makes 'for all x' return NaN (F1). With n >= 1 always, it cannot happen."),
 "DIM-D5": ("D5 quantifier setup: p values and stable flag", {"LTNFUZZ_FREEZE": "quant"}, "p = 0 / negative p and unstable variants trigger division by zero (F2) and gradient problems."),
 "DIM-D6": ("D6 satisfaction + backward pass", {"LTNFUZZ_FREEZE": "backward"}, "Gradient NaNs (F27) only show when loss.backward() runs."),
 "DIM-D7": ("D7 API boundary (misuse of the API)", {"LTNFUZZ_MISUSE": "0"}, "Vacuous quantification and SatAgg on open formulas expose input-validation bugs (F3, F30-F33)."),
}
def run_dim(i):
    name, env, why = DIMS[i]; b = base(); w = abl("without: " + name.split(":")[0], env)
    lost = sorted(set(b["bugs"]) - set(w["bugs"]))
    return compare(LTN, i, "dimension " + name, why, b, w, COLS, note="bug groups lost without this dimension: %s" % (lost or "none in this budget"))
def A1():
    on = base(); off = abl("A1 off: no instrumentation", {"LTNFUZZ_NO_INSTRUMENT": "1"})
    return compare(LTN, "A1", "libFuzzer needs coverage feedback from Python (Atheris)", "No instrumentation: corpus never grows; blind random testing.", on, off, COLS)
def A2():
    on = base(); off = abl("A2 off: instrument ltn + torch", {"LTNFUZZ_INSTR_TORCH": "1"})
    return compare(LTN, "A2", "instrument ONLY ltn (not torch)", "torch is huge; counting its branches slows every run and rewards inputs that only wiggle torch.", on, off, ("label", "execs", "cov", "ft", "corpus", "exec_s", "unique_findings", "bugs"))
def A3():
    naive = dict(LTN, cwd=os.path.dirname(os.path.abspath(__file__)), script="naive_ltn.py", name="ltn_naive", out_env="NAIVE_OUT", base_env={})
    on = base(); off = run_fuzz(naive, "A3 off: fixed formula, bytes->numbers only", runs=RUNS, max_time=MT)
    return compare(LTN, "A3", "decode bytes into a PROGRAM (domains, predicates, connectives, quantifiers)", "One fixed formula reaches a tiny part of LTN; structured decoding reaches operators, quantifier variants and the backward pass.", on, off, ("label", "execs", "cov", "ft", "corpus", "unique_findings", "bugs"))
def A4():
    on = base(); off = abl("A4 off: basic oracles (crash/NaN/range)", {"LTNFUZZ_ORACLES": "basic"})
    return compare(LTN, "A4", "independent ORACLES (dtype, metamorphic, formula-vs-math checks)", "Many LTN bugs return a wrong number or wrong dtype without crashing; the basic checks cannot see them.", on, off, COLS)
def A5():
    r = base(); print("=" * 78); print("A5 - keep going + de-duplicate findings by signature")
    print("raw failures hit: %s   unique findings kept: %s" % (r["raw_calls"], r["unique_findings"])); save(LIBNAME + "_A5", {"id": "A5", "raw": r["raw_calls"], "unique": r["unique_findings"]})
def A7():
    a = run_fuzz(LTN, "A7 short run", env=FULL, runs=2000, max_time=MT); b = run_fuzz(LTN, "A7 long run", env=FULL, runs=15000, max_time=MT)
    slope = (b["rss_mb"] - a["rss_mb"]) * 1024 / max(1, b["execs"] - a["execs"])
    print("=" * 78); print("A7 - run in restarted chunks because Atheris leaks memory"); table([a, b], ["label", "execs", "rss_mb"])
    print("growth: about %.1f KB per execution -> restart in chunks, keep the corpus" % slope); save(LIBNAME + "_A7", {"id": "A7", "short": a, "long": b, "kb_per_exec": slope})
def A8():
    r3 = dict(LTN, script="ltn_round3.py", name="ltn_r3", out_env="R3_OUT", base_env={})
    on = base(); off = run_fuzz(r3, "A8: round-3 harness (extra dimensions)", runs=RUNS, max_time=MT)
    return compare(LTN, "A8", "round 3: more dimensions and oracles (reuse, nested, same label, mask, duality)", "A second harness with different dimensions finds things the first cannot: the bug SETS differ.", on, off, ("label", "execs", "cov", "corpus", "unique_findings"), note="compare the finding kinds in results/A8.json")
EXP = {**{k: (lambda k=k: run_dim(k)) for k in DIMS}, "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A7": A7, "A8": A8}
if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--") and not x.isdigit()]; cmd = a[0] if a else "list"
    if cmd == "list":
        for k in EXP: print(k, "-", DIMS[k][0] if k in DIMS else "")
    else:
        ids = [k for k in EXP if k.startswith("DIM")] if cmd == "dim" else [k for k in EXP if not k.startswith("DIM")] if cmd == "adapt" else list(EXP) if cmd == "all" else [cmd]
        for k in ids: EXP[k]()
