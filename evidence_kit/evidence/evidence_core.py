"""Shared helper for the "why this dimension / why this adaptation" experiments.
Every experiment runs the REAL harness under libFuzzer twice (feature on / feature off), with the same seed and
the same number of executions, then reads the libFuzzer log and the finding directories."""
import os, re, sys, json, glob, time, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

def run_fuzz(lib, label, env=None, flags=None, runs=1500, max_time=90, seed=1, corpus_seed=None):
    """lib = dict(cwd, script, out_env, count_env, bugmap). Returns a metrics dict."""
    work = os.path.join("/tmp/evidence_work", lib["name"] + "_%d" % os.getpid(), re.sub(r"\W+", "_", label))
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work + "/corpus"); os.makedirs(work + "/cwd")
    if corpus_seed:
        for f in glob.glob(corpus_seed + "/*"): shutil.copy(f, work + "/corpus")
    e = dict(os.environ); e.update(lib.get("base_env", {})); e.update(env or {})
    e[lib["out_env"]] = work + "/findings"
    if lib.get("src"): e["PYTHONPATH"] = lib["src"]
    if lib.get("count_env"): e[lib["count_env"]] = "1"
    cmd = [sys.executable, os.path.join(lib["cwd"], lib["script"]), work + "/corpus", "-seed=%d" % seed,
           "-runs=%d" % runs, "-max_total_time=%d" % max_time, "-timeout=25", "-rss_limit_mb=3500"]
    cmd += ["-len_control=0", "-max_len=200"] if flags is None else flags
    t0 = time.time()
    p = subprocess.run(cmd, cwd=work + "/cwd", env=e, capture_output=True, text=True)
    dt = time.time() - t0
    log = p.stdout + p.stderr
    open(work + "/log.txt", "w").write(log)
    return parse(lib, label, log, work, dt)

def parse(lib, label, log, work, dt):
    m = {"label": label, "seconds": round(dt, 1)}
    last = None
    for ln in log.splitlines():
        r = re.match(r"#(\d+)\s+(\w+)\s+(.*)", ln)
        if r: last = r
    if last:
        m["execs"] = int(last.group(1)); rest = last.group(3)
        for k, pat in (("cov", r"cov: (\d+)"), ("ft", r"ft: (\d+)"), ("corpus", r"corp: (\d+)"),
                       ("exec_s", r"exec/s: (\d+)"), ("rss_mb", r"rss: (\d+)")):
            x = re.search(pat, rest); m[k] = int(x.group(1)) if x else None
    m["timeout"] = "ALARM: working on the last Unit" in log or "timeout after" in log
    m["oom"] = "out-of-memory" in log
    fd = work + "/findings"
    kinds, bugs, raw_sum = [], set(), 0
    if os.path.isdir(fd):
        for d in sorted(os.listdir(fd)):
            mp = os.path.join(fd, d, "meta.json")
            if os.path.exists(mp):
                j = json.load(open(mp)); txt = j.get("kind", "") + " " + j.get("msg", "") + " " + j.get("signature", "") + " " + j.get("detail", "") + " " + j.get("where", "") + " " + j.get("loc", "") + " " + j.get("trace", "")
                kinds.append(j.get("signature") or (j.get("kind", "?") + " @ " + (j.get("where") or j.get("loc") or "")))
                raw_sum += j.get("count", 0)
                for bid, pats in lib.get("bugmap", {}).items():
                    if any(re.search(p, txt) for p in pats): bugs.add(bid)
    m["unique_findings"] = len(kinds); m["finding_kinds"] = kinds; m["bugs"] = sorted(bugs)
    m["stray_files"] = sum(len(f) for _, _, f in os.walk(work + "/cwd"))
    cf = fd + "/_calls"
    m["raw_calls"] = os.path.getsize(cf) if os.path.exists(cf) else (raw_sum or None)
    return m

def table(rows, cols, title=""):
    if title: print("\n" + title)
    w = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    print("  ".join(c.ljust(x) for c, x in zip(cols, w)))
    print("  ".join("-" * x for x in w))
    for r in rows: print("  ".join(str(r.get(c, "")).ljust(x) for c, x in zip(cols, w)))

def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(obj, open(os.path.join(RESULTS, name + ".json"), "w"), indent=1)

def compare(lib, exp_id, title, why, on, off, cols=("label", "execs", "cov", "ft", "corpus", "exec_s", "rss_mb", "unique_findings", "bugs"), note=""):
    print("=" * 78); print(exp_id, "-", title); print("WHY:", why)
    table([on, off], list(cols))
    if note: print("READ:", note)
    out = {"id": exp_id, "title": title, "why": why, "on": on, "off": off, "note": note}
    save(lib["name"] + "_" + exp_id.replace(" ", "_"), out); return out

# ---------------- minimal pairs: the same library call, ONE dimension value changed ----------------
import signal, io, contextlib, traceback as _tb
def _run_one(fn, secs=8):
    def stop(*_): raise TimeoutError("did not return within %d s" % secs)
    old = signal.signal(signal.SIGALRM, stop); signal.alarm(secs)
    try:
        with contextlib.redirect_stdout(io.StringIO()): r = fn()
        return "OK" + (" (%s)" % r if r else "")
    except TimeoutError as e: return "BUG: " + str(e)
    except AssertionError as e: return "BUG: " + str(e)[:90]
    except Exception as e: return "BUG: %s: %s" % (type(e).__name__, str(e).split("\n")[0][:80])
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)

def run_pairs(name, pairs):
    """pairs = [(dimension, bug id, what changes, control_fn, fuzzer_fn)]"""
    rows = []
    for dim, bug, change, c, f in pairs:
        rows.append({"dimension": dim, "bug": bug, "only change": change, "control": _run_one(c), "fuzzer value": _run_one(f)})
    print("=" * 78); print("%s - minimal pairs: same call, one dimension value changed" % name)
    table(rows, ["dimension", "bug", "only change", "control", "fuzzer value"])
    save(name, {"id": name, "rows": rows}); return rows
