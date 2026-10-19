#!/usr/bin/env python3
"""Coverage of the `ltn` package for a saved corpus of ltn_hybrid_fuzz.py. Replays the corpus without libFuzzer.

What it prints and writes (into --out, default ./cov_out):
  1. a per-file table: statements, missed, branches, percent            (printed)
  2. cov_functions.csv   one row per function/method of ltn: statements, executed, percent, called yes/no
  3. never-called functions, grouped by file                             (printed)
  4. cov_growth.csv      cumulative covered lines after each input, in the order the inputs were saved
                         (the fuzzer's learning curve: plot it, or look for where it flattens)
  5. cov_summary.json    the totals (read by run_until_plateau.py)
  5b. never_called.csv, covered_lines.txt, cov_lines.csv   which functions and lines were / were not reached
  5c. verdicts.csv, failures.csv   every harness call with its oracle verdict (the replay also writes the trace)
  6. --html              also writes an HTML report you can open in a browser

libFuzzer's own `cov:` number counts edges. This reports lines and branches of ltn/*.py, which is what you can read.

Run:   python cov_hybrid.py corpus_hybrid
       python cov_hybrid.py corpus_hybrid --out cov_out --html
Needs: coverage, torch, numpy and atheris installed (the harness imports atheris), and the LTN source at LTN_SRC
       (default ./ltn_torch).
"""
import argparse
import ast
import csv
import glob
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("corpus", help="folder of saved inputs (the corpus folder libFuzzer wrote), or a single file")
ap.add_argument("--out", default="cov_out")
ap.add_argument("--html", action="store_true")
ap.add_argument("--growth-every", type=int, default=0,
                help="record cumulative coverage every N inputs (default: about 300 points across the corpus)")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
LTN_SRC = os.path.abspath(os.environ.get("LTN_SRC", os.path.join(HERE, "ltn_torch")))
LTN_DIR = os.path.join(LTN_SRC, "ltn")
sys.path.insert(0, LTN_SRC)
sys.path.insert(0, HERE)
os.environ["HYBRID_NO_INSTRUMENT"] = "1"       # coverage.py and Atheris's bytecode probes should not share a run
os.makedirs(args.out, exist_ok=True)

# the harness reads TRACE_CSV at import time, so set it first: one replay then gives coverage AND verdicts
TRACE = os.environ.get("TRACE_CSV") or os.path.join(args.out, "verdicts.csv")
os.environ["TRACE_CSV"] = TRACE
if os.path.exists(TRACE):
    os.remove(TRACE)

import coverage  # noqa: E402

cov = coverage.Coverage(source=[LTN_DIR], branch=True)
cov.start()                                    # start BEFORE the import so module-level lines of ltn count
import ltn_hybrid_fuzz as H  # noqa: E402

files = [args.corpus] if os.path.isfile(args.corpus) else [p for p in glob.glob(os.path.join(args.corpus, "*"))
                                                           if os.path.isfile(p)]
files.sort(key=lambda p: (os.path.getmtime(p), p))      # oldest first = the order the fuzzer found them
if not files:
    sys.exit(f"no inputs found in {args.corpus}")
every = args.growth_every or max(1, len(files) // 300)


def covered_lines():
    cov.stop()
    d = cov.get_data()
    n = sum(len(d.lines(f) or []) for f in d.measured_files())
    cov.start()
    return n


growth = [(0, "", covered_lines())]
for i, p in enumerate(files, 1):
    try:
        H.TestOneInput(open(p, "rb").read())
    except Exception as exc:  # noqa: BLE001  (the harness catches its own; this is only a safety net)
        print(f"  [{os.path.basename(p)}] escaped the harness: {type(exc).__name__}: {exc}", file=sys.stderr)
    if i % every == 0 or i == len(files):
        growth.append((i, os.path.basename(p), covered_lines()))
cov.stop()
cov.save()
print(f"replayed {len(files)} inputs\n")

# ---------------------------------------------------------------- 1. per-file table
buf = io.StringIO()
total_percent = cov.report(file=buf, show_missing=False, skip_empty=True)
print(buf.getvalue())

# ---------------------------------------------------------------- 2/3. per-function
rows, never, line_rows, ranges = [], {}, [], {}


def compress(nums):
    """[1,2,3,7,9,10] -> '1-3, 7, 9-10'"""
    out, nums = [], sorted(nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


for path in sorted(cov.get_data().measured_files()):
    _, statements, _, missing, _ = cov.analysis2(path)
    executed = set(statements) - set(missing)
    rel = os.path.relpath(path, LTN_SRC)
    tree = ast.parse(open(path).read())
    spans = []                                   # (first line, last line, qualified function name)

    def visit(node, prefix=""):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                visit(ch, prefix + ch.name + ".")
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((ch.lineno, ch.end_lineno, prefix + ch.name))
                start = ch.body[0].lineno
                stmts = [l for l in statements if start <= l <= ch.end_lineno]
                hit = [l for l in stmts if l in executed]
                called = bool(hit)
                rows.append(dict(file=rel, function=prefix + ch.name, line=ch.lineno, statements=len(stmts),
                                 executed=len(hit), percent=round(100 * len(hit) / len(stmts), 1) if stmts else 100.0,
                                 called="yes" if called else "NO"))
                if not called and stmts:
                    never.setdefault(rel, []).append(f"{prefix}{ch.name} (line {ch.lineno})")
                visit(ch, prefix + ch.name + ".<locals>.")
            else:
                visit(ch, prefix)
    visit(tree)
    for ln in sorted(statements):                # every executable line, and which function it belongs to
        inside = [sp for sp in spans if sp[0] <= ln <= sp[1]]
        fn = min(inside, key=lambda sp: sp[1] - sp[0])[2] if inside else "<module level>"
        line_rows.append(dict(file=rel, line=ln, function=fn, status="covered" if ln in executed else "MISSED"))
    ranges[rel] = (compress(executed), compress(set(statements) - executed))

with open(os.path.join(args.out, "cov_functions.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "function", "line", "statements", "executed", "percent", "called"])
    w.writeheader()
    w.writerows(rows)

with open(os.path.join(args.out, "cov_lines.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "line", "function", "status"])
    w.writeheader()
    w.writerows(line_rows)
with open(os.path.join(args.out, "covered_lines.txt"), "w") as fh:
    for rel, (cov_r, miss_r) in ranges.items():
        fh.write(f"{rel}\n  covered: {cov_r}\n  missed : {miss_r}\n\n")
with open(os.path.join(args.out, "never_called.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "function", "line", "statements"])
    w.writeheader()
    w.writerows([{k: r[k] for k in w.fieldnames} for r in rows if r["called"] == "NO" and r["statements"]])

total_fn = len(rows)
called_fn = sum(r["called"] == "yes" for r in rows)
print(f"functions called at least once: {called_fn} / {total_fn}")
for rel, names in never.items():
    print(f"\nNEVER CALLED in {rel}  ({len(names)})")
    for n in names:
        print("   ", n)

# ---------------------------------------------------------------- 4. growth curve
with open(os.path.join(args.out, "cov_growth.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["inputs_replayed", "last_input", "lines_covered", "new_lines_since_last_point"])
    prev = 0
    for n, name, lines in growth:
        w.writerow([n, name, lines, lines - prev])
        prev = lines
print(f"\ngrowth: {growth[0][2]} lines at start -> {growth[-1][2]} after {len(files)} inputs "
      f"(last point added {growth[-1][2] - growth[-2][2] if len(growth) > 1 else 0})")

import json  # noqa: E402
json.dump({"inputs": len(files), "lines_covered": growth[-1][2], "percent": round(total_percent, 2),
           "functions_called": called_fn, "functions_total": total_fn},
          open(os.path.join(args.out, "cov_summary.json"), "w"))

# ---------------------------------------------------------------- verdicts from the same replay
from collections import Counter  # noqa: E402
verdicts, fails = Counter(), {}
if os.path.exists(TRACE):
    with open(TRACE, newline="") as fh:
        for r in csv.DictReader(fh):
            verdicts[r["verdict"]] += 1
            if r["verdict"].startswith("FAIL"):
                fails.setdefault((r["verdict"], r["call"], r["exception"][:100]), r)
print("\nverdicts over every call made during the replay:")
for k, v in verdicts.most_common():
    print(f"  {v:>7}  {k}")
with open(os.path.join(args.out, "failures.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["verdict", "target", "call", "choices", "output", "exception", "input_sha1"],
                       extrasaction="ignore")
    w.writeheader()
    for r in fails.values():
        w.writerow(r)
print(f"distinct failures: {len(fails)}  (see failures.csv)")

if args.html:
    cov.html_report(directory=os.path.join(args.out, "html"))
    print(f"html report: {os.path.join(args.out, 'html', 'index.html')}")
print(f"""
wrote in {args.out}/:
  never_called.csv    functions of ltn that no input reached
  covered_lines.txt   covered and missed line ranges per file
  cov_lines.csv       every executable line: covered or MISSED, and which function it is in
  cov_functions.csv   per function: statements, executed, percent
  verdicts.csv        one row per call: what was chosen, the output, the oracle, the verdict
  failures.csv        the distinct FAIL rows from verdicts.csv
  cov_growth.csv      cumulative lines covered, in discovery order""")
