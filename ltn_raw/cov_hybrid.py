#!/usr/bin/env python3
"""Coverage of the `ltn` package for a saved corpus of ltn_hybrid_fuzz.py. Replays the corpus without libFuzzer.

What it prints and writes (into --out, default ./cov_out):
  1. a per-file table: statements, missed, branches, percent            (printed)
  2. cov_functions.csv   one row per function/method of ltn: statements, executed, percent, called yes/no
  3. never-called functions, grouped by file                             (printed)
  4. cov_growth.csv      cumulative covered lines after each input, in the order the inputs were saved
                         (the fuzzer's learning curve: plot it, or look for where it flattens)
  5. cov_summary.json    the totals (read by run_until_plateau.py)
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
rows, never = [], {}
for path in sorted(cov.get_data().measured_files()):
    _, statements, _, missing, _ = cov.analysis2(path)
    executed = set(statements) - set(missing)
    rel = os.path.relpath(path, LTN_SRC)
    tree = ast.parse(open(path).read())

    def visit(node, prefix=""):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                visit(ch, prefix + ch.name + ".")
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
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

with open(os.path.join(args.out, "cov_functions.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "function", "line", "statements", "executed", "percent", "called"])
    w.writeheader()
    w.writerows(rows)

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

if args.html:
    cov.html_report(directory=os.path.join(args.out, "html"))
    print(f"html report: {os.path.join(args.out, 'html', 'index.html')}")
print(f"\nwrote {args.out}/cov_functions.csv and {args.out}/cov_growth.csv")
