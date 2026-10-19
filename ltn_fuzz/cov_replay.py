#!/usr/bin/env python3
"""Replay a saved corpus of ltn_fuzz.py under coverage.py and report what of `ltn` was reached.

Same replay as measure_cov.py (Builder(data).run() for each input, no libFuzzer), but it also writes files that run_until_plateau.py
reads, and it lists the functions and lines that were never reached.

Writes into --out (default ./cov_out):
    cov_summary.json    inputs replayed, lines covered, percent, functions called / total   (read by run_until_plateau.py)
    cov_growth.csv      cumulative covered lines, in the order the fuzzer saved the inputs
    cov_functions.csv   per function: statements, executed, percent, called
    never_called.csv    functions whose body never ran
    cov_lines.csv       every executable line: covered or MISSED, and the function it is in
    covered_lines.txt   covered / missed line ranges per file
    verdicts.csv        one row per input: the oracles that applied, the verdict, the exception, and for failures the
                        generated program step by step (the same idea as the hybrid harness's verdicts.csv)
    failures.csv        the distinct FAIL rows of verdicts.csv, one per failure signature

Run:   LTN_SRC=/path/to/ltn_torch python cov_replay.py corpus_valid --out cov_out
"""
import argparse
import ast
import csv
import glob
import hashlib
import io
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("corpus")
ap.add_argument("--out", default="cov_out")
ap.add_argument("--growth-every", type=int, default=0)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
LTN_SRC = os.path.abspath(os.environ.get("LTN_SRC", os.path.join(HERE, "..", "ltn_torch")))
LTN_DIR = os.path.join(LTN_SRC, "ltn")
os.environ["LTNFUZZ_NO_INSTRUMENT"] = "1"       # coverage.py and Atheris's probes should not share a run
os.environ["LTNFUZZ_COLLECT"] = "0"
sys.argv = [sys.argv[0]]
sys.path.insert(0, HERE)
sys.path.insert(0, LTN_SRC)
os.makedirs(args.out, exist_ok=True)

import coverage  # noqa: E402

cov = coverage.Coverage(source=[LTN_DIR], branch=True)
cov.start()                                      # before the import, so module-level lines of ltn count
import ltn_fuzz as F  # noqa: E402

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


VERDICT_STAGES = os.environ.get("VERDICT_STAGES", "fail")   # "fail": program steps only on FAIL rows; "all": every row
DEFINITION = re.compile(r"^(x|y|z|c\d+) = ")                   # trace lines that define the input data


def replay_one(data):
    """Run one input exactly as ltn_fuzz.TestOneInput does, but classify the outcome instead of raising or saving."""
    bld = F.Builder(data)
    verdict, exc_text, sig = "pass", "", ""
    try:
        bld.run()
    except Exception as exc:  # noqa: BLE001
        sig = F.signature(exc)
        exc_text = f"{type(exc).__name__}: {str(exc)[:200]}"
        if isinstance(exc, F.OracleViolation):
            verdict = "FAIL: value outside [0,1] or NaN"
        elif isinstance(exc, F.GradientViolation):
            verdict = "FAIL: non-finite gradient"
        elif F.is_expected(exc):
            verdict = "expected validation error"
        else:
            verdict = "FAIL: unexpected exception"
    # which oracles could have fired for this program (the flags are the harness's own bookkeeping)
    oracle = "exception"
    if bld.all_finite:
        oracle += "+value[0,1]"
        if bld.all_stable:
            oracle += "+gradient"
    return dict(
        verdict=verdict, exception=exc_text, signature=sig, oracle_applied=oracle,
        last_completed_stage=bld.trace[-1] if bld.trace else "none (failed before the first step)",
        inputs=json.dumps([t for t in bld.trace if DEFINITION.match(t)]),
        stages=json.dumps(bld.trace),
        flags=json.dumps(dict(all_finite=bld.all_finite, all_stable=bld.all_stable,
                              empty_domain=bld.empty_domain, aggregators=sorted(bld.aggs))))


growth = [(0, "", covered_lines())]
verdict_rows, fail_rows, seen_sigs = [], {}, set()
for i, p in enumerate(files, 1):
    data = open(p, "rb").read()
    try:
        r = replay_one(data)
    except Exception as exc:  # noqa: BLE001  (the classifier itself must never stop the replay)
        r = dict(verdict="FAIL: replay error", exception=f"{type(exc).__name__}: {str(exc)[:200]}", signature="",
                 oracle_applied="", last_completed_stage="", inputs="", stages="", flags="")
    failed = r["verdict"].startswith("FAIL")
    r.update(exec=i, input_file=os.path.basename(p), input_sha1=hashlib.sha1(data).hexdigest()[:12],
             input_hex=data[:96].hex(), n_bytes=len(data),
             new_finding=bool(failed and r["signature"] not in seen_sigs))
    if failed:
        seen_sigs.add(r["signature"])
        fail_rows.setdefault(r["signature"] or r["exception"], dict(r))
    if not failed and VERDICT_STAGES != "all":
        r["inputs"] = r["stages"] = ""              # keep the file small: program steps only for failures
    verdict_rows.append(r)
    if i % every == 0 or i == len(files):
        growth.append((i, os.path.basename(p), covered_lines()))
cov.stop()
cov.save()
print(f"replayed {len(files)} inputs\n")

buf = io.StringIO()
total_percent = cov.report(file=buf, show_missing=False, skip_empty=True)
print(buf.getvalue())


def compress(nums):
    out, nums = [], sorted(nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


rows, never, line_rows, ranges = [], {}, [], {}
for path in sorted(cov.get_data().measured_files()):
    _, statements, _, missing, _ = cov.analysis2(path)
    executed = set(statements) - set(missing)
    rel = os.path.relpath(path, LTN_SRC)
    tree = ast.parse(open(path).read())
    spans = []

    def visit(node, prefix=""):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                visit(ch, prefix + ch.name + ".")
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((ch.lineno, ch.end_lineno, prefix + ch.name))
                start = ch.body[0].lineno
                stmts = [l for l in statements if start <= l <= ch.end_lineno]
                hit = [l for l in stmts if l in executed]
                rows.append(dict(file=rel, function=prefix + ch.name, line=ch.lineno, statements=len(stmts),
                                 executed=len(hit), percent=round(100 * len(hit) / len(stmts), 1) if stmts else 100.0,
                                 called="yes" if hit else "NO"))
                if not hit and stmts:
                    never.setdefault(rel, []).append(f"{prefix}{ch.name} (line {ch.lineno})")
                visit(ch, prefix + ch.name + ".<locals>.")
            else:
                visit(ch, prefix)
    visit(tree)
    for ln in sorted(statements):
        inside = [sp for sp in spans if sp[0] <= ln <= sp[1]]
        fn = min(inside, key=lambda sp: sp[1] - sp[0])[2] if inside else "<module level>"
        line_rows.append(dict(file=rel, line=ln, function=fn, status="covered" if ln in executed else "MISSED"))
    ranges[rel] = (compress(executed), compress(set(statements) - executed))


def write_csv(name, fields, data):
    with open(os.path.join(args.out, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


write_csv("cov_functions.csv", ["file", "function", "line", "statements", "executed", "percent", "called"], rows)
write_csv("never_called.csv", ["file", "function", "line", "statements"],
          [{k: r[k] for k in ("file", "function", "line", "statements")} for r in rows
           if r["called"] == "NO" and r["statements"]])
write_csv("cov_lines.csv", ["file", "line", "function", "status"], line_rows)
with open(os.path.join(args.out, "covered_lines.txt"), "w") as fh:
    for rel, (c, m) in ranges.items():
        fh.write(f"{rel}\n  covered: {c}\n  missed : {m}\n\n")
with open(os.path.join(args.out, "cov_growth.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["inputs_replayed", "last_input", "lines_covered", "new_lines_since_last_point"])
    prev = 0
    for n, name, lines in growth:
        w.writerow([n, name, lines, lines - prev])
        prev = lines

VERDICT_FIELDS = ["exec", "input_file", "input_sha1", "input_hex", "n_bytes", "oracle_applied", "verdict", "exception",
                  "signature", "new_finding", "last_completed_stage", "inputs", "stages", "flags"]
write_csv("verdicts.csv", VERDICT_FIELDS, verdict_rows)
write_csv("failures.csv", VERDICT_FIELDS, list(fail_rows.values()))

called_fn = sum(r["called"] == "yes" for r in rows)
json.dump({"inputs": len(files), "lines_covered": growth[-1][2], "percent": round(total_percent, 2),
           "functions_called": called_fn, "functions_total": len(rows), "distinct_failures": len(fail_rows)},
          open(os.path.join(args.out, "cov_summary.json"), "w"))

print(f"functions called at least once: {called_fn} / {len(rows)}")
for rel, names in never.items():
    print(f"\nNEVER CALLED in {rel}  ({len(names)})")
    for n in names:
        print("   ", n)
from collections import Counter  # noqa: E402
print("\nverdicts, one per input:")
for k, v in Counter(r["verdict"] for r in verdict_rows).most_common():
    print(f"  {v:>7}  {k}")
print(f"distinct failures: {len(fail_rows)}  (failures.csv)")
print(f"\ngrowth: {growth[0][2]} lines at start -> {growth[-1][2]} after {len(files)} inputs")
print(f"wrote results in {args.out}/")
