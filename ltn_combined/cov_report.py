#!/usr/bin/env python3
"""Per-function line coverage of the ltn package, from the .coverage data written by cov_replay.py.

usage:  python cov_report.py <path to ltn package dir>      (run after `coverage combine`)
"""
import ast
import os
import sys
import warnings

import coverage

warnings.filterwarnings("ignore")   # LTN's docstrings have invalid escape sequences

pkg = os.path.abspath(sys.argv[1])
cov = coverage.Coverage()
cov.load()

grand_total = grand_missing = 0
rows = []
for fname in ("core.py", "fuzzy_ops.py"):
    path = os.path.join(pkg, fname)
    _, statements, _, missing, _ = cov.analysis2(path)
    statements, missing = set(statements), set(missing)
    tree = ast.parse(open(path).read())

    funcs = []                                   # (start, end, qualname)

    def walk(node, prefix=""):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + ".")
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append((ch.lineno, ch.end_lineno, prefix + ch.name))
                walk(ch, prefix + ch.name + ".")

    walk(tree)

    def owner(line):                             # innermost function containing this line
        best = None
        for s, e, q in funcs:
            if s <= line <= e and (best is None or e - s < best[1] - best[0]):
                best = (s, e, q)
        return best

    per = {}
    for ln in statements:
        o = owner(ln)
        key = o[2] if o else "<module level>"
        t, m, ml = per.get(key, (0, 0, []))
        per[key] = (t + 1, m + (ln in missing), ml + ([ln] if ln in missing else []))
    for q, (t, m, ml) in per.items():
        rows.append((fname, q, t, m, ml))
    grand_total += len(statements)
    grand_missing += len(missing)

pct = lambda t, m: 100.0 * (t - m) / t if t else 100.0
never = [r for r in rows if r[2] and r[3] == r[2]]
partial = [r for r in rows if 0 < r[3] < r[2]]
full = [r for r in rows if r[3] == 0]

print(f"\nLINE COVERAGE of ltn/: {pct(grand_total, grand_missing):.1f}%  "
      f"({grand_total - grand_missing} of {grand_total} statements)")
print(f"functions: {len(rows)} total | {len(full)} fully covered | {len(partial)} partly | {len(never)} never run\n")

print("NEVER RUN")
for f, q, t, m, ml in sorted(never):
    print(f"  {f:13} {q:45} 0/{t} statements")
print("\nPARTLY COVERED (lowest first)")
for f, q, t, m, ml in sorted(partial, key=lambda r: pct(r[2], r[3])):
    shown = ",".join(map(str, sorted(ml)[:8])) + ("..." if len(ml) > 8 else "")
    print(f"  {f:13} {q:45} {pct(t, m):5.1f}%  missed lines: {shown}")
print(f"\nFULLY COVERED: {len(full)} functions")
