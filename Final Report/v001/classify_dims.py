#!/usr/bin/env python3
"""Auto-classify LTN's public API into fuzzing dimensions D1-D7.

Usage:
    python3 classify_dims.py <path-to-ltn-package>            # table on screen
    python3 classify_dims.py <path-to-ltn-package> --csv out.csv

How it works (static analysis with Python's `ast`, LTN is never imported):
  1. Find every public class, method and function in core.py and fuzzy_ops.py.
  2. Give each class a default dimension from its name or base class
     (e.g. anything inheriting BinaryConnectiveOperator -> D4).
  3. Give each parameter a dimension from its name (e.g. `p`, `mask` -> D5);
     if no rule matches, the parameter inherits the class dimension.
  4. Mark a parameter VALIDATED if the method raises an error in an `if` that
     tests it, or passes it to LTN's own check_values / check_mask.
     Unvalidated parameters are where the fuzzer should look first.

To apply this to another library (DeepProbLog, LNN, ...), edit the three rule
tables below; the rest of the script stays the same.
"""
import ast
import csv
import os
import sys

FILES = ["core.py", "fuzzy_ops.py"]

# --- rule table 1: class / function name or base class -> default dimension ---
CLASS_RULES = {
    "LTNObject": "D2", "Variable": "D2", "Constant": "D2",
    "Predicate": "D3", "Function": "D3", "LambdaModel": "D3",
    "Connective": "D4", "UnaryConnectiveOperator": "D4",
    "BinaryConnectiveOperator": "D4", "ConnectiveOperator": "D4",
    "Quantifier": "D5", "AggregationOperator": "D5",
    "diag": "D5", "undiag": "D5",
    "SatAgg": "D6",
    "pi_0": "D4/D5 (stable)", "pi_1": "D4/D5 (stable)",
    "process_ltn_objects": "internal",
    "check_values": "validator", "check_mask": "validator",
}

# --- rule table 2: parameter name -> dimension (overrides the class default) ---
PARAM_RULES = {
    "individuals": "D1+D2", "value": "D1+D2", "x": "D1", "y": "D1", "xs": "D1+D2",
    "add_batch_dim": "D2", "var_labels": "D2",
    "model": "D3", "func": "D3", "inputs": "D3 (arity)",
    "connective_op": "D4", "operands": "D4", "and_op": "D4",
    "implies_op": "D4", "formula": "D4",
    "agg_op": None,          # D5 in Quantifier, D6 in SatAgg -> use class default
    "quantifier": "D5", "p": "D5", "mask": "D5", "dim": "D5", "keepdim": "D5",
    "cond_vars": "D5", "cond_fn": "D5",
    "trainable": "D6",
    "closed_formulas": "D6+D7", "vars": "D5+D7", "var_label": "D7",
    "stable": None,          # D4 on connectives, D5 on aggregators -> class default
}

# --- rule table 3: LTN's own validation helpers ---
VALIDATORS = {"check_values", "check_mask"}

PUBLIC_METHODS = {"__init__", "__call__", "forward"}


def class_dim(name, bases, all_bases):
    """Dimension of a class: by its own name, else by the nearest base class."""
    seen, todo = set(), [name]
    while todo:
        n = todo.pop(0)
        if n in CLASS_RULES:
            return CLASS_RULES[n]
        if n not in seen:
            seen.add(n)
            todo.extend(all_bases.get(n, []))
    return "?"


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def validated_params(func):
    """Names of parameters that the function checks before using them.

    Follows copies: in `t = list(closed_formulas)` a check on `t` counts as a
    check on `closed_formulas`.
    """
    origin = {p: {p} for p in params_of(func)}          # variable -> params it came from
    changed = True
    while changed:                                     # propagate through assignments
        changed = False
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and node.value is not None:
                src = set().union(*(origin.get(n, set()) for n in names_in(node.value)))
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for name in names_in(t):
                        if not src <= origin.get(name, set()):
                            origin[name] = origin.get(name, set()) | src
                            changed = True

    checked = set()
    for node in ast.walk(func):
        if isinstance(node, ast.If) and any(isinstance(n, ast.Raise) for n in ast.walk(node)):
            checked |= names_in(node.test)
        if isinstance(node, ast.Call):
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if callee in VALIDATORS:
                for arg in node.args:
                    checked |= names_in(arg)
    return set().union(*(origin.get(n, set()) for n in checked)) if checked else set()


def params_of(func):
    a = func.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    return [n for n in names if n not in ("self", "cls")]


def rows_for(path):
    tree = ast.parse(open(path).read())
    fname = os.path.basename(path)
    all_bases = {n.name: [ast.unparse(b) for b in n.bases]
                 for n in tree.body if isinstance(n, ast.ClassDef)}
    rows = []

    def emit(owner, func, default):
        checked = validated_params(func)
        params = params_of(func) or ["(none)"]
        for p in params:
            dim = PARAM_RULES.get(p) or default
            status = "-" if p == "(none)" else ("validated" if p in checked else "UNVALIDATED")
            rows.append((fname, func.lineno, owner, func.name, p, dim, status))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            default = class_dim(node.name, all_bases[node.name], all_bases)
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and (m.name in PUBLIC_METHODS or not m.name.startswith("_")):
                    emit(node.name, m, default)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            emit("(module)", node, CLASS_RULES.get(node.name, "?"))
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pkg = sys.argv[1]
    rows = []
    for f in FILES:
        rows += rows_for(os.path.join(pkg, f))

    header = ("file", "line", "class", "method", "parameter", "dimension", "validation")
    if "--csv" in sys.argv:
        out = sys.argv[sys.argv.index("--csv") + 1]
        with open(out, "w", newline="") as fh:
            csv.writer(fh).writerows([header, *rows])
        print(f"wrote {len(rows)} rows to {out}")

    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))

    print("\nParameters per dimension:")
    counts = {}
    for r in rows:
        for d in r[5].split("+"):
            counts[d.strip()] = counts.get(d.strip(), 0) + 1
    for d in sorted(counts):
        print(f"  {d:16} {counts[d]}")

    unknown = [r for r in rows if "?" in r[5]]
    if unknown:
        print("\nNo rule matched (add to CLASS_RULES or PARAM_RULES):")
        for r in unknown:
            print(f"  {r[2]}.{r[3]}({r[4]})")


if __name__ == "__main__":
    main()
