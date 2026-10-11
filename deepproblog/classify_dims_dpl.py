#!/usr/bin/env python3
"""Auto-classify DeepProbLog's public API into fuzzing dimensions P1-P7.

Usage:
    python3 classify_dims_dpl.py <path-to-deepproblog-package>            # table on screen
    python3 classify_dims_dpl.py <path-to-deepproblog-package> --csv out.csv

How it works (static analysis with Python's `ast`, DeepProbLog is never imported):
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

FILES = ["model.py", "network.py", "query.py", "solver.py", "optimizer.py", "dataset.py",
         "arithmetic_circuit.py", "semiring/__init__.py", "semiring/graph_semiring.py",
         "engines/exact_engine.py", "engines/builtins.py", "train.py", "evaluate.py", "tensor.py", "embeddings.py",
         "utils/__init__.py", "utils/cache.py", "utils/logger.py", "utils/confusion_matrix.py", "utils/stop_condition.py"]

# --- rule table 1: class / function name -> default dimension ---
#  P1 logic program   P2 probability values   P3 neural predicates
#  P4 query           P5 solver / semiring     P6 training / gradients
#  P7 API boundary (state, datasets, foreign functions)
CLASS_RULES = {
    "Model": "P1", "Query": "P4", "Network": "P3", "Solver": "P5",
    "ArithmeticCircuit": "P5", "Semiring": "P5", "GraphSemiring": "P5", "Result": "P5",
    "Optimizer": "P6", "SGD": "P6", "TrainObject": "P6",
    "Dataset": "P7", "DataLoader": "P7", "Subset": "P7", "Extension": "P7", "Fold": "P7",
    "MutatingDataset": "P7", "ImageDataset": "P7", "QueryDataset": "P7",
    "ExactEngine": "P5", "Engine": "P5",
    "is_exact_zero": "internal", "is_exact_one": "internal", "get_hook": "internal",
    "wrap_tensor": "internal", "create_with_substitution": "internal",
    "get_predicate": "P3", "get_det_predicate": "P3", "get_tensor_function": "P3",
    "tensor_wrapper": "P7", "tensor_wrapper_nondet": "P7", "to_tensor": "P7", "embed": "P7",
    "TensorStore": "P3", "TermEmbedder": "P3", "OneHotEncoding": "P3", "Embedding": "P3",
    "Cache": "P5", "Logger": "P7", "ConfusionMatrix": "P7", "Table": "P7", "TabularFormatter": "P7",
    "StopCondition": "P6", "EpochStop": "P6", "StopOnPlateau": "P6", "StopOnNoChange": "P6",
    "Or": "P6", "And": "P6", "Threshold": "P6",
    "log_exists": "P7", "check_path": "P7", "get_top_path": "P7", "split": "P7", "count_parameters": "P7",
    "nth": "P7", "get_configuration": "P7", "parse": "P1", "format_time": "P7", "format_time_precise": "P7",
    "term2list2": "P1", "config_to_string": "P7", "equalize": "P7", "aggregate": "P7", "aggregate_loggers": "P7",
    "register_tensor_predicates": "P7", "SolverException": "P5", "NoiseMutatorDecorator": "P7",
    "NumericIncorrectOutputMutator": "P7", "cat": "P7", "stack": "P7", "list_to_tensor": "P7",
    "tensor_index": "P7", "less_than": "P7", "load": "P7", "get_fact_accuracy": "P6", "train_model": "P6", "get_confusion_matrix": "P6",
}

# --- rule table 2: parameter name -> dimension (overrides the class default) ---
PARAM_RULES = {
    "program_string": "P1", "load": "P1", "networks": "P3", "embeddings": "P3",
    "network_module": "P3", "name": "P3", "k": "P3", "batching": "P3",
    "optimizer": "P6", "scheduler": "P6", "param_lr": "P6",
    "query": "P4", "substitution": "P4", "output_ind": "P4", "new_values": "P4", "batch": "P4",
    "p": "P2", "target": "P2", "eps": "P2", "weight": "P6",
    "semiring": "P5", "cache": "P5", "cache_root": "P5", "sdd_auto_gc": "P5",
    "engine": "P5", "result": "P5", "q": "P4",
    "filename": "P7", "complete": "P7", "dataset": "P7", "batch_size": "P7",
    "shuffle": "P7", "seed": "P7", "func": "P7", "function_name": "P7",
    "arity_in": "P7", "arity_out": "P7", "to_evaluate": "P3", "i": "P7", "j": "P7",
    "n": "P7", "stop_criterion": "P6", "loss_function_name": "P6", "with_negatives": "P6",
    "model": None, "verbose": "P6", "threshold": "P2", "empty_answer": "P6", "max_size": "P7",
    "tensors": "P7", "list_term": "P7", "index": "P7", "tensor": "P7", "tensor1": "P7", "tensor2": "P7",
    "inner_mutator": "P7", "domain": "P3", "change_p": "P2", "args": "P5", "values": "P3", "a": "P2", "b": "P2", "z": "P2", "x": "P3", "y": "P3",
}

# --- rule table 3: DeepProbLog's own validation helpers ---
VALIDATORS = {"check_path"}

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
        if r[4] == "(none)":                       # methods without parameters are not counted
            continue
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
