#!/usr/bin/env python3
"""Auto-classify LNN's public API into fuzzing dimensions L1-L7.

Usage:
    python3 classify_dims_lnn.py <path-to-lnn-package>            # table on screen
    python3 classify_dims_lnn.py <path-to-lnn-package> --csv out.csv

How it works (static analysis with Python's `ast`, LNN is never imported):
  1. Find every public class, method and function in core.py and fuzzy_ops.py.
  2. Give each class a default dimension from its name or base class
     (e.g. anything inheriting BinaryConnectiveOperator -> D4).
  3. Give each parameter a dimension from its name (e.g. `p`, `mask` -> D5);
     if no rule matches, the parameter inherits the class dimension.
  4. Mark a parameter VALIDATED if the method raises an error in an `if` that
     tests it, or passes it to LTN's own check_values / check_mask.
     Unvalidated parameters are where the fuzzer should look first.

To apply this to another library (LNN, LNN, ...), edit the three rule
tables below; the rest of the script stays the same.
"""
import ast
import csv
import os
import sys

FILES = ["model.py", "utils.py", "_utils.py", "constants.py", "symbolic/_gm.py",
         "symbolic/logic/formula.py", "symbolic/logic/leaf_formula.py", "symbolic/logic/connective_neuron.py",
         "symbolic/logic/n_ary_neuron.py", "symbolic/logic/n_ary_operator.py", "symbolic/logic/binary_neuron.py",
         "symbolic/logic/unary_operator.py", "symbolic/logic/variable.py", "symbolic/logic/neural_activation.py",
         "neural/activations/node.py", "neural/activations/neuron/neuron.py", "neural/activations/neuron/dynamic.py",
         "neural/parameters/node.py", "neural/parameters/neuron.py", "neural/methods/lukasiewicz.py",
         "neural/methods/lukasiewicztransparent.py", "optim/constraint_optim.py", "optim/primal_dual.py"]

# --- rule table 1: class / function name -> default dimension ---
#  L1 formula structure (connectives)   L2 truth bounds and data      L3 first-order structure (predicates, quantifiers)
#  L4 neuron configuration (weights, bias, alpha, t-norm)   L5 inference control (direction, steps, query, world)
#  L6 learning (losses, optimizer, parameters)   L7 API boundary (truth tables, printing, naming, utilities)
CLASS_RULES = {
    "Formula": "L1", "_ConnectiveFormula": "L1", "_ConnectiveNeuron": "L1", "_NAryNeuron": "L1", "And": "L1", "Or": "L1",
    "XOr": "L1", "Not": "L1", "Implies": "L1", "Iff": "L1", "Congruent": "L1", "_NAryOperator": "L1",
    "_LeafFormula": "L3", "Proposition": "L1", "Propositions": "L1", "Predicate": "L3", "Predicates": "L3",
    "Variable": "L3", "Variables": "L3", "_Quantifier": "L3", "Exists": "L3", "Forall": "L3",
    "NeuralActivation": "L4", "_NodeActivation": "L4", "_NeuronActivation": "L4", "_StaticActivation": "L4",
    "_DynamicActivation": "L4", "Lukasiewicz": "L4", "LukasiewiczTransparent": "L4",
    "_NodeParameters": "L4", "_NeuronParameters": "L4",
    "Model": "L5", "Direction": "L5", "World": "L5", "Fact": "L2", "Bound": "L2", "Join": "L3", "Loss": "L6",
    "_ConstraintOptim": "L6", "PrimalDual": "L6", "plot_loss": "L7", "plot_params": "L7",
    "split_string_into_groundings": "L7", "alphanumeric_sort": "L7", "get_ternary_table": "L7", "get_n_ary_truth_table": "L7",
    "generate_truth_table": "L7", "return1": "internal", "fact_to_bounds": "L2", "negate_bounds": "internal", "node_state": "L7",
    "dict_rekey": "internal", "val_clamp": "internal", "list_to_str": "L7", "reset_checkpoints": "internal",
    "unpack_checkpoints": "internal", "add_checkpoint": "internal", "average_time": "internal", "logger_setup": "internal",
    "upward_bounds": "L5", "downward_bounds": "L5", "is_grounding_in_bindings": "L3", "tensorise": "internal",
    "bool_and": "L1", "bool_or": "L1", "bool_tensor": "L1", "ConstraintOptimizer": "L6", "ConstraintOptim": "L6",
    "bool_to_fact": "L2", "fact_to_bool": "L2", "truth_table": "L7", "truth_table_dict": "L7",
    "predicate_truth_table": "L7", "pretty_truth_table": "L7",
}

# --- rule table 2: parameter name -> dimension (overrides the class default) ---
PARAM_RULES = {
    "formula": "L1", "formulae": "L1", "operands": "L1", "operand": "L1", "name": "L7", "names": "L7",
    "bounds": "L2", "data": "L2", "facts": "L2", "fact": "L2", "labels": "L2", "lower": "L2", "upper": "L2",
    "arity": "L3", "variables": "L3", "groundings": "L3", "grounding": "L3", "vars": "L3", "var": "L3",
    "activation": "L4", "weights": "L4", "bias": "L4", "alpha": "L4", "weights_learning": "L4", "bias_learning": "L4",
    "direction": "L5", "source": "L5", "max_steps": "L5", "world": "L5", "converge": "L5", "index": "L5",
    "losses": "L6", "optimizer": "L6", "learning_rate": "L6", "epochs": "L6", "parameters": "L6", "pbar": "L6",
    "model": None, "params": "L7", "ax": "L7", "filename": "L7", "kwds": None, "args": None,
}

# --- rule table 3: LNN's own validation helpers (its _exceptions module) ---
VALIDATORS = {n for n in ("AssertWorld", "AssertBoundsBroadcasting", "AssertBoundsType", "AssertTupledBounds",
    "AssertBoundsLen", "AssertBoundsInputs", "AssertBounds", "AssertPropositionalInheritance", "AssertPropositionalType",
    "AssertFormulaInModel", "AssertFormula", "AssertGroundingKeyType", "AssertBindingsInputType", "AssertFOLFacts",
    "AssertDirection", "AssertValidDirection", "AssertDirectionType", "AssertNeuronActivationType", "AssertLossType",
    "AssertBias", "AssertWeights", "AssertAlphaNodeValue", "AssertAlphaNeuronArityValue", "AssertLeafFormulaNaming")}

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
