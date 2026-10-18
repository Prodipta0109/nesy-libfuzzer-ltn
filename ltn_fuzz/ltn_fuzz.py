"""
Fuzzing dimensions (each is a group of choices read from the bytes):
  D1  data values        individuals' feature values incl. 0, 1, eps, NaN, +-inf, huge
  D2  data shape         number of individuals per variable (0 allowed), feature width,
                         1-D vs 2-D individuals, constants
  D3  predicate model    sigmoid / crisp step / clamp, arity 1 or 2, weights
  D4  formula structure  Not / And / Or / Implies / Equiv trees, operator family, depth,
                         function terms f(x)
  D5  quantification     Forall / Exists, aggregator (Min, Mean, pMean, pMeanError),
                         p value, stable flag, diag, guarded (masked) quantification
  D6  satisfaction       SatAgg operator, number of closed formulas, backward pass


Oracle:
  - any exception that is NOT one of LTN's own documented input-validation errors
  - value oracle: a closed formula / SatAgg result that is NaN or outside [0, 1]
    while every input value was finite
  - gradient oracle: NaN/inf gradient while every input was finite and only the
    "stable" variants of the operators were used (LTN documents those as the
    fix for gradient problems)
"""
import hashlib
import json
import math
import os
import re
import sys
import traceback

import atheris

# --- import torch WITHOUT instrumentation (too large, and not our target) ---
import torch

torch.set_num_threads(1)
torch.manual_seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
LTN_SRC = os.environ.get("LTN_SRC", os.path.join(HERE, "..", "ltn_torch"))
sys.path.insert(0, os.path.abspath(LTN_SRC))

# --- instrument ONLY the ltn package so coverage reflects LTN code ---
if os.environ.get("LTNFUZZ_NO_INSTRUMENT") == "1":   # plain replay / triage
    import ltn
    import ltn.fuzzy_ops as fo
else:
    with atheris.instrument_imports(include=["ltn"]):
        import ltn
        import ltn.fuzzy_ops as fo

ltn.device = torch.device("cpu")

MISUSE = os.environ.get("LTNFUZZ_MISUSE") == "1"
COLLECT = os.environ.get("LTNFUZZ_COLLECT") == "1"
FINDINGS = os.environ.get("LTNFUZZ_FINDINGS", os.path.join(HERE, "findings"))

# ---------------------------------------------------------------- oracle ----
# Messages raised by LTN's own validation code. Exceptions with these prefixes
# mean "LTN correctly rejected the input" and are NOT findings.
EXPECTED_PREFIXES = (
    "Expected the output of a predicate",
    "Expected inputs of connectives and quantifiers",
    "Expected one operand", "Expected two operands",
    "Expected parameter", "Expected the given LTN variables",
    "LTNObject()", "Variable()", "Predicate()", "Function()",
    "Connective()", "Quantifier()", "SatAgg()",
    "Labels starting with 'diag_'", "Both model and func",
    "Since 'cond_", "'xs' and 'mask' must have the same shape",
)


class OracleViolation(Exception):
    pass


class GradientViolation(Exception):
    pass


def is_expected(exc):
    if isinstance(exc, (ValueError, TypeError)):
        msg = str(exc)
        return msg.startswith(EXPECTED_PREFIXES)
    return False


def signature(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    ltn_frames = [f for f in tb if os.sep + "ltn" + os.sep in f.filename]
    frame = ltn_frames[-1] if ltn_frames else (tb[-1] if tb else None)
    where = f"{os.path.basename(frame.filename)}:{frame.lineno}" if frame else "?"
    msg = re.sub(r"[-+]?\d[\d.e+-]*", "N", str(exc))[:160]
    return f"{type(exc).__name__}@{where}: {msg}"


# --------------------------------------------------------- program builder --
def _flat(x):
    """[n, ...] -> [n, k]; safe for n == 0 (reshape(0, -1) is ambiguous)."""
    return x.flatten(1) if x.dim() > 1 else x.reshape(-1, 1)


SPECIALS = [0.0, 1.0, -1.0, 0.5, 1e-4, 1 - 1e-4, 1e-8, 1e30, -1e30,
            float("nan"), float("inf"), float("-inf"), 1.4e-45, 88.0, -88.0, 3.4e38]


class Builder:
    """Reads choices from the fuzzer bytes and builds + runs one LTN program."""

    def __init__(self, data):
        self.fdp = atheris.FuzzedDataProvider(data)
        self.trace = []           # human readable program, for triage
        self.all_finite = True    # D1: were all input numbers finite?
        self.all_stable = True    # D5: only stable operator variants used?
        self.params = []          # trainable tensors (for the gradient oracle)
        self.vars = {}            # label -> ltn.Variable
        self.consts = []
        self.preds = []
        self.funcs = []
        self.empty_domain = False  # D2: some variable has 0 individuals
        self.aggs = set()          # D5: aggregators used
        self.truth = []           # predicate outputs (leaves of the logic layer)

    # -- primitive choices --
    def i(self, lo, hi):
        return self.fdp.ConsumeIntInRange(lo, hi)

    def b(self):
        return self.fdp.ConsumeBool()

    def num(self):
        if self.i(0, 3) == 0:
            v = SPECIALS[self.i(0, len(SPECIALS) - 1)]
        else:
            v = self.fdp.ConsumeFloatInRange(-10.0, 10.0)
        if not math.isfinite(v):
            self.all_finite = False
        return v

    def weight(self):
        if self.i(0, 7) == 0:
            v = SPECIALS[self.i(0, len(SPECIALS) - 1)]
        else:
            v = self.fdp.ConsumeFloatInRange(-5.0, 5.0)
        if not math.isfinite(v):
            self.all_finite = False
        return v

    def log(self, s):
        self.trace.append(s)

    # -- D2: domain --
    def make_domain(self):
        n_vars = self.i(1, 3)
        for k in range(n_vars):
            label = "xyz"[k]
            n = self.i(0, 5)                     
            self.empty_domain |= n == 0
            one_d = self.i(0, 4) == 0             
            feat = 1 if one_d else self.i(1, 3)
            vals = [self.num() for _ in range(n * feat)]
            t = torch.tensor(vals, dtype=torch.float32)
            t = t if one_d else t.reshape(n, feat)
            self.vars[label] = ltn.Variable(label, t)
            self.log(f"{label} = ltn.Variable('{label}', torch.tensor({t.tolist()}))")
        for k in range(self.i(0, 1)):
            feat = self.i(1, 3)
            t = torch.tensor([self.num() for _ in range(feat)], dtype=torch.float32)
            c = ltn.Constant(t)
            self.consts.append(c)
            self.log(f"c{k} = ltn.Constant(torch.tensor({t.tolist()}))")


    def make_predicate(self, idx):
        kind = self.i(0, 2)
        arity = self.i(1, 2)
        w = torch.tensor([self.weight() for _ in range(arity)], requires_grad=True)
        bias = torch.tensor(self.weight(), requires_grad=True)
        self.params += [w, bias]

        def score(*xs):
            s = bias
            for j, x in enumerate(xs):
                s = s + w[j] * _flat(x).sum(dim=1)
            return s

        if kind == 0:
            f = lambda *xs: torch.sigmoid(score(*xs))
            name = "sigmoid"
        elif kind == 1:
            f = lambda *xs: (score(*xs) > 0).float()
            name = "step"
        else:
            f = lambda *xs: torch.clamp(score(*xs), 0.0, 1.0)
            name = "clamp"
        p = ltn.Predicate(func=f)
        self.preds.append((p, arity))
        self.log(f"P{idx} = Predicate({name}, arity={arity}, w={w.tolist()}, b={bias.item()})")

    def make_function(self, idx):
        a = torch.tensor(self.weight(), requires_grad=True)
        self.params.append(a)
        f = ltn.Function(func=lambda x: x * a)
        self.funcs.append(f)
        self.log(f"f{idx} = Function(lambda x: x * {a.item()})")

    
    def term(self, depth=0):
        labels = list(self.vars)
        choice = self.i(0, 9)
        if choice == 0 and self.consts:
            k = self.i(0, len(self.consts) - 1)
            return self.consts[k], f"c{k}"
        if choice == 1 and self.funcs and depth < 2:
            k = self.i(0, len(self.funcs) - 1)
            t, s = self.term(depth + 1)
            return self.funcs[k](t), f"f{k}({s})"
        lab = labels[self.i(0, len(labels) - 1)]
        return self.vars[lab], lab

    def atom(self):
        k = self.i(0, len(self.preds) - 1)
        p, arity = self.preds[k]
        args = [self.term() for _ in range(arity)]
        out = p(*[a for a, _ in args])
        if out.value.requires_grad:
            out.value.retain_grad()
            self.truth.append(out.value)
        return out, f"P{k}({', '.join(s for _, s in args)})"

    def connective(self, binary):
        if not binary:
            ops = [("NotStandard", fo.NotStandard()), ("NotGodel", fo.NotGodel())]
        else:
            fam = self.i(0, 3)
            stable = self.b()
            if not stable and fam in (0, 1, 2):
                self.all_stable = False
            if fam == 0:
                ops = [("AndMin", fo.AndMin()), ("AndProd", fo.AndProd(stable)),
                       ("AndLuk", fo.AndLuk())]
            elif fam == 1:
                ops = [("OrMax", fo.OrMax()), ("OrProbSum", fo.OrProbSum(stable)),
                       ("OrLuk", fo.OrLuk())]
            elif fam == 2:
                ops = [("ImpliesKleeneDienes", fo.ImpliesKleeneDienes()),
                       ("ImpliesGodel", fo.ImpliesGodel()),
                       ("ImpliesReichenbach", fo.ImpliesReichenbach(stable)),
                       ("ImpliesGoguen", fo.ImpliesGoguen(stable)),
                       ("ImpliesLuk", fo.ImpliesLuk())]
            else:
                and_op = [fo.AndMin(), fo.AndProd(), fo.AndLuk()][self.i(0, 2)]
                imp = [fo.ImpliesKleeneDienes(), fo.ImpliesGodel(), fo.ImpliesReichenbach(),
                       fo.ImpliesGoguen(), fo.ImpliesLuk()][self.i(0, 4)]
                ops = [(f"Equiv({and_op!r},{imp!r})", fo.Equiv(and_op, imp))]
        name, op = ops[self.i(0, len(ops) - 1)]
        if "stable=False" in repr(op):
            self.all_stable = False
        return ltn.Connective(op), name

    def aggregator(self):
        k = self.i(0, 3)
        self.aggs.add(["Min", "Mean", "PMean", "PMeanError"][k])
        if k == 0:
            return fo.AggregMin(), "AggregMin()"
        if k == 1:
            return fo.AggregMean(), "AggregMean()"
        p = [1, 2, 3, 6, 10, 0, -1, 0.5][self.i(0, 7)]
        stable = self.b()
        if not stable:
            self.all_stable = False
        cls = fo.AggregPMean if k == 2 else fo.AggregPMeanError
        return cls(p=p, stable=stable), f"{cls.__name__}(p={p}, stable={stable})"

    def quantified(self, depth):
        q = "f" if self.b() else "e"
        agg, agg_s = self.aggregator()
        Q = ltn.Quantifier(agg, quantifier=q)
        qname = "Forall" if q == "f" else "Exists"

        # D5: diagonal quantification over two vars of equal size
        labels = list(self.vars)
        if len(labels) >= 2 and self.i(0, 5) == 0:
            a, b = labels[0], labels[1]
            if self.vars[a].shape()[0] == self.vars[b].shape()[0]:
                binp = [k for k, (_, ar) in enumerate(self.preds) if ar == 2]
                if binp:
                    k = binp[self.i(0, len(binp) - 1)]
                    dv = ltn.diag(self.vars[a], self.vars[b])
                    body = self.preds[k][0](self.vars[a], self.vars[b])
                    out = Q(dv, body)
                    return out, f"{qname}(ltn.diag({a},{b}), P{k}({a},{b}), {agg_s})"

        body, body_s = self.formula(depth + 1)
        free = list(body.free_vars)
        if MISUSE and self.i(0, 9) == 0:          # D7: vacuous quantification
            qvars = [self.vars[labels[self.i(0, len(labels) - 1)]]]
        elif not free:
            return body, body_s
        else:
            n = self.i(1, len(free))
            qvars = [self.vars[v] for v in free[:n]]
        qlabels = [v.latent_var for v in qvars]

        if self.i(0, 3) == 0:                     # D5: guarded quantification
            cond_lab = (free or labels)[self.i(0, len(free or labels) - 1)]
            thr = self.num()
            cond_fn = lambda x, thr=thr: torch.lt(_flat(x.value).sum(dim=1), thr)
            out = Q(qvars, body, cond_vars=[self.vars[cond_lab]], cond_fn=cond_fn)
            return out, (f"{qname}({qlabels}, {body_s}, cond_vars=[{cond_lab}], "
                         f"cond_fn=sum<{thr}, {agg_s})")
        out = Q(qvars, body)
        return out, f"{qname}({qlabels}, {body_s}, {agg_s})"

    def formula(self, depth=0):
        choice = self.i(0, 9) if depth < 3 else 0
        if choice <= 2:
            return self.atom()
        if choice == 3:
            c, n = self.connective(binary=False)
            f, s = self.formula(depth + 1)
            return c(f), f"{n}({s})"
        if choice <= 6:
            c, n = self.connective(binary=True)
            f, s1 = self.formula(depth + 1)
            g, s2 = self.formula(depth + 1)
            return c(f, g), f"{n}({s1}, {s2})"
        return self.quantified(depth)

    def close(self, f, s):
        """Wrap remaining free variables in a universal quantifier."""
        if f.free_vars:
            agg, agg_s = self.aggregator()
            Q = ltn.Quantifier(agg, quantifier="f")
            vs = [self.vars[v] for v in list(f.free_vars) if v in self.vars]
            f = Q(vs, f)
            s = f"Forall({[v.latent_var for v in vs]}, {s}, {agg_s})"
        return f, s

    def run(self):
        self.make_domain()
        for k in range(self.i(1, 2)):
            self.make_predicate(k)
        for k in range(self.i(0, 1)):
            self.make_function(k)

        closed = []
        for k in range(self.i(1, 2)):
            f, s = self.formula()
            if not (MISUSE and self.i(0, 9) == 0):   # D7: SatAgg on open formula
                f, s = self.close(f, s)
            self.log(f"phi{k} = {s}")
            v = f.value
            if self.all_finite and (torch.isnan(v).any() or (v < 0).any() or (v > 1).any()):
                raise OracleViolation(f"closed formula has truth value {v.tolist()} outside [0,1] "
                                      f"[empty_domain={self.empty_domain} aggs={sorted(self.aggs)}]")
            closed.append(f)

        agg, agg_s = self.aggregator()
        sat = fo.SatAgg(agg)(*closed)
        self.log(f"sat = SatAgg({agg_s})(phi...) = {sat.item()}")
        if self.all_finite and (torch.isnan(sat) or sat < 0 or sat > 1):
            raise OracleViolation(f"SatAgg returned {sat.item()} with finite inputs")

        if self.b():
            loss = 1.0 - sat
            if loss.requires_grad:
                loss.backward()
                # Check gradients at the predicate outputs (truth values). This
                # isolates LTN's connectives/quantifiers/SatAgg from the harness'
                # own predicate models (which can overflow on huge inputs).
                bad = [t for t in self.truth if t.grad is not None
                       and not torch.isfinite(t.grad).all()]
                self.log(f"loss.backward(); non-finite truth-value grads: {len(bad)}")
                if bad and self.all_finite and self.all_stable:
                    raise GradientViolation(
                        "non-finite gradient with finite inputs and stable operators")


# ---------------------------------------------------------------- driver ----
def _save(sig, data, trace, exc_text):
    h = hashlib.sha1(sig.encode()).hexdigest()[:10]
    d = os.path.join(FINDINGS, h)
    os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        m["count"] += 1
        json.dump(m, open(meta, "w"), indent=1)
        return
    with open(os.path.join(d, "input.bin"), "wb") as fh:
        fh.write(data)
    with open(os.path.join(d, "program.txt"), "w") as fh:
        fh.write("\n".join(trace) + "\n\n" + exc_text)
    json.dump({"signature": sig, "count": 1}, open(meta, "w"), indent=1)
    print(f"\n[NEW FINDING] {sig}", file=sys.stderr, flush=True)


def TestOneInput(data):
    bld = Builder(data)
    try:
        bld.run()
    except Exception as exc: 
        if is_expected(exc):
            return
        if not COLLECT:
            raise
        _save(signature(exc), data, bld.trace, traceback.format_exc())


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
