#!/usr/bin/env python3
"""
Hybrid libFuzzer (Atheris) harness for LTNtorch: raw float32 data + a tiny byte "header" that picks structure.

ltn_raw_fuzz.py can only call fuzzy_ops classes, because nothing in raw bytes can choose a Predicate, a
Quantifier or a formula. This harness keeps the raw data (bytes reinterpreted as float32, so NaN, inf, denormals
and huge values all appear) but reads the first 7 bytes as a header of small choices:

    byte 0   target        which part of LTN to exercise (see TARGETS)
    byte 1   operator      which connective / fuzzy_ops class
    byte 2   aggregator    which aggregation operator
    byte 3   p and flags   p of the p-mean, quantifier f/e, guarded quantification (stable is always True)
    byte 4   predicate     which predicate function
    byte 5   shape         feature width of the individuals (1..3), 1-D vs 2-D
    byte 6   spare         second operator
    rest     the data      float32 values, shared out between the variables

Targets, and what they reach in ltn/core.py and ltn/fuzzy_ops.py:
    0 ops          every fuzzy_ops class called directly with raw tensors (as ltn_raw_fuzz.py), incl. Equiv
    1 connective   Connective(op) over predicate outputs
    2 quantifier   Quantifier(agg, 'f'|'e'), plain and guarded
    3 predfunc     Predicate(func=...) and Function(func=...), arity 1 and 2, on Variables and Constants
    4 diag         diag / undiag
    5 formula      Forall(x, Implies(P(x), Exists(y, Q(x, y)))) closed with SatAgg, plus a value oracle

Oracles:
  - an exception that is not one of LTN's own validation errors
  - value oracle (target 5 only): a closed formula or SatAgg result that is NaN or outside [0, 1] while every
    input float was finite

Run:   python ltn_hybrid_fuzz.py corpus -max_total_time=300 -rss_limit_mb=4000
Trace: TRACE_CSV=trace.csv python ltn_hybrid_fuzz.py corpus -runs=0     (replays the corpus once, one CSV row per call;
       also works on a single crash file). Off unless TRACE_CSV is set, because it slows the run down.
"""
import atexit
import csv
import hashlib
import json
import os
import re
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "deps"))
sys.path.insert(0, os.environ.get("LTN_SRC", os.path.join(HERE, "ltn_torch")))

import atheris  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

if os.environ.get("HYBRID_NO_INSTRUMENT") == "1":      # plain import, used by cov_hybrid.py (coverage.py replay)
    import ltn  # noqa: E402
    import ltn.fuzzy_ops as fo  # noqa: E402
else:
    with atheris.instrument_imports(include=["ltn"]):
        import ltn  # noqa: E402
        import ltn.fuzzy_ops as fo  # noqa: E402

FINDINGS = os.environ.get("RAW_FINDINGS", os.path.join(HERE, "findings_hybrid"))
os.makedirs(FINDINGS, exist_ok=True)
SEEN = set()
HEADER = 7
MAX_FLOATS = 64                       # keep each run fast

EXPECTED_PREFIXES = (
    "Expected the output of a predicate", "Expected inputs of connectives and quantifiers",
    "Expected one operand", "Expected two operands", "Expected parameter",
    "Expected the given LTN variables", "LTNObject()", "Variable()", "Predicate()", "Function()",
    "Connective()", "Quantifier()", "SatAgg()", "Labels starting with 'diag_'", "Both model and func",
    "Since 'cond_", "'xs' and 'mask' must have the same shape",
)


class OracleViolation(Exception):
    pass


def expected(exc):
    msg = str(exc)
    if isinstance(exc, TypeError) and "__call__()" in msg and (
            "positional argument" in msg or "unexpected keyword" in msg):
        return True                       # the harness called an operator with the wrong arity (e.g. unary op with 2 args)
    return isinstance(exc, (ValueError, TypeError)) and msg.startswith(EXPECTED_PREFIXES)


def signature(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if os.sep + "ltn" + os.sep in f.filename]
    fr = frames[-1] if frames else (tb[-1] if tb else None)
    where = f"{os.path.basename(fr.filename)}:{fr.lineno}" if fr else "?"
    return f"{type(exc).__name__}@{where}: " + re.sub(r"[-+]?\d[\d.e+-]*", "N", str(exc))[:120]


def save(sig, data, label):
    if sig in SEEN:
        return False
    SEEN.add(sig)
    d = os.path.join(FINDINGS, hashlib.sha1(sig.encode()).hexdigest()[:10])
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "input.bin"), "wb").write(data)
    json.dump({"signature": sig, "call": label}, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"\n[NEW FINDING] {label}: {sig}", file=sys.stderr, flush=True)
    return True


# ---------------------------------------------------------------- optional CSV trace (TRACE_CSV=path)
TRACE_PATH = os.environ.get("TRACE_CSV")
TRACE_FIELDS = ["exec", "input_sha1", "input_hex", "header", "target", "call", "choices", "n_floats",
                "inputs_finite", "oracle_applied", "output", "verdict", "exception", "new_finding"]
_trace = {"fh": None, "w": None, "n": 0}


def _trace_row(**row):
    if not TRACE_PATH:
        return
    if _trace["w"] is None:
        new_file = not os.path.exists(TRACE_PATH) or os.path.getsize(TRACE_PATH) == 0
        _trace["fh"] = open(TRACE_PATH, "a", newline="")
        _trace["w"] = csv.DictWriter(_trace["fh"], fieldnames=TRACE_FIELDS)
        if new_file:
            _trace["w"].writeheader()
        atexit.register(lambda: _trace["fh"].close())
    _trace["w"].writerow({k: row.get(k, "") for k in TRACE_FIELDS})
    _trace["fh"].flush()                  # libFuzzer may _exit() on a crash, so do not rely on atexit


def summarize(out):
    """One short line describing what a call returned."""
    v = getattr(out, "value", out)
    if isinstance(v, torch.Tensor):
        t = v.detach().float()
        if t.numel() == 0:
            return f"shape={tuple(t.shape)} (empty)"
        return (f"shape={tuple(t.shape)} min={t.min().item():.6g} max={t.max().item():.6g} "
                f"nan={bool(torch.isnan(t).any())} inf={bool(torch.isinf(t).any())}")
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__} of {len(v)}"
    return type(v).__name__


# ---------------------------------------------------------------- the menus the header bytes pick from
ABSTRACT = ("ConnectiveOperator", "UnaryConnectiveOperator", "BinaryConnectiveOperator", "AggregationOperator")


def _build_menus():
    ops, conns, aggs = [], [], []
    for name in dir(fo):
        cls = getattr(fo, name)
        if not (isinstance(cls, type) and cls.__module__ == fo.__name__):
            continue
        if name in ABSTRACT:              # their __call__ only raises NotImplementedError
            continue
        for kwargs in ({"stable": True}, {}):
            try:
                op = cls(**kwargs)
            except TypeError:
                continue
            ops.append((f"{name}{kwargs or ''}", op))
            if isinstance(op, fo.ConnectiveOperator):
                conns.append((f"{name}{kwargs or ''}", op))
    # Equiv needs two operators: this is what ltn_raw_fuzz.py silently skipped
    for a, i in ((fo.AndProd(), fo.ImpliesReichenbach()), (fo.AndMin(), fo.ImpliesGodel()),
                 (fo.AndLuk(), fo.ImpliesLuk())):
        eq = fo.Equiv(a, i)
        ops.append((f"Equiv({type(a).__name__},{type(i).__name__})", eq))
        conns.append((f"Equiv({type(a).__name__},{type(i).__name__})", eq))
    return ops, conns


OPS, CONNS = _build_menus()
UNARY = [c for c in CONNS if isinstance(c[1], fo.UnaryConnectiveOperator)]
BINARY = [c for c in CONNS if not isinstance(c[1], fo.UnaryConnectiveOperator)]


def make_agg(h):
    p = h % 9 - 3                     # p in -3..5 (0 and negatives on purpose)
    stable = True                     # stable variants only
    k = (h >> 5) % 4
    if k == 0:
        return f"AggregMin", fo.AggregMin()
    if k == 1:
        return f"AggregMean", fo.AggregMean()
    if k == 2:
        return f"AggregPMean(p={p},stable={stable})", fo.AggregPMean(p=p, stable=stable)
    return f"AggregPMeanError(p={p},stable={stable})", fo.AggregPMeanError(p=p, stable=stable)


PRED1 = [lambda x: torch.sigmoid(x.sum(-1)),
         lambda x: x.mean(-1).clamp(0, 1),
         lambda x: (x.sum(-1) > 0).float(),
         lambda x: torch.sigmoid(x).prod(-1)]
PRED2 = [lambda x, y: torch.sigmoid((x * y).sum(-1)),
         lambda x, y: (x - y).abs().mean(-1).clamp(0, 1),
         lambda x, y: (x.sum(-1) > y.sum(-1)).float(),
         lambda x, y: torch.sigmoid(x).prod(-1) * torch.sigmoid(y).prod(-1)]
FUNCS = [lambda x: x * 0.5, lambda x: torch.tanh(x), lambda x: -x, lambda x: x.flip(-1)]


# ---------------------------------------------------------------- data
class Ctx:
    """Everything one input decodes to."""

    def __init__(self, data):
        h = data[:HEADER]
        self.h = h
        self.info = {}                    # what the target picked, filled in by the target (goes to the CSV)
        raw = data[HEADER:][: MAX_FLOATS * 4]
        self.t = torch.from_numpy(np.frombuffer(raw[: len(raw) // 4 * 4], dtype=np.float32).copy())
        self.feat = 1 + h[5] % 3
        self.oned = bool((h[5] >> 2) & 1)
        self.finite = bool(torch.isfinite(self.t).all())
        half = self.t.numel() // 2
        self.ta, self.tb = (self.t[:half], self.t[half:2 * half]) if half else (self.t, self.t)

    def var(self, label, t=None, force2d=False):
        t = self.t if t is None else t
        if self.oned and not force2d:
            return ltn.Variable(label, t)
        n = t.numel() // self.feat
        return ltn.Variable(label, t[: n * self.feat].reshape(n, self.feat))

    def pred1(self):
        return ltn.Predicate(func=PRED1[self.h[4] % len(PRED1)])

    def pred2(self):
        return ltn.Predicate(func=PRED2[(self.h[4] >> 2) % len(PRED2)])


# ---------------------------------------------------------------- targets
def t_ops(c):
    name, op = OPS[c.h[1] % len(OPS)]
    c.info.update(op=name)
    a, b = c.ta, c.tb
    return [(f"{name}(a)", lambda: op(a)),
            (f"{name}(a, b)", lambda: op(a, b)),
            (f"{name}(a, dim=0)", lambda: op(a, dim=0)),
            (f"SatAgg({name})(a)", lambda: fo.SatAgg(op)(a))]


def t_connective(c):
    name, op = (BINARY + UNARY)[c.h[1] % len(BINARY + UNARY)]
    C = ltn.Connective(op)
    c.info.update(connective=name, pred1=c.h[4] % len(PRED1), second_pred=c.h[6] % 4)
    x = c.var("x")
    P, Q = c.pred1(), c.pred1()
    return [(f"Connective({name})(P(x))", lambda: C(P(x))),
            (f"Connective({name})(P(x), P(x))", lambda: C(P(x), P(x))),
            (f"Connective({name})(P(x), Q(x))", lambda: C(P(x), ltn.Predicate(func=PRED1[(c.h[6]) % 4])(x)))]


def t_quantifier(c):
    aname, agg = make_agg(c.h[3] if c.h[2] % 2 else c.h[2])
    q = "f" if (c.h[3] >> 7) & 1 else "e"
    Qn = ltn.Quantifier(agg, q)
    c.info.update(aggregator=aname, quantifier=q, pred1=c.h[4] % len(PRED1), pred2=(c.h[4] >> 2) % len(PRED2))
    x, y = c.var("x"), c.var("y", c.tb)
    P, R = c.pred1(), c.pred2()
    calls = [(f"Quantifier({aname},{q})(x, P(x))", lambda: Qn(x, P(x))),
             (f"Quantifier({aname},{q})([x,y], R(x,y))", lambda: Qn([x, y], R(x, y))),
             (f"Quantifier({aname},{q})(x, P(x), guarded)",
              lambda: Qn(x, P(x), cond_vars=[x], cond_fn=lambda v: v.value.sum(-1) > 0)),
             (f"Quantifier({aname},{q}) bad quantifier", lambda: ltn.Quantifier(agg, "x"))]
    if isinstance(agg, (fo.AggregPMean, fo.AggregPMeanError)):      # only these accept p / stable per call
        calls.append((f"Quantifier({aname},{q})(x, P(x), p=2 per call)", lambda: Qn(x, P(x), p=2, stable=True)))
    return calls


def t_predfunc(c):
    x, y = c.var("x"), c.var("y", c.tb)
    const = ltn.Constant(c.ta[: c.feat] if c.ta.numel() >= c.feat else c.ta, trainable=bool(c.h[3] & 1))
    P, R = c.pred1(), c.pred2()
    F = ltn.Function(func=FUNCS[c.h[4] % len(FUNCS)])
    c.info.update(pred1=c.h[4] % len(PRED1), pred2=(c.h[4] >> 2) % len(PRED2), func=c.h[4] % len(FUNCS),
                  trainable_constant=bool(c.h[3] & 1))
    return [("Predicate(x)", lambda: P(x)),
            ("Predicate(c)", lambda: P(const)),
            ("Predicate2(x, y)", lambda: R(x, y)),
            ("Predicate2(x, c)", lambda: R(x, const)),
            ("Function(x)", lambda: F(x)),
            ("Predicate(Function(x))", lambda: P(F(x))),
            ("Function(x) chained", lambda: F(F(x))),
            ("Predicate(func and model)", lambda: ltn.Predicate(model=torch.nn.Identity(), func=PRED1[0]))]


def t_diag(c):
    x, y, z = c.var("x", c.ta), c.var("y", c.tb), c.var("z")      # x and y same length, z usually different
    R = c.pred2()
    P = c.pred1()
    c.info.update(pred1=c.h[4] % len(PRED1), pred2=(c.h[4] >> 2) % len(PRED2))
    return [("diag(x, y)", lambda: ltn.diag(x, y)),
            ("diag(x, z) length mismatch", lambda: ltn.diag(x, z)),
            ("R(diag(x,y))", lambda: R(*ltn.diag(x, y))),
            ("undiag(diag(x,y))", lambda: ltn.undiag(*ltn.diag(x, y))),
            ("P(x) after diag", lambda: (ltn.diag(x, y), P(x))),
            ("diag(x)", lambda: ltn.diag(x))]


def t_formula(c):
    aname, agg = make_agg(c.h[2])
    bname, bagg = make_agg(c.h[3])
    imp = ltn.Connective(BINARY[c.h[1] % len(BINARY)][1])
    notc = ltn.Connective(UNARY[c.h[6] % len(UNARY)][1])
    Forall, Exists = ltn.Quantifier(agg, "f"), ltn.Quantifier(bagg, "e")
    c.info.update(forall_agg=aname, exists_agg=bname, binary_connective=BINARY[c.h[1] % len(BINARY)][0],
                  unary_connective=UNARY[c.h[6] % len(UNARY)][0], pred1=c.h[4] % len(PRED1),
                  pred2=(c.h[4] >> 2) % len(PRED2))
    x, y = c.var("x"), c.var("y", c.tb)
    P, R = c.pred1(), c.pred2()

    def build():
        f = Forall(x, imp(P(x), Exists(y, R(x, y))))
        g = Exists(x, notc(P(x)))
        sat = fo.SatAgg(bagg)(f, g)
        if c.finite:
            for label, v in (("formula", f), ("neg-exists", g), ("SatAgg", sat)):
                val = v.value if hasattr(v, "value") else v
                if not torch.isfinite(val).all() or (val < 0).any() or (val > 1).any():
                    raise OracleViolation(f"{label} = {val.tolist()} outside [0,1] "
                                          f"({aname}, {bname}, {BINARY[c.h[1] % len(BINARY)][0]})")
        return sat
    return [("formula+SatAgg", build)]


TARGETS = [t_ops, t_connective, t_quantifier, t_predfunc, t_diag, t_formula]


def TestOneInput(data):
    if len(data) < HEADER + 4:
        return
    _trace["n"] += 1
    base = dict(exec=_trace["n"], input_sha1=hashlib.sha1(data).hexdigest()[:12], input_hex=data[:96].hex(),
                header=list(data[:HEADER]), target=TARGETS[data[0] % len(TARGETS)].__name__)
    try:
        c = Ctx(data)
        attempts = TARGETS[data[0] % len(TARGETS)](c)
    except Exception as exc:  # noqa: BLE001  (building the menu objects failed)
        bad = not expected(exc)
        new = save(signature(exc), data, "setup") if bad else False
        _trace_row(**base, call="setup", oracle_applied="exception",
                   verdict="FAIL: unexpected exception" if bad else "expected validation error",
                   exception=f"{type(exc).__name__}: {str(exc)[:200]}", new_finding=new)
        return
    choices = dict(c.info, feat=c.feat, one_dim=c.oned)
    # which oracle can fire: the exception oracle always; the value oracle only for the closed formula on finite input
    oracle = "exception+value[0,1]" if (base["target"] == "t_formula" and c.finite) else "exception"
    base.update(choices=json.dumps(choices, default=str), n_floats=c.t.numel(), inputs_finite=c.finite,
                oracle_applied=oracle)
    for label, fn in attempts:
        out_s, exc_s, new = "", "", False
        try:
            out_s, verdict = summarize(fn()), "pass"
        except Exception as exc:  # noqa: BLE001
            exc_s = f"{type(exc).__name__}: {str(exc)[:200]}"
            if isinstance(exc, OracleViolation):
                verdict = "FAIL: value outside [0,1] or NaN"
            elif expected(exc):
                verdict = "expected validation error"
            else:
                verdict = "FAIL: unexpected exception"
            if verdict.startswith("FAIL"):
                new = save(signature(exc), data, label)
        _trace_row(**base, call=label, output=out_s, verdict=verdict, exception=exc_s, new_finding=new)


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
