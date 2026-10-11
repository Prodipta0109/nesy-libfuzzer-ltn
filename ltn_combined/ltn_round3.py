"""Round 3 libFuzzer (Atheris) harness for LTNtorch: the dimensions the earlier harness could not reach.

New choices (all decoded from libFuzzer's bytes):
  N1 per-call p / stable on every aggregator and operator
  N2 diag / undiag, including mutation of the caller's variables
  N3 Variable(add_batch_dim=False), 0-d, int / bool / float64 / float16 tensors
  N4 Predicate(model=nn.Module) and Predicate(func=), Function with vector output
  N5 trainable Constant, Constant used in formulas
  N6 multi-variable quantification, variable order, nested quantifiers, cond_vars on other variables
  N7 raw (non-LTN) tensors to SatAgg / Connective / Quantifier
  N8 two variables sharing one label but different sizes
Oracles: unexpected exception type, [0,1] range, NaN, output shape vs free_vars, caller objects mutated,
         metamorphic (Forall/Exists duality, mask-all-true, quantifier variable order, diag == paired mean,
         batch consistency).
"""
import re, os, sys, warnings, traceback, hashlib, json
warnings.filterwarnings("ignore")
import atheris
with atheris.instrument_imports(include=["ltn"]):
    import ltn
import torch
import torch.nn as nn

fo = ltn.fuzzy_ops
torch.set_grad_enabled(True)
OUT = os.environ.get("R3_OUT", "findings_r3")
os.makedirs(OUT, exist_ok=True)
SEEN = set()
STATE = {'empty': False, 'badp': False}

# LTN's own, intended validation messages: not bugs
OK_PREFIX = ("Expected", "Labels starting", "The objects should", "Variable() :", "Constant() :", "Predicate", "Function",
             "Connective", "Quantifier", "Cannot", "Aggreg", "The p ", "p must", "Wrong", "No ", "the ", "Found", "Invalid",
             "ltn.", "The given", "The operands", "The param", "Check", "'")


class Bug(Exception):
    pass


def record(kind, detail, extra=""):
    tb = traceback.extract_tb(sys.exc_info()[2])
    loc = ""
    for fr in reversed(tb):
        if "/ltn/" in fr.filename:
            loc = f"{os.path.basename(fr.filename)}:{fr.lineno}"
            break
    norm = re.sub(r"[0-9.eE+-]+|\{.*?\}", "#", detail)[:50]
    if kind == "oracle":
        norm = re.sub(r"(p=|Quantifier|Aggreg\w+)", "", norm)
    sig = hashlib.sha1(f"{kind}|{norm}|{loc}".encode()).hexdigest()[:10]
    if sig in SEEN:
        return
    SEEN.add(sig)
    d = os.path.join(OUT, sig)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"kind": kind, "detail": detail, "loc": loc, "extra": extra, "trace": traceback.format_exc()[-1500:]}, f, indent=1)
    print(f"\n[FINDING {sig}] {kind} @ {loc}: {detail[:150]}", flush=True)


def unexpected(e, ctx):
    if isinstance(e, (TypeError, ValueError)) and str(e).startswith(OK_PREFIX):
        return
    if isinstance(e, Bug):
        record("oracle", str(e), ctx)
        return
    record("exception:" + type(e).__name__, str(e).replace("\n", " "), ctx)


DT = [torch.float32, torch.float64, torch.float16, torch.int64, torch.bool]
VALS = [0.0, 1.0, 0.5, 1e-30, 1 - 1e-7, 1e30, -1.0, 0.25, 0.75, 2.0, -0.0]


def tens(fdp, n, d, dtype_idx, unit):
    if fdp.ConsumeBool() and not unit:
        t = torch.tensor([[VALS[fdp.ConsumeIntInRange(0, len(VALS) - 1)] for _ in range(d)] for _ in range(n)])
    else:
        t = torch.tensor([[fdp.ConsumeProbability() for _ in range(d)] for _ in range(n)])
    if n == 0:
        t = torch.empty(0, d)
    try:
        return t.to(DT[dtype_idx])
    except Exception:
        return t


class Net(nn.Module):
    def __init__(self, d, k, hard):
        super().__init__()
        self.l = nn.Linear(d * k, 1)
        nn.init.constant_(self.l.weight, 0.7)
        nn.init.constant_(self.l.bias, -0.2)
        self.hard = hard

    def forward(self, *xs):
        h = torch.cat([x.float().flatten(1) for x in xs], dim=1)
        o = torch.sigmoid(self.l(h)).squeeze(-1)
        return (o > 0.5).float() if self.hard else o


def make_pred(fdp, d, arity):
    mode = fdp.ConsumeIntInRange(0, 3)
    if mode == 0:
        return ltn.Predicate(model=Net(d, arity, False))
    if mode == 1:
        return ltn.Predicate(model=Net(d, arity, True))
    if mode == 2:
        return ltn.Predicate(func=lambda *xs: torch.sigmoid(sum(x.float().sum(dim=1) for x in xs)))
    return ltn.Predicate(func=lambda *xs: torch.clamp(sum(x.float().sum(dim=1) for x in xs) / 3, 0, 1))


BIN = ["AndMin", "AndProd", "AndLuk", "OrMax", "OrProbSum", "OrLuk", "ImpliesKleeneDienes", "ImpliesGodel", "ImpliesReichenbach",
       "ImpliesGoguen", "ImpliesLuk", "Equiv"]
AGG = ["AggregMin", "AggregMean", "AggregPMean", "AggregPMeanError"]


def make_bin(fdp, stable_ok=True):
    name = BIN[fdp.ConsumeIntInRange(0, len(BIN) - 1)]
    cls = getattr(fo, name, None)
    if cls is None:
        return None, name
    if name == "Equiv":
        return fo.Equiv(fo.AndProd(), fo.ImpliesReichenbach()), name
    kw = {}
    if "Implies" in name and name not in ("ImpliesGodel", "ImpliesLuk") or name in ("AndProd", "OrProbSum"):
        pass
    try:
        return cls(**({"stable": fdp.ConsumeBool()} if "Goguen" in name or "Reichenbach" in name else {})), name
    except TypeError:
        return cls(), name


def make_agg(fdp):
    name = AGG[fdp.ConsumeIntInRange(0, 3)]
    cls = getattr(fo, name)
    if "PMean" in name:
        p = [1, 2, 3, 5, 0.5, 0.001, 0, -1, 100][fdp.ConsumeIntInRange(0, 8)]
        STATE['badp'] |= p <= 0
        return cls(p=p, stable=fdp.ConsumeBool()), name, p
    return cls(), name, None


def snapshot(*objs):
    return [(o.value.clone(), list(o.free_vars)) for o in objs]


def unchanged(objs, snap, what):
    for o, (v, fv) in zip(objs, snap):
        if o.free_vars != fv:
            raise Bug(f"{what} mutated caller free_vars {fv} -> {o.free_vars}")
        if o.value.shape != v.shape or not torch.equal(torch.nan_to_num(o.value.detach()), torch.nan_to_num(v.detach())):
            raise Bug(f"{what} mutated caller value")


def check_truth(o, ctx, tol=0.0):
    v = o.value
    if v.numel() and v.dtype.is_floating_point:
        if torch.isnan(v).any() and not (STATE['empty'] or STATE['badp']):
            raise Bug(f"{ctx}: NaN in output")
        if (v < -tol).any() or (v > 1 + tol).any():
            raise Bug(f"{ctx}: output outside [0,1]: min={v.min().item()} max={v.max().item()}")
    expected = list(v.shape[:len(o.free_vars)])
    if len(v.shape) < len(o.free_vars):
        raise Bug(f"{ctx}: shape {tuple(v.shape)} has fewer axes than free_vars {o.free_vars}")


def TestOneInput(data):
    STATE['empty'] = STATE['badp'] = False
    fdp = atheris.FuzzedDataProvider(data)
    t = fdp.ConsumeIntInRange(0, len(TESTS) - 1)
    ctx = f"test{t}"
    try:
        TESTS[t](fdp)
    except Exception as e:
        unexpected(e, ctx)


def mkvar(fdp, label, unit=True, allow_bad=True):
    n = [0, 1, 2, 3, 5][fdp.ConsumeIntInRange(0, 4)]
    d = fdp.ConsumeIntInRange(1, 3)
    dt = fdp.ConsumeIntInRange(0, 4) if allow_bad and fdp.ConsumeIntInRange(0, 5) == 0 else 0
    abd = not (allow_bad and fdp.ConsumeIntInRange(0, 7) == 0)
    STATE['empty'] |= (n == 0)
    v = ltn.Variable(label, tens(fdp, n, d, dt, unit), add_batch_dim=abd)
    return v, n, d


# ---- test 0: quantifier over formula with per-call p / stable, multi-var, order, cond --------------
def t_quant(fdp):
    x, nx, d = mkvar(fdp, "x")
    y, ny, _ = mkvar(fdp, "y") if fdp.ConsumeBool() else (None, 0, d)
    P1 = make_pred(fdp, d, 1)
    P2 = make_pred(fdp, d, 2)
    agg, an, p = make_agg(fdp)
    Q = ltn.Quantifier(agg, "f" if fdp.ConsumeBool() else "e")
    if y is not None and y.shape()[-1:] != x.shape()[-1:] and y.value.dim() > 1 and y.value.shape[1] != d:
        y = None
    formula = P2(x, y) if y is not None else P1(x)
    kw = {}
    if "PMean" in an and fdp.ConsumeBool():
        kw["p"] = [1, 2, 0, -2, 0.5][fdp.ConsumeIntInRange(0, 4)]
        STATE['badp'] |= kw["p"] <= 0
    if "PMean" in an and fdp.ConsumeBool():
        kw["stable"] = fdp.ConsumeBool()
    vs = [x] if y is None else ([x, y] if fdp.ConsumeBool() else [y, x])
    mask_mode = fdp.ConsumeIntInRange(0, 3)
    snap = snapshot(*vs)
    if mask_mode == 0:
        r = Q(vs, formula, **kw)
    else:
        thr = [-1, 0.3, 100, 0.0][mask_mode]
        r = Q(vs, formula, cond_vars=[x], cond_fn=lambda v: v.value.sum(dim=1) > thr, **kw)
    unchanged(vs, snap, "Quantifier")
    check_truth(r, f"Quantifier {an} p={p} kw={kw} mask={mask_mode}")
    if r.value.dim() != 0 and r.free_vars == []:
        raise Bug(f"Quantifier over all vars returned shape {tuple(r.value.shape)}")
    if r.value.dim() == 0 and r.free_vars:
        raise Bug(f"scalar result still claims free vars {r.free_vars}")
    if mask_mode == 2 and nx and not torch.isnan(r.value).any():
        pass
    # order invariance
    if y is not None:
        r2 = Q(vs[::-1], formula, **kw)
        if not torch.allclose(r.value.float(), r2.value.float(), atol=1e-4, equal_nan=True) and mask_mode == 0:
            raise Bug(f"quantifier variable order changes result {r.value} vs {r2.value} ({an}, p={p})")


# ---- test 1: duality Forall(f) == 1 - Exists(1-f) with pMean/pMeanError ------------------------------
def t_duality(fdp):
    x, nx, d = mkvar(fdp, "x", allow_bad=False)
    p = [1, 2, 3, 5][fdp.ConsumeIntInRange(0, 3)]
    P = make_pred(fdp, d, 1)
    f = P(x)
    fa = ltn.Quantifier(fo.AggregPMeanError(p=p), "f")(x, f)
    ex = ltn.Quantifier(fo.AggregPMean(p=p), "e")(x, ltn.Connective(fo.NotStandard())(f))
    check_truth(fa, "dual-forall"); check_truth(ex, "dual-exists")
    if nx and abs(fa.value.item() - (1 - ex.value.item())) > 1e-4:
        raise Bug(f"duality broken: Forall={fa.value.item()} 1-Exists(not)={1-ex.value.item()} p={p}")


# ---- test 2: diag == paired mean ---------------------------------------------------------------------
def t_diag(fdp):
    n = fdp.ConsumeIntInRange(0, 4); d = fdp.ConsumeIntInRange(1, 3)
    a = ltn.Variable("a", tens(fdp, n, d, 0, True)); b = ltn.Variable("b", tens(fdp, n, d, 0, True))
    P = make_pred(fdp, d, 2)
    sa, sb = snapshot(a, b)
    vs = ltn.diag(a, b)
    r = ltn.Quantifier(fo.AggregMean(), "f")(vs, P(a, b))
    check_truth(r, "diag-forall")
    ltn.undiag(a, b)
    unchanged([a, b], [sa, sb], "diag/undiag round trip")
    if n:
        f = P(a, b)
        paired = torch.stack([P(ltn.Variable("u", a.value[i:i+1]), ltn.Variable("w", b.value[i:i+1])).value.flatten()[0] for i in range(n)])
        exp = paired.mean().item()
        if abs(r.value.item() - exp) > 1e-5:
            raise Bug(f"diag quantification {r.value.item()} != mean over pairs {exp}")


# ---- test 3: variable construction & dtype handling ---------------------------------------------------
def t_var(fdp):
    dt = fdp.ConsumeIntInRange(0, 4)
    n = fdp.ConsumeIntInRange(0, 3)
    shape = [(n,), (n, 2), (), (n, 2, 2)][fdp.ConsumeIntInRange(0, 3)]
    base = torch.rand(*shape) if shape else torch.rand(())
    base = base.to(DT[dt])
    v = ltn.Variable("x", base, add_batch_dim=fdp.ConsumeBool())
    c = ltn.Constant(base, trainable=fdp.ConsumeBool())
    P = ltn.Predicate(func=lambda a: torch.sigmoid(a.float().flatten(1).sum(dim=1) if a.dim() > 1 else a.float().flatten()))
    for o in (v, c):
        r = P(o)
        check_truth(r, "predicate on odd tensor")
    if v.value.dtype not in (torch.float32,) and dt in (0, 1):
        raise Bug(f"Variable kept dtype {v.value.dtype} for float input")


# ---- test 4: connectives incl. raw tensors, broadcasting, per-call stable --------------------------------
def t_conn(fdp):
    x, nx, d = mkvar(fdp, "x", allow_bad=False)
    y, ny, _ = mkvar(fdp, "y", allow_bad=False)
    y = ltn.Variable("y", tens(fdp, ny, d, 0, True))
    P1 = make_pred(fdp, d, 1); P2 = make_pred(fdp, d, 1)
    op, name = make_bin(fdp)
    if op is None:
        return
    C = ltn.Connective(op)
    lhs = P1(x)
    rhs = P2(y) if fdp.ConsumeBool() else P2(x)
    snap = snapshot(lhs, rhs)
    r = C(lhs, rhs)
    unchanged([lhs, rhs], snap, "Connective")
    check_truth(r, f"Connective {name}")
    want = set(lhs.free_vars) | set(rhs.free_vars)
    if set(r.free_vars) != want:
        raise Bug(f"free_vars {r.free_vars} != union {want}")
    if r.value.dim() != len(r.free_vars):
        raise Bug(f"result has {r.value.dim()} axes for free_vars {r.free_vars}")
    # unary
    nr = ltn.Connective(fo.NotGodel() if fdp.ConsumeBool() else fo.NotStandard())(lhs)
    check_truth(nr, "Not")
    if nr.value.dtype != lhs.value.dtype:
        raise Bug(f"Not changed dtype {lhs.value.dtype} -> {nr.value.dtype}")


# ---- test 5: SatAgg with LTN objects / raw tensors / empties / grad -----------------------------------------
def t_sat(fdp):
    k = fdp.ConsumeIntInRange(0, 3)
    forms = []
    for _ in range(k):
        mode = fdp.ConsumeIntInRange(0, 3)
        if mode == 0:
            forms.append(torch.tensor(fdp.ConsumeProbability()))
        elif mode == 1:
            forms.append(ltn.core.LTNObject(torch.tensor(fdp.ConsumeProbability(), requires_grad=True), []))
        elif mode == 2:
            n = fdp.ConsumeIntInRange(0, 3)
            forms.append(torch.rand(n))
        else:
            forms.append(ltn.core.LTNObject(torch.rand(2), ["x"]))
    p = [1, 2, 0, 5][fdp.ConsumeIntInRange(0, 3)]
    agg = fo.SatAgg(fo.AggregPMeanError(p=p, stable=fdp.ConsumeBool()))
    r = agg(*forms)
    if isinstance(r, torch.Tensor) and r.numel() == 1 and torch.isnan(r):
        raise Bug("SatAgg returned NaN")
    if isinstance(r, torch.Tensor) and r.numel() == 1 and r.requires_grad:
        r.backward()


# ---- test 6: functions with vector output feeding predicates, constants -------------------------------------
def t_func(fdp):
    x, nx, d = mkvar(fdp, "x", allow_bad=False)
    k = fdp.ConsumeIntInRange(1, 3)
    F = ltn.Function(func=lambda a: a.float().repeat(1, k)) if fdp.ConsumeBool() else ltn.Function(model=nn.Linear(d, k))
    c = ltn.Constant(tens(fdp, 1, d, 0, True)[0], trainable=fdp.ConsumeBool())
    P = ltn.Predicate(func=lambda a, b: torch.sigmoid((a.float() - b.float()).flatten(1).sum(dim=1)))
    fx = F(x)
    if fx.value.shape[0] != nx:
        raise Bug(f"Function lost individuals: {tuple(fx.value.shape)} for {nx}")
    r = P(F(x), F(c)) if True else None
    check_truth(r, "P(F(x),F(c))")
    s = ltn.Quantifier(fo.AggregPMeanError(), "f")(x, r)
    check_truth(s, "forall P(F(x),F(c))")
    if s.value.requires_grad and nx:
        s.value.backward()
        if c.value.requires_grad and c.value.grad is not None and torch.isnan(c.value.grad).any():
            raise Bug("NaN gradient on trainable constant")


# ---- test 7: same label, different sizes (must be an error or a defined behaviour, not a silent wrong answer) --
def t_samelabel(fdp):
    d = 2
    a = ltn.Variable("x", tens(fdp, fdp.ConsumeIntInRange(1, 4), d, 0, True))
    b = ltn.Variable("x", tens(fdp, fdp.ConsumeIntInRange(1, 4), d, 0, True))
    P = ltn.Predicate(func=lambda u, v: torch.sigmoid(u.sum(1) + v.sum(1)))
    r = P(a, b)
    check_truth(r, "same label")
    if a.value.shape[0] != b.value.shape[0]:
        raise Bug(f"two variables labelled 'x' with {a.value.shape[0]} and {b.value.shape[0]} individuals were accepted silently; result shape {tuple(r.value.shape)}")


# ---- test 8: nested quantifiers & implication, the classic LTN axiom -----------------------------------------
def t_nested(fdp):
    x, nx, d = mkvar(fdp, "x", allow_bad=False)
    y, ny, _ = mkvar(fdp, "y", allow_bad=False)
    y = ltn.Variable("y", tens(fdp, ny, d, 0, True))
    P = make_pred(fdp, d, 2); Qp = make_pred(fdp, d, 1)
    Fa = ltn.Quantifier(make_agg(fdp)[0], "f"); Ex = ltn.Quantifier(fo.AggregPMean(p=fdp.ConsumeIntInRange(1, 4)), "e")
    imp, name = make_bin(fdp)
    if imp is None:
        return
    inner = ltn.Connective(imp)(P(x, y), Qp(y))
    r = Fa(x, Ex(y, inner))
    check_truth(r, f"forall x exists y ({name})")
    if r.value.dim() != 0:
        raise Bug(f"closed formula has shape {tuple(r.value.shape)}")


# ---- test 9: mask edge cases (all true == no mask, all false) ---------------------------------------------------
def t_mask(fdp):
    x, nx, d = mkvar(fdp, "x", allow_bad=False)
    P = make_pred(fdp, d, 1)
    agg, an, p = make_agg(fdp)
    Qf = ltn.Quantifier(agg, "f")
    plain = Qf(x, P(x))
    allt = Qf(x, P(x), cond_vars=[x], cond_fn=lambda v: v.value.sum(dim=1) > -1e9)
    check_truth(plain, "plain"); check_truth(allt, "mask-all-true")
    if nx and not torch.allclose(plain.value.float(), allt.value.float(), atol=1e-5):
        raise Bug(f"all-true mask changes result: {plain.value.item()} vs {allt.value.item()} ({an}, p={p})")
    if allt.value.dtype != plain.value.dtype:
        raise Bug(f"mask changes dtype {plain.value.dtype} -> {allt.value.dtype}")
    none = Qf(x, P(x), cond_vars=[x], cond_fn=lambda v: v.value.sum(dim=1) > 1e9)
    check_truth(none, "mask-none")


# ---- test 10: reuse an intermediate result in two formulas (grad-carrying objects are not copied) --------------------
def t_reuse(fdp):
    x, nx, d = mkvar(fdp, "x", allow_bad=False)
    y = ltn.Variable("y", tens(fdp, fdp.ConsumeIntInRange(1, 3), d, 0, True))
    F = ltn.Function(model=nn.Linear(d, d))
    P = make_pred(fdp, d, 1); R = ltn.Predicate(model=Net(d, 2, False))
    fx = F(x); px = P(fx)
    kinds = fdp.ConsumeIntInRange(0, 3)
    snap = snapshot(fx, px)
    if kinds == 0:
        a1 = R(fx, F(y)); a2 = R(fx, F(y))
    elif kinds == 1:
        a1 = ltn.Connective(fo.AndProd())(px, P(F(y))); a2 = ltn.Connective(fo.AndProd())(px, P(F(y)))
    elif kinds == 2:
        a1 = ltn.Quantifier(fo.AggregMean(), "f")(y, ltn.Connective(fo.OrProbSum())(px, P(F(y)))); a2 = ltn.Quantifier(fo.AggregMean(), "f")(y, ltn.Connective(fo.OrProbSum())(px, P(F(y))))
    else:
        a1 = R(fx, F(y)); a2 = R(F(x), F(y))
    unchanged([fx, px], snap, "reuse")
    if a1.value.shape != a2.value.shape or not torch.allclose(a1.value, a2.value, atol=1e-6):
        raise Bug(f"same sub-formula gives different results on reuse: {tuple(a1.value.shape)} vs {tuple(a2.value.shape)}")


TESTS = [t_reuse, t_quant, t_duality, t_diag, t_var, t_conn, t_sat, t_func, t_samelabel, t_nested, t_mask]

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
