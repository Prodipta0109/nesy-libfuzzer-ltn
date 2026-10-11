"""Atheris/libFuzzer harness for IBM LNN 1.1.0 (coverage-guided, byte-driven)."""
import sys, os, json, hashlib, traceback, warnings, logging, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("LNN_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")))
import atheris, contextlib
# LNN_INSTR selects what libFuzzer measures (used by the evidence experiments): lnn (default) | none | lnn+torch
_INSTR = os.environ.get("LNN_INSTR", "lnn")
_ctx = (contextlib.nullcontext() if _INSTR == "none" else
        atheris.instrument_imports(include=["lnn", "torch"]) if _INSTR == "lnn+torch" else
        atheris.instrument_imports(include=["lnn"]))
with _ctx:
    import lnn
    from lnn import *
import torch
FREEZE = set(filter(None, os.environ.get("LNN_FREEZE", "").split(",")))   # dimension ablations: weights, points
SKIP = set(filter(None, os.environ.get("LNN_SKIP", "").split(",")))        # leave out whole test families
logging.disable(logging.CRITICAL)
EPS = 1e-4
OUT = os.environ.get("LNN_FINDINGS", "findings_lnn")
CTX = {"data": b""}
MAXSTEPS = None if os.environ.get("LNN_NOCAP") == "1" else 60     # A13; LNN_NOCAP=1 removes the step limit

class Finding(Exception):
    def __init__(self, kind, msg): super().__init__(msg); self.kind = kind

def record(kind, msg, exc=None):
    if os.environ.get("LNN_COUNT") == "1":          # evidence experiments: count every call, also repeated ones
        os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "_calls"), "a").write(".")
    tb = ""
    if exc is not None:
        fr = traceback.extract_tb(exc.__traceback__)
        tb = "%s:%s" % (os.path.basename(fr[-1].filename), fr[-1].name) if fr else ""
    sig = hashlib.sha1((kind + type(exc).__name__ + tb if exc else kind + msg.split()[0]).encode()).hexdigest()[:10]
    d = os.path.join(OUT, sig)
    if not os.path.exists(d) or os.environ.get("LNN_REPLAY"):
        print("[FINDING %s] %s: %s" % (sig, kind, msg[:160]), flush=True)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "input.bin"), "wb").write(CTX["data"])
        json.dump({"kind": kind, "msg": msg[:600], "where": tb,
                   "trace": traceback.format_exc()[-1500:] if exc else ""}, open(os.path.join(d, "meta.json"), "w"), indent=1)
    return sig

# ---------- formulas ----------
def mk_bounds(fdp):
    k = fdp.ConsumeIntInRange(0, 7)
    if k == 7 and "points" in FREEZE: k = 3
    if k == 7:
        v = fdp.ConsumeIntInRange(1, 9) / 10
        return (v, v), (v, v)
    if k == 0: return Fact.TRUE, (1.0, 1.0)
    if k == 1: return Fact.FALSE, (0.0, 0.0)
    if k == 2: return Fact.UNKNOWN, (0.0, 1.0)
    a = fdp.ConsumeIntInRange(0, 10) / 10; b = fdp.ConsumeIntInRange(0, 10) / 10
    lo, hi = min(a, b), max(a, b)
    return (lo, hi), (lo, hi)

def lk(op, ins):  # closed-form Lukasiewicz upward interval
    ls = [i[0] for i in ins]; us = [i[1] for i in ins]; n = len(ins)
    if op == "And": return (max(0, sum(ls) - (n - 1)), max(0, sum(us) - (n - 1)))
    if op == "Or": return (min(1, sum(ls)), min(1, sum(us)))
    if op == "Not": return (1 - us[0], 1 - ls[0])
    if op == "Implies":
        return (min(1, 1 - us[0] + ls[1]), min(1, 1 - ls[0] + us[1]))
    if op == "Iff":
        a = lk("Implies", ins); b = lk("Implies", ins[::-1])
        return lk("And", [a, b])
    raise ValueError(op)

def gd(op, ins):
    ls = [i[0] for i in ins]; us = [i[1] for i in ins]
    if op == "And": return (min(ls), min(us))
    if op == "Or": return (max(ls), max(us))
    if op == "Not": return (1 - us[0], 1 - ls[0])
    raise ValueError(op)

OPS = ["And", "Or", "Not", "Implies", "Iff"]
CLS = {"And": And, "Or": Or, "Not": Not, "Implies": Implies, "Iff": Iff}

def build(fdp, leaves, depth):
    """returns (formula, expected-evaluator over leaf bounds, op tree)"""
    if depth == 0 or fdp.ConsumeIntInRange(0, 3) == 0:
        i = fdp.ConsumeIntInRange(0, len(leaves) - 1)
        return leaves[i], ("leaf", i)
    op = OPS[fdp.ConsumeIntInRange(0, len(OPS) - 1)]
    ar = 1 if op == "Not" else 2 if op in ("Implies", "Iff") else fdp.ConsumeIntInRange(2, 4)
    kids = [build(fdp, leaves, depth - 1) for _ in range(ar)]
    return kids, (op, [k[1] for k in kids])

def realise(kids_tree, kw):
    # kids_tree: either a leaf formula or ([child pairs], (op,..))
    pass

SEEN = {}
def construct(fdp, leaves, depth, kw, nodes):
    if depth == 0 or fdp.ConsumeIntInRange(0, 3) == 0:
        i = fdp.ConsumeIntInRange(0, len(leaves) - 1)
        return leaves[i], ("leaf", i)
    op = OPS[fdp.ConsumeIntInRange(0, len(OPS) - 1)]
    ar = 1 if op == "Not" else 2 if op in ("Implies", "Iff") else fdp.ConsumeIntInRange(2, 4)
    kids = [construct(fdp, leaves, depth - 1, kw, nodes) for _ in range(ar)]
    f = CLS[op](*[k[0] for k in kids], **kw)
    tree = (op, [k[1] for k in kids])
    if f.name in SEEN: f = SEEN[f.name]
    else:
        SEEN[f.name] = f; nodes.append((f, tree))
    return f, tree

def evaluate(tree, leafb, sem):
    if tree[0] == "leaf": return leafb[tree[1]]
    ins = [evaluate(t, leafb, sem) for t in tree[1]]
    return sem(tree[0], ins)

def stored(m, f):
    """the node the model keeps for f: formulas are identified by their printed name, a second object with the same name is not stored"""
    for n in m.nodes.values():
        if n.name == f.name: return n
    return f

def bounds_of(f):
    d = f.get_data().detach().flatten().tolist()
    return d[0], d[1]

def close(a, b): return abs(a[0] - b[0]) <= EPS and abs(a[1] - b[1]) <= EPS

def run_infer(m, direction=None):
    return m.infer(direction, max_steps=MAXSTEPS) if direction else m.infer(max_steps=MAXSTEPS)

def classical_eval(tree, vals):
    if tree[0] == "leaf": return vals[tree[1]]
    c = [classical_eval(t, vals) for t in tree[1]]
    op = tree[0]
    if op == "And": return all(c)
    if op == "Or": return any(c)
    if op == "Not": return not c[0]
    if op == "Implies": return (not c[0]) or c[1]
    if op == "Iff": return c[0] == c[1]

def prop_setup(fdp, semantics):
    n = fdp.ConsumeIntInRange(2, 4)
    leaves = list(Propositions(*["P%d" % i for i in range(n)]))
    kw = {}
    if semantics == "godel": kw = {"activation": {"type": NeuralActivation.Godel}}
    nodes = []; SEEN.clear()
    root, tree = construct(fdp, leaves, fdp.ConsumeIntInRange(1, 3), kw, nodes)
    return leaves, nodes, tree, kw

# ---------- families ----------
def t_upward(fdp):
    sem = "godel" if fdp.ConsumeIntInRange(0, 7) == 0 else "luk"
    if sem == "godel":
        global OPS
    leaves, nodes, tree, kw = prop_setup(fdp, sem)
    if not nodes: return
    m = Model(); m.add_knowledge(*[n[0] for n in nodes], *leaves)
    lb = [mk_bounds(fdp) for _ in leaves]
    m.add_data({l: b[0] for l, b in zip(leaves, lb)})
    run_infer(m, Direction.UPWARD)
    f = lk if sem == "luk" else gd
    for node, t in nodes:
        if sem == "godel" and not set(_ops(t)) <= {"And", "Or", "Not"}: continue
        exp = evaluate(t, [b[1] for b in lb], f)
        got = bounds_of(stored(m, node))
        if not close(got, exp):
            raise Finding("wrong_bounds", "ops=%s %s %s expected %s got %s data=%s" % ("+".join(sorted(set(_ops(t)))), sem, t, exp, got, [b[1] for b in lb]))

def _ops(t):
    if t[0] == "leaf": return []
    r = [t[0]]
    for c in t[1]: r += _ops(c)
    return r

def t_classical(fdp):
    leaves, nodes, tree, kw = prop_setup(fdp, "luk")
    if not nodes: return
    vals = [fdp.ConsumeBool() for _ in leaves]
    m = Model(); m.add_knowledge(*[n[0] for n in nodes], *leaves)
    m.add_data({l: (Fact.TRUE if v else Fact.FALSE) for l, v in zip(leaves, vals)})
    steps, _ = run_infer(m)
    if m.has_contradiction():
        raise Finding("contradiction_on_consistent", "classical data %s tree %s" % (vals, tree))
    for node, t in nodes:
        want = classical_eval(t, vals)
        got = bounds_of(stored(m, node))
        if not close(got, (1.0, 1.0) if want else (0.0, 0.0)):
            raise Finding("classical_mismatch", "ops=%s tree %s vals %s want %s got %s" % ("+".join(sorted(set(_ops(t)))), t, vals, want, got))

def t_downward(fdp):
    # classical axiom root + known leaves except one: downward results must be sound
    leaves, nodes, tree, kw = prop_setup(fdp, "luk")
    if not nodes: return
    vals = [fdp.ConsumeBool() for _ in leaves]
    root, rt = nodes[-1]
    m = Model(); m.add_knowledge(root, *leaves); root.reset_world(World.AXIOM if classical_eval(rt, vals) else World.CLOSED)
    hide = fdp.ConsumeIntInRange(0, len(leaves) - 1)
    m.add_data({l: (Fact.TRUE if v else Fact.FALSE) for i, (l, v) in enumerate(zip(leaves, vals)) if i != hide})
    run_infer(m)
    if m.has_contradiction():
        # the hidden leaf could be either value; contradiction is legal only if no completion satisfies
        ok = any(classical_eval(rt, vals[:hide] + [h] + vals[hide + 1:]) == classical_eval(rt, vals) for h in (True, False))
        raise Finding("downward_contradiction", "ops=%s tree %s vals %s hidden %d" % ("+".join(sorted(set(_ops(rt)))), rt, vals, hide))
    lo, hi = bounds_of(leaves[hide])
    if not (lo - EPS <= float(vals[hide]) <= hi + EPS) and lo <= hi:
        pass  # leaf may be excluded legitimately by other completions; check completion soundness below
    comps = [h for h in (True, False) if classical_eval(rt, vals[:hide] + [h] + vals[hide + 1:]) == classical_eval(rt, vals)]
    if comps:
        if lo > hi + EPS: raise Finding("inverted", "leaf %.3f %.3f" % (lo, hi))
        for h in comps:
            if not (lo - EPS <= float(h) <= hi + EPS):
                raise Finding("downward_unsound", "ops=%s tree %s vals %s hidden %d leaf bounds %s exclude %s" % ("+".join(sorted(set(_ops(rt)))), rt, vals, hide, (lo, hi), h))

def t_invariants(fdp):
    leaves, nodes, tree, kw = prop_setup(fdp, "luk")
    if not nodes: return
    m = Model(); m.add_knowledge(*[n[0] for n in nodes], *leaves)
    m.add_data({l: mk_bounds(fdp)[0] for l in leaves})
    run_infer(m)
    contr = m.has_contradiction()
    for node, t in nodes:
        lo, hi = bounds_of(stored(m, node))
        if lo != lo or hi != hi: raise Finding("nan", str(t))
        if not contr and (lo < -EPS or hi > 1 + EPS or lo > hi + EPS):
            raise Finding("bad_interval", "%s -> %s" % (t, (lo, hi)))

def t_conv(fdp):
    # first-order convergence: does inference terminate within MAXSTEPS?
    x = Variable("x"); P = Predicate("S")
    q = Exists if fdp.ConsumeBool() else Forall
    grounds = ["a", "b", "c"][:fdp.ConsumeIntInRange(1, 3)]
    wrap = fdp.ConsumeIntInRange(0, 2)
    body = P(x) if wrap == 0 else Not(P(x)) if wrap == 1 else Implies(P(x), P(x))
    f = q(x, body)
    m = Model(); m.add_knowledge(f)
    data = {g: mk_bounds(fdp)[0] for g in grounds}
    m.add_data({P: data})
    steps, upd = m.infer(max_steps=MAXSTEPS)
    if steps >= MAXSTEPS:
        raise Finding("nonconvergence", "%s_wrap%d %s %s wrap=%d data=%s steps=%s upd=%s" % (q.__name__, wrap, q.__name__, grounds, wrap, data, steps, upd))

def t_fol(fdp):
    x = Variable("x"); P = Predicate("S")
    grounds = ["a", "b", "c"][:fdp.ConsumeIntInRange(1, 3)]
    vals = {g: mk_bounds(fdp) for g in grounds}
    ex = fdp.ConsumeBool()
    f = (Exists if ex else Forall)(x, P(x))
    m = Model(); m.add_knowledge(f)
    m.add_data({P: {g: v[0] for g, v in vals.items()}})
    m.infer(max_steps=MAXSTEPS)
    lo, hi = bounds_of(f)
    bs = [v[1] for v in vals.values()] + [(0.0, 1.0)]  # implicit open-world grounding
    exp = lk("Or" if ex else "And", bs)
    if not (lo >= exp[0] - 0.05 and hi <= exp[1] + 0.05):  # inference may not exceed what the data supports
        raise Finding("fol_unsound", "%s %s expected within %s got %s" % ("Exists" if ex else "Forall", vals, exp, (lo, hi)))

def t_learn(fdp):
    A, B = Propositions("A", "B")
    ops = [And, Or, Implies, Iff]
    f = ops[fdp.ConsumeIntInRange(0, 3)](A, B, activation={"bias_learning": fdp.ConsumeBool(), "weights_learning": fdp.ConsumeBool()})
    m = Model(); m.add_knowledge(f)
    m.add_data({A: mk_bounds(fdp)[0], B: mk_bounds(fdp)[0]})
    lab = mk_bounds(fdp)[0]
    m.add_labels({f: lab})
    losses = [Loss.SUPERVISED, Loss.CONTRADICTION, Loss.LOGICAL]
    L = losses[fdp.ConsumeIntInRange(0, 2)]
    epochs = fdp.ConsumeIntInRange(1, 4)
    m.train(losses=[L], epochs=epochs, pbar=False, max_steps=MAXSTEPS)
    for p in m.parameters() if callable(getattr(m, "parameters", None)) else []:
        v = p.detach()
        if torch.isnan(v).any(): raise Finding("nan_param", L.name)

def t_api(fdp):
    A, B = Propositions("A", "B")
    f = And(A, B)
    m = Model(); m.add_knowledge(f)
    k = fdp.ConsumeIntInRange(0, 7)
    if k == 0: m.add_data({A: (fdp.ConsumeFloat(), fdp.ConsumeFloat())})
    elif k == 1: m.add_data({A: fdp.ConsumeFloat()})
    elif k == 2:
        m.add_data({A: Fact.TRUE, B: Fact.TRUE}); m.infer(max_steps=5); m.reset_bounds()
        if bounds_of(f) != (0.0, 1.0) or bounds_of(A) != (1.0, 1.0): raise Finding("reset_bounds", "after reset %s %s" % (bounds_of(f), bounds_of(A)))
    elif k == 3:
        m.add_data({A: Fact.TRUE, B: Fact.TRUE}); m.infer(max_steps=5); m.flush()
        lo, hi = bounds_of(f)
        if (lo, hi) != (0.0, 1.0): raise Finding("flush", "after flush %s" % ((lo, hi),))
    elif k == 4:
        t = m.print(); 
    elif k == 5:
        for fa in (True, False, Fact.TRUE, Fact.FALSE, Fact.UNKNOWN, Fact.CONTRADICTION):
            pass
        for fk in (Fact.TRUE, Fact.FALSE, Fact.UNKNOWN, Fact.CONTRADICTION):
            b = utils_roundtrip(fk)
    elif k == 6:
        n = fdp.ConsumeIntInRange(0, 3)
        names = [fdp.ConsumeUnicodeNoSurrogates(3) for _ in range(n)]
        Propositions(*names)
    else:
        m.add_data({A: (0.7, 0.2)})  # inverted bounds

def utils_roundtrip(fk):
    from lnn.utils import bool_to_fact, fact_to_bool
    return fact_to_bool(fk)


def t_weights(fdp):
    """weighted Lukasiewicz neurons: exact upward bounds and convergence of inference"""
    op = ["And", "Or", "Implies"][fdp.ConsumeIntInRange(0, 2)]
    ar = 2 if op == "Implies" else fdp.ConsumeIntInRange(2, 3)
    W = [0.5, 0.9, 1.0, 1.5, 2.0]
    w = tuple(W[fdp.ConsumeIntInRange(0, 4)] for _ in range(ar))
    beta = [1.0, 0.9, 0.8][fdp.ConsumeIntInRange(0, 2)]
    if "weights" in FREEZE: w, beta = (1.0,) * ar, 1.0
    leaves = list(Propositions(*["Q%d" % i for i in range(ar)]))
    f = CLS[op](*leaves, activation={"weights": w, "bias": beta, "weights_learning": False, "bias_learning": False})
    m = Model(); m.add_knowledge(f)
    lb = [mk_bounds(fdp) for _ in leaves]
    m.add_data({l: b[0] for l, b in zip(leaves, lb)})
    cl = lambda v: min(1.0, max(0.0, v))
    ls = [b[1][0] for b in lb]; us = [b[1][1] for b in lb]
    if op == "And": exp = (cl(beta - sum(wi * (1 - l) for wi, l in zip(w, ls))), cl(beta - sum(wi * (1 - u) for wi, u in zip(w, us))))
    elif op == "Or": exp = (cl(1 - beta + sum(wi * l for wi, l in zip(w, ls))), cl(1 - beta + sum(wi * u for wi, u in zip(w, us))))
    else: exp = (cl(1 - beta + w[0] * (1 - us[0]) + w[1] * ls[1]), cl(1 - beta + w[0] * (1 - ls[0]) + w[1] * us[1]))
    m.infer(Direction.UPWARD, max_steps=MAXSTEPS)
    got = bounds_of(f)
    if not close(got, exp):
        raise Finding("weighted_upward", "weighted_%s w=%s beta=%s data=%s expected %s got %s" % (op, w, beta, [b[1] for b in lb], exp, got))
    m.reset_bounds(); m.add_data({l: b[0] for l, b in zip(leaves, lb)})
    steps, upd = m.infer(max_steps=MAXSTEPS)
    if steps >= MAXSTEPS:
        raise Finding("nonconvergence", "weights_%s w=%s beta=%s data=%s steps=%s" % (op, w, beta, [b[1] for b in lb], steps))

TESTS = [t for t in [t_upward, t_classical, t_downward, t_invariants, t_conv, t_fol, t_weights, t_learn, t_api] if t.__name__ not in SKIP]

def TestOneInput(data):
    CTX["data"] = data
    fdp = atheris.FuzzedDataProvider(data)
    if len(data) < 4: return
    t = TESTS[fdp.ConsumeIntInRange(0, len(TESTS) - 1)]
    try:
        t(fdp)
    except Finding as e:
        record(e.kind + "/" + t.__name__, str(e))
    except (KeyError, TypeError, IndexError, ValueError, AssertionError) as e:
        # LNN's own validation uses these types; only count crashes raised outside _exceptions.py
        fr = traceback.extract_tb(e.__traceback__)
        if fr and os.path.basename(fr[-1].filename) == "_exceptions.py": return
        record("exception/" + t.__name__, "%s: %s" % (type(e).__name__, str(e)[:300]), e)
    except RecursionError: pass
    except Exception as e:
        record("exception/" + t.__name__, "%s: %s" % (type(e).__name__, str(e)[:300]), e)

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]) and not sys.argv[1].startswith("-"):
        os.environ["LNN_REPLAY"] = "1"; TestOneInput(open(sys.argv[1], "rb").read()); print("replayed", sys.argv[1]); sys.exit(0)
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
