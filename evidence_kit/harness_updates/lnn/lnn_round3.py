"""Round 2/3 harness for LNN: metamorphic oracles and extended dimensions (binary predicates, nested quantifiers,
contradiction, learning ranges).  Same engine (libFuzzer via Atheris), coverage-only guidance, same record()/replay as lnn_fuzz."""
import sys, os, itertools
import lnn_fuzz as F
from lnn_fuzz import *   # Finding, record, build helpers, Propositions, Model, ... (lnn names come from lnn_fuzz)
import atheris

def pts(fdp):
    return F.mk_bounds(fdp)

def node_bounds(m, f): return F.bounds_of(F.stored(m, f))

def prop_model(formulas, leaves, data):
    m = Model(); m.add_knowledge(*formulas, *leaves); m.add_data({l: d[0] for l, d in zip(leaves, data)}); return m

def leaves_and_data(fdp, n):
    leaves = list(Propositions(*["R%d" % i for i in range(n)]))
    return leaves, [pts(fdp) for _ in leaves]

# ---------- metamorphic, propositional ----------
def t_meta_prop(fdp):
    k = fdp.ConsumeIntInRange(0, 6)
    n = 3
    leaves, data = leaves_and_data(fdp, n)
    A, B, C = leaves
    pairs = {
        0: lambda: (And(A, B, C), And(C, A, B)),                       # commutativity / associativity of the n-ary form
        1: lambda: (Or(A, B, C), Or(B, C, A)),
        2: lambda: (Not(And(A, B)), Or(Not(A), Not(B))),                # De Morgan
        3: lambda: (Not(Or(A, B)), And(Not(A), Not(B))),
        4: lambda: (Implies(A, B), Or(Not(A), B)),                      # material implication
        5: lambda: (Not(Not(A)), A),                                    # double negation (A is stored as a leaf)
        6: lambda: (Iff(A, B), Iff(B, A)),
    }
    f, g = pairs[k]()
    m = prop_model([f, g], leaves, data)
    m.infer(Direction.UPWARD, max_steps=F.MAXSTEPS)
    bf, bg = node_bounds(m, f), node_bounds(m, g)
    if not F.close(bf, bg):
        raise Finding("metamorphic_prop", "case%d %s vs %s data=%s: %s vs %s" % (k, f.name, g.name, [d[1] for d in data], bf, bg))

def t_meta_model(fdp):
    # order of add_knowledge, repeated infer, flush + re-add must not change the result (formulas with distinct names)
    leaves, data = leaves_and_data(fdp, 3)
    A, B, C = leaves
    fs = [And(A, B), Or(B, C), Implies(A, C), Not(B)]
    order = list(range(4)); 
    for i in range(3, 0, -1):
        j = fdp.ConsumeIntInRange(0, i); order[i], order[j] = order[j], order[i]
    def run(ordr, twice):
        L = list(Propositions("R0", "R1", "R2")); a, b, c = L
        F2 = [And(a, b), Or(b, c), Implies(a, c), Not(b)]
        m = Model(); m.add_knowledge(*[F2[i] for i in ordr], *L)
        m.add_data({l: d[0] for l, d in zip(L, data)})
        m.infer(max_steps=F.MAXSTEPS)
        if twice: m.infer(max_steps=F.MAXSTEPS)
        return [node_bounds(m, f) for f in F2], m.has_contradiction()
    r1, c1 = run(list(range(4)), False)
    r2, c2 = run(order, True)
    if c1 != c2 or not all(F.close(x, y) for x, y in zip(r1, r2)):
        raise Finding("order_dependence", "order=%s data=%s: %s vs %s" % (order, [d[1] for d in data], r1, r2))
    # flush and re-add data
    L = list(Propositions("R0", "R1", "R2")); a, b, c = L
    f = And(a, b); m = Model(); m.add_knowledge(f, *L); m.add_data({l: d[0] for l, d in zip(L, data)}); m.infer(max_steps=F.MAXSTEPS)
    first = node_bounds(m, f); m.flush(); m.add_data({l: d[0] for l, d in zip(L, data)}); m.infer(max_steps=F.MAXSTEPS)
    if not F.close(first, node_bounds(m, f)):
        raise Finding("flush_dependence", "data=%s first=%s after flush=%s" % ([d[1] for d in data], first, node_bounds(m, f)))

def t_contradiction(fdp):
    # a classically inconsistent knowledge base must be reported as a contradiction
    leaves, _ = leaves_and_data(fdp, 2)
    A, B = leaves
    k = fdp.ConsumeIntInRange(0, 3)
    va, vb = fdp.ConsumeBool(), fdp.ConsumeBool()
    f = [And(A, B), Or(A, B), Implies(A, B), Iff(A, B)][k]
    truth = [va and vb, va or vb, (not va) or vb, va == vb][k]
    m = Model(); m.add_knowledge(f, A, B)
    f.reset_world(World.CLOSED if truth else World.AXIOM)            # asserts the opposite of what the data give
    m.add_data({A: Fact.TRUE if va else Fact.FALSE, B: Fact.TRUE if vb else Fact.FALSE})
    m.infer(max_steps=F.MAXSTEPS)
    if not m.has_contradiction():
        raise Finding("missed_contradiction", "op=%d A=%s B=%s asserted=%s bounds=%s" % (k, va, vb, not truth, node_bounds(m, f)))

# ---------- first-order: binary predicates, nested quantifiers ----------
def t_fol2(fdp):
    x, y = Variable("x"), Variable("y")
    R, S = Predicate("R", arity=2), Predicate("S")
    dom = ["a", "b", "c"][:fdp.ConsumeIntInRange(2, 3)]
    Rv = {(i, j): fdp.ConsumeBool() for i in dom for j in dom}
    Sv = {i: fdp.ConsumeBool() for i in dom}
    shape = fdp.ConsumeIntInRange(0, 4)
    if shape == 0: f = Exists(y, And(R(x, y), S(y))); desc = "Exists y R(x,y)&S(y)"
    elif shape == 1: f = Forall(x, Exists(y, And(R(x, y), S(y)))); desc = "Forall x Exists y R&S"
    elif shape == 2: f = Forall(x, Forall(y, Implies(R(x, y), S(y)))); desc = "Forall x Forall y R->S"
    elif shape == 3: f = Forall(x, y, Implies(R(x, y), S(y))); desc = "Forall x,y R->S"
    else: f = Exists(x, Exists(y, And(R(x, y), S(y)))); desc = "Exists x Exists y R&S"
    m = Model(); m.add_knowledge(f)
    m.add_data({R: {k: (Fact.TRUE if v else Fact.FALSE) for k, v in Rv.items()}, S: {k: (Fact.TRUE if v else Fact.FALSE) for k, v in Sv.items()}})
    steps, _ = m.infer(max_steps=F.MAXSTEPS)
    if m.has_contradiction():
        raise Finding("fol_contradiction", "%s dom=%s R=%s S=%s" % (desc, dom, Rv, Sv))
    # classical value over the finite domain
    imp = lambda p, q: (not p) or q
    if shape == 0: pass
    cv = {1: all(any(Rv[i, j] and Sv[j] for j in dom) for i in dom),
          2: all(imp(Rv[i, j], Sv[j]) for i in dom for j in dom),
          3: all(imp(Rv[i, j], Sv[j]) for i in dom for j in dom),
          4: any(Rv[i, j] and Sv[j] for i in dom for j in dom)}.get(shape)
    if cv is not None:
        lo, hi = F.bounds_of(F.stored(m, f))
        if not (lo - F.EPS <= float(cv) <= hi + F.EPS):
            raise Finding("fol_unsound", "%s dom=%s R=%s S=%s classical=%s bounds=%s" % (desc, dom, Rv, Sv, cv, (lo, hi)))

def t_fol_shapes(fdp):
    # shapes that LNN's own documentation uses; any raw exception is a finding
    x, y = Variable("x"), Variable("y")
    R, S = Predicate("R", arity=2), Predicate("S")
    k = fdp.ConsumeIntInRange(0, 5)
    if k == 0: f = Exists(y, R(x, y))
    elif k == 1: f = Forall(x, Implies(S(x), Exists(y, R(x, y))))
    elif k == 2: f = Or(S(x), R(x, x))
    elif k == 3: f = And(S(x), R(x, x))
    elif k == 4: f = Forall(x, R(x, x))
    else: f = Implies(R(x, y), R(y, x))
    m = Model(); m.add_knowledge(f)
    dom = ["a", "b"]
    data = {R: {(i, j): pts(fdp)[0] for i in dom for j in dom}}
    if k in (1, 2, 3): data[S] = {i: pts(fdp)[0] for i in dom}
    m.add_data(data)
    m.infer(max_steps=F.MAXSTEPS)

# ---------- learning ranges ----------
def t_learn_ranges(fdp):
    A, B = Propositions("A", "B")
    ops = [And, Or, Implies]
    f = ops[fdp.ConsumeIntInRange(0, 2)](A, B, activation={"weights_learning": True, "bias_learning": True})
    m = Model(); m.add_knowledge(f); m.add_data({A: pts(fdp)[0], B: pts(fdp)[0]}); m.add_labels({f: pts(fdp)[0]})
    L = [Loss.SUPERVISED, Loss.CONTRADICTION, Loss.UNCERTAINTY][fdp.ConsumeIntInRange(0, 2)]
    lr = [0.0, 0.05, 1.0, 100.0][fdp.ConsumeIntInRange(0, 3)]
    m.train(losses=[L], epochs=fdp.ConsumeIntInRange(1, 5), pbar=False, learning_rate=lr, max_steps=F.MAXSTEPS)
    w = f.neuron.weights.data; b = f.neuron.bias.data
    if torch.isnan(w).any() or torch.isnan(b).any() or (w < -1e-6).any() or (b < -1e-6).any():
        raise Finding("param_range", "%s lr=%s weights=%s bias=%s" % (L.name, lr, w.tolist(), b.tolist()))

TESTS3 = [t for t in [t_meta_prop, t_meta_model, t_contradiction, t_fol2, t_fol_shapes, t_learn_ranges] if t.__name__ not in F.SKIP]

def TestOneInput(data):
    F.CTX["data"] = data
    fdp = atheris.FuzzedDataProvider(data)
    if len(data) < 4: return
    t = TESTS3[fdp.ConsumeIntInRange(0, len(TESTS3) - 1)]
    try:
        t(fdp)
    except Finding as e:
        F.record(e.kind + "/" + t.__name__, str(e))
    except (KeyError, TypeError, IndexError, ValueError, AssertionError) as e:
        fr = F.traceback.extract_tb(e.__traceback__)
        if fr and os.path.basename(fr[-1].filename) == "_exceptions.py": return
        F.record("exception/" + t.__name__, "%s: %s" % (type(e).__name__, str(e)[:300]), e)
    except RecursionError: pass
    except Exception as e:
        F.record("exception/" + t.__name__, "%s: %s" % (type(e).__name__, str(e)[:300]), e)

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]) and not sys.argv[1].startswith("-"):
        os.environ["LNN_REPLAY"] = "1"; TestOneInput(open(sys.argv[1], "rb").read()); print("replayed", sys.argv[1]); sys.exit(0)
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
