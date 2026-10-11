"""libFuzzer (Atheris) harness for DeepProbLog 2.1.0, ExactEngine + GraphSemiring.

Guidance: code coverage only (libFuzzer). Dimensions: the bytes libFuzzer mutates are decoded into the
choices of dimensions P1-P7 (see deepproblog_dimensions.csv, derived by classify_dims_dpl.py).

  P1 logic program      random stratified programs (facts, annotated disjunctions, rules, negation), graphs with cycles
  P2 probability values 0, 1, 1e-9, 1-1e-9, random; saturated softmax / sigmoid outputs
  P3 neural predicates  nn/4 (annotated disjunction), nn/2 (neural fact), batching, k, tensor sources
  P4 query              Query objects, substitution, output_ind, variable_output / replace_output
  P5 solver             cache on/off, sdd_auto_gc, repeated solve, batches of queries
  P6 training           cross_entropy / mse, SGD on probabilistic parameters, AD groups, learning rate
  P7 API boundary       Dataset / Subset / Fold / DataLoader, save_state / load_state, foreign predicates

Oracles (independent of DeepProbLog and of ProbLog):
  O1 exact probability by brute-force enumeration of all possible worlds
  O2 gradient of that probability via torch autograd on a hand-written formula
  O3 invariants: results in [0,1], no NaN, caching does not change results, round trips, partitions
  O4 any unexpected exception on a valid input
"""
import os, sys, re, io, json, math, random, types, hashlib, itertools, tempfile, traceback, warnings, contextlib

warnings.filterwarnings("ignore")
import atheris

# DPL_INSTR selects what libFuzzer measures (used by the evidence experiments):
#   both (default: deepproblog + problog) | deepproblog | none
_INSTR = os.environ.get("DPL_INSTR", "both")
_ctx = (contextlib.nullcontext() if _INSTR == "none" else
        atheris.instrument_imports(include=["deepproblog"]) if _INSTR == "deepproblog" else
        atheris.instrument_imports(include=["deepproblog", "problog"]))
FREEZE = set(filter(None, os.environ.get("DPL_FREEZE", "").split(",")))   # dimension ablation: form (file-name form)
GUARDS = os.environ.get("DPL_NOGUARD") != "1"                                # A11: loop counters in the harness
with _ctx:
    import torch
    import torch.nn as nn
    from problog.logic import Term, Constant, Var
    from deepproblog.model import Model
    from deepproblog.network import Network
    from deepproblog.engines import ExactEngine
    from deepproblog.query import Query
    from deepproblog.dataset import (Dataset, DataLoader, QueryDataset, NumericIncorrectOutputMutator)
    from deepproblog.train import train_model
    from deepproblog.evaluate import get_confusion_matrix
    from deepproblog.utils.logger import Logger
    from deepproblog.utils.confusion_matrix import ConfusionMatrix
    from deepproblog.utils.stop_condition import EpochStop, Threshold, StopOnPlateau, StopOnNoChange
    from deepproblog.optimizer import SGD
    from deepproblog.semiring.graph_semiring import GraphSemiring

OUT = os.environ.get("DPL_OUT", "findings_dpl")
os.makedirs(OUT, exist_ok=True)
SEEN = set()
CTX = {"txt": ""}
TOL = 1e-6


class Bug(Exception):
    pass


def record(kind, detail):
    tb = traceback.extract_tb(sys.exc_info()[2])
    loc = ""
    for fr in reversed(tb):
        if "/deepproblog/" in fr.filename or "/problog/" in fr.filename:
            loc = f"{os.path.basename(fr.filename)}:{fr.lineno}"
            break
    if os.environ.get("DPL_COUNT") == "1":        # evidence experiments: count every call, also repeated ones
        os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "_calls"), "a").write(".")
    norm = re.sub(r"[0-9.eE+-]+|\{.*?\}", "#", detail)[:60]
    sig = hashlib.sha1(f"{kind}|{norm}|{loc}".encode()).hexdigest()[:10]
    if sig in SEEN:
        return
    SEEN.add(sig)
    d = os.path.join(OUT, sig)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "input.bin"), "wb") as f:       # replay: python3 dpl_fuzz.py <dir>/input.bin
        f.write(CTX.get("data", b""))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"kind": kind, "detail": detail, "loc": loc, "context": CTX["txt"],
                   "trace": traceback.format_exc()[-1800:]}, f, indent=1)
    print(f"\n[FINDING {sig}] {kind} @ {loc}: {detail[:160]}", flush=True)


# ------------------------------------------------------------------ helpers
PROBS = [0.0, 1.0, 0.5, 0.25, 0.75, 0.1, 0.9, 1e-9, 1 - 1e-9]


def prob(fdp):
    return PROBS[fdp.ConsumeIntInRange(0, len(PROBS) - 1)] if fdp.ConsumeBool() else fdp.ConsumeProbability()


def fmt(p):
    return repr(float(p))


def as_float(x):
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def make_model(prog, nets=(), cache=False, gc=False, k_eval=False):
    m = Model(prog, list(nets), load=False)
    m.set_engine(ExactEngine(m), cache=cache, sdd_auto_gc=gc)
    if k_eval:
        m.eval()
    return m


def query_value(m, q):
    """Probability DeepProbLog returns for ground query term q (0.0 when it reports no answer)."""
    res = m.solve([Query(q)])[0].result
    if q in res:
        return res[q], True
    if len(res) == 0:
        return 0.0, False
    raise Bug(f"result keys {list(res)} do not contain the query {q}")


def check_close(name, got, want):
    g = as_float(got)
    if math.isnan(g):
        raise Bug(f"{name}: NaN probability (oracle {want})")
    if g < -TOL or g > 1 + TOL:
        raise Bug(f"{name}: probability {g} outside [0,1] (oracle {want})")
    if abs(g - want) > TOL:
        raise Bug(f"{name}: DeepProbLog {g} but possible-worlds oracle {want}")


@contextlib.contextmanager
def in_tmp():
    if os.environ.get("DPL_NOTMP") == "1":      # ablation of A12: write into the current directory
        yield os.getcwd(); return
    old = os.getcwd()
    td = tempfile.TemporaryDirectory()
    os.chdir(td.name)
    try:
        yield td.name
    finally:
        os.chdir(old)
        td.cleanup()


FORMS = ["{}", "./{}", "sub/{}"]          # bare file name, ./name, nested directory
if "form" in FREEZE: FORMS = ["sub/{}"] * 3   # ablation: never use a bare file name

# ------------------------------------------------------------------ T1: random propositional programs
def gen_prop(fdp):
    n = fdp.ConsumeIntInRange(1, 6)
    lines, facts, ads, rules = [], [], [], {}
    i = 0
    while i < n:
        kind = fdp.ConsumeIntInRange(0, 2)
        if kind == 1 and i == 0:
            kind = 0
        if kind == 2 and i + 1 >= n:
            kind = 0
        if kind == 0:
            p = prob(fdp)
            facts.append((i, p))
            lines.append(f"{fmt(p)}::a{i}.")
            i += 1
        elif kind == 2:
            members, left = [], 1.0
            for j in range(min(fdp.ConsumeIntInRange(2, 3), n - i)):
                pj = min(left * prob(fdp), left)
                pj = max(pj, 0.0)
                members.append((i + j, pj))
                left = max(left - pj, 0.0)
            while sum(p for _, p in members) > 1.0:
                members[-1] = (members[-1][0], math.nextafter(members[-1][1], 0.0))
            ads.append(members)
            lines.append("; ".join(f"{fmt(p)}::a{a}" for a, p in members) + ".")
            i += len(members)
        else:
            clauses = []
            for _ in range(fdp.ConsumeIntInRange(1, 2)):
                lits = []
                for _ in range(fdp.ConsumeIntInRange(1, 3)):
                    lits.append((fdp.ConsumeIntInRange(0, i - 1), fdp.ConsumeBool()))
                rp = prob(fdp) if fdp.ConsumeIntInRange(0, 3) == 0 else None
                clauses.append((lits, rp))
            rules[i] = clauses
            for lits, rp in clauses:
                body = ", ".join((("\\+ " if neg else "") + f"a{j}") for j, neg in lits)
                head = f"{fmt(rp)}::a{i}" if rp is not None else f"a{i}"
                lines.append(f"{head} :- {body}.")
            i += 1
    q = fdp.ConsumeIntInRange(0, n - 1)
    lines.append(f"nq :- \\+ a{q}.")
    return "\n".join(lines), n, facts, ads, rules, q


def prop_oracle(n, facts, ads, rules, q):
    choices = [[(("a", a), p), (None, 1 - p)] for a, p in facts]
    for g in ads:
        choices.append([(("a", a), p) for a, p in g] + [(None, max(1 - sum(p for _, p in g), 0.0))])
    for h, clauses in rules.items():
        for ci, (lits, rp) in enumerate(clauses):
            if rp is not None:
                choices.append([(("c", h, ci), rp), (None, 1 - rp)])
    total = 0.0
    for combo in itertools.product(*choices):
        w, tok = 1.0, set()
        for t_, p in combo:
            w *= p
            if t_ is not None:
                tok.add(t_)
        if w == 0.0:
            continue
        true = {t_[1] for t_ in tok if t_[0] == "a"}
        for h in range(n):
            if h in rules:
                for ci, (lits, rp) in enumerate(rules[h]):
                    if all((j in true) != neg for j, neg in lits) and (rp is None or ("c", h, ci) in tok):
                        true.add(h)
        if q in true:
            total += w
    return total


def t_prop(fdp):
    prog, n, facts, ads, rules, q = gen_prop(fdp)
    cache, gc, use_root = fdp.ConsumeBool(), fdp.ConsumeBool(), fdp.ConsumeBool()
    CTX["txt"] = f"T1 cache={cache} gc={gc} cache_root={use_root} query=a{q}\n{prog}"
    want = prop_oracle(n, facts, ads, rules, q)
    with tempfile.TemporaryDirectory() as td:
        m = Model(prog, [], load=False)
        m.set_engine(ExactEngine(m), cache=cache, sdd_auto_gc=gc, cache_root=(os.path.join(td, "c") if use_root else None))
        got, present = query_value(m, Term(f"a{q}"))
        if not present and want > TOL:
            raise Bug(f"a{q}: DeepProbLog returned no answer but oracle probability is {want}")
        check_close(f"a{q}", got, want)
        both = m.solve([Query(Term(f"a{q}")), Query(Term("nq")), Query(Term(f"a{q}"))])
        vals = [list(r.result.values()) for r in both]
        if len(vals[0]) and len(vals[2]) and abs(as_float(vals[0][0]) - as_float(vals[2][0])) > 1e-12:
            raise Bug("same query twice in one batch gave different answers")
        gotn = vals[1][0] if vals[1] else 0.0
        check_close("nq", gotn, 1 - want)


# ------------------------------------------------------------------ T2: probabilistic graph with cycles
def t_graph(fdp):
    nn_ = fdp.ConsumeIntInRange(2, 4)
    edges = [(fdp.ConsumeIntInRange(0, nn_ - 1), fdp.ConsumeIntInRange(0, nn_ - 1), prob(fdp))
             for _ in range(fdp.ConsumeIntInRange(1, 6))]
    s, t = fdp.ConsumeIntInRange(0, nn_ - 1), fdp.ConsumeIntInRange(0, nn_ - 1)
    prog = "\n".join(f"{fmt(p)}::edge(n{u},n{v})." for u, v, p in edges)
    prog += "\npath(X,Y) :- edge(X,Y).\npath(X,Y) :- edge(X,Z), path(Z,Y).\n"
    cache = fdp.ConsumeBool()
    CTX["txt"] = f"T2 cache={cache} query=path(n{s},n{t})\n{prog}"
    total = 0.0
    for mask in range(1 << len(edges)):
        w, adj = 1.0, {}
        for k, (u, v, p) in enumerate(edges):
            if mask >> k & 1:
                w *= p
                adj.setdefault(u, set()).add(v)
            else:
                w *= 1 - p
        if w == 0.0:
            continue
        seen, todo = set(), [s]
        while todo:
            x = todo.pop()
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        if t in seen:
            total += w
    m = make_model(prog, cache=cache)
    q = Term("path", Term(f"n{s}"), Term(f"n{t}"))
    got, present = query_value(m, q)
    if not present and total > TOL:
        raise Bug(f"path(n{s},n{t}): no answer but oracle probability {total}")
    check_close(f"path(n{s},n{t})", got, total)


# ------------------------------------------------------------------ T3: neural predicates
class DNet(nn.Module):
    def __init__(self, d_in, d_out, w, b, scale, sigmoid=False):
        super().__init__()
        self.l = nn.Linear(d_in, d_out)
        with torch.no_grad():
            self.l.weight.copy_(torch.tensor(w, dtype=torch.float32).reshape(d_out, d_in) * scale)
            self.l.bias.copy_(torch.tensor(b, dtype=torch.float32) * scale)
        self.sigmoid = sigmoid

    def forward(self, x):
        o = self.l(x)
        return torch.sigmoid(o) if self.sigmoid else torch.softmax(o, -1)


def flt(fdp, lo=-3.0, hi=3.0):
    return fdp.ConsumeFloatInRange(lo, hi)


def t_neural(fdp):
    D = fdp.ConsumeIntInRange(2, 4)
    tmpl = fdp.ConsumeIntInRange(0, 6)
    batching, cache = fdp.ConsumeBool(), fdp.ConsumeBool()
    scale = [1.0, 10.0, 100.0][fdp.ConsumeIntInRange(0, 2)]
    w = [flt(fdp) for _ in range(D * 2)]
    b = [flt(fdp) for _ in range(D)]
    xs = [torch.tensor([flt(fdp, -1, 1), flt(fdp, -1, 1)]) for _ in range(2)]
    z = fdp.ConsumeIntInRange(-1, 2 * D)
    fact_net = tmpl >= 5
    kval = fdp.ConsumeIntInRange(1, D) if (fdp.ConsumeBool() and not fact_net) else None
    eval_mode = fdp.ConsumeBool()
    ids = [0, 1] if fdp.ConsumeBool() else [0, 0]
    net_mod = DNet(2, 1 if fact_net else D, w[: (2 if fact_net else 2 * D)], b[: (1 if fact_net else D)], scale, sigmoid=fact_net)
    net = Network(net_mod, "net", batching=batching, k=kval)
    head = "nn(net,[X]) :: f(X)." if fact_net else "nn(net,[X],Y,[" + ",".join(map(str, range(D))) + "]) :: d(X,Y)."
    rules = {
        0: "s(X,Y,Z) :- d(X,A), d(Y,B), Z is A+B.",
        1: "s(X,Y) :- d(X,A), d(Y,A).",
        2: "s(X,Y) :- d(X,A), d(Y,B), A > B.",
        3: "nz(X) :- d(X,A), A > 0.\nzr(X) :- \\+ nz(X).",
        4: "s(X,Z) :- d(X,A), d(X,B), Z is A+B.",
        5: "s(X,Y) :- f(X), \\+ f(Y).",
        6: "s(X,Y) :- f(X), f(Y).",
    }[tmpl]
    prog = head + "\n" + rules
    CTX["txt"] = (f"T3 tmpl={tmpl} D={D} batching={batching} cache={cache} k={kval} eval_mode={eval_mode} scale={scale} z={z} ids={ids}\n"
                  f"w={w[:2*D]} b={b[:D]} x={[x.tolist() for x in xs]}\n{prog}")
    m = make_model(prog, [net], cache=cache)
    if eval_mode:
        m.eval()
    m.add_tensor_source("src", {(Constant(0),): xs[0], (Constant(1),): xs[1]})
    T = lambda i: Term("tensor", Term("src", Constant(i)))
    # oracle (differentiable, written independently with torch). Two uses of the SAME input are the SAME
    # ground atom, so they are perfectly correlated: the joint over (A, B) is diagonal in that case.
    o = [net_mod(xs[i]) for i in ids]
    if kval is not None:                      # top-k: only the k most likely values keep their probability
        om = []
        for oi in o:
            mask = torch.zeros(D)
            mask[torch.topk(oi.detach(), kval, 0)[1]] = 1.0
            om.append(oi * mask)
        o = om
    same = ids[0] == ids[1]
    if fact_net:
        pa, pb = o[0][0], o[1][0]
        if same:
            want = {5: torch.zeros(()), 6: pa}[tmpl]
        else:
            want = {5: pa * (1 - pb), 6: pa * pb}[tmpl]
        q = Term("s", T(ids[0]), T(ids[1]))
    else:
        J = torch.diag(o[0]) if same else torch.outer(o[0], o[1])
        pairs = [(a, bb) for a in range(D) for bb in range(D)]
        if tmpl == 0:
            want = sum((J[a, bb] for a, bb in pairs if a + bb == z), torch.zeros(()))
            q = Term("s", T(ids[0]), T(ids[1]), Constant(z))
        elif tmpl == 1:
            want = sum((J[a, a] for a in range(D)), torch.zeros(()))
            q = Term("s", T(ids[0]), T(ids[1]))
        elif tmpl == 2:
            want = sum((J[a, bb] for a, bb in pairs if a > bb), torch.zeros(()))
            q = Term("s", T(ids[0]), T(ids[1]))
        elif tmpl == 3:
            want = 1 - sum((o[0][a] for a in range(1, D)), torch.zeros(()))
            q = Term("zr", T(ids[0]))
        else:
            want = o[0][z // 2] if (z >= 0 and z % 2 == 0 and z // 2 < D) else torch.zeros(())
            q = Term("s", T(ids[0]), Constant(z))
    want = want if isinstance(want, torch.Tensor) else torch.tensor(float(want))
    for p_ in net_mod.parameters():
        p_.grad = None
    got, present = query_value(m, q)
    if kval is not None and not eval_mode:
        # training mode samples the k values: no exact oracle, but the answer must be a valid probability
        g = as_float(got)
        if math.isnan(g) or g < -TOL or g > 1 + TOL:
            raise Bug(f"top-k (train mode) probability {g}")
        return
    wv = float(want.detach())
    if not present and wv > TOL:
        raise Bug(f"neural query returned no answer; oracle {wv}")
    check_close("neural query", got, wv)
    if isinstance(got, torch.Tensor) and got.requires_grad and want.requires_grad:
        g_oracle = torch.autograd.grad(want, list(net_mod.parameters()), allow_unused=True, retain_graph=True)
        got.backward()
        for p_, go in zip(net_mod.parameters(), g_oracle):
            gg = p_.grad if p_.grad is not None else torch.zeros_like(p_)
            go = go if go is not None else torch.zeros_like(p_)
            if torch.isnan(gg).any():
                raise Bug("gradient contains NaN (probability valid)")
            if (gg - go).abs().max() > 1e-4 * (1 + go.abs().max()):
                raise Bug(f"gradient wrong: DeepProbLog {gg.flatten().tolist()} vs autograd oracle {go.flatten().tolist()}")


# ------------------------------------------------------------------ T4: learnable probabilistic parameters
def t_learn(fdp):
    pf = prob(fdp)
    pa = prob(fdp)
    pb = (1 - pa) * prob(fdp)
    while pa + pb > 1.0:
        pb = math.nextafter(pb, 0.0)
    lr = [0.0, 0.1, 1.0, 10.0, 1e6, -1.0][fdp.ConsumeIntInRange(0, 5)]
    target = prob(fdp)
    loss_name = ["cross_entropy", "mse"][fdp.ConsumeIntInRange(0, 1)]
    mode = fdp.ConsumeIntInRange(0, 3)       # 0 plain fact g1, 1 AD all learnable, 2 AD first learnable, 3 AD second learnable
    steps = fdp.ConsumeIntInRange(1, 3)
    if mode == 0:
        ad = "t(0.5)::g1.\n"
    elif mode == 1:
        ad = f"t({fmt(pa)})::g1; t({fmt(pb)})::g2.\n"
    elif mode == 2:
        ad = f"t({fmt(pa)})::g1; {fmt(pb)}::g2.\n"
    else:
        ad = f"{fmt(pa)}::g1; t({fmt(pb)})::g2.\n"
    prog = f"t({fmt(pf)})::f.\n{ad}q :- f, g1.\n"
    CTX["txt"] = f"T4 lr={lr} target={target} loss={loss_name} steps={steps} mode={mode}\n{prog}"
    m = make_model(prog)
    m.optimizer = SGD(m, lr)
    loss_fn = getattr(GraphSemiring, loss_name)
    q = Term("q")
    for _ in range(steps):
        m.optimizer.zero_grad()
        r = m.solve([Query(q)])[0]
        px = r.result.get(q)
        if px is None:
            return
        loss = loss_fn(r, target, 1.0, q=q)
        if math.isnan(loss):
            raise Bug(f"{loss_name} loss is NaN for p={as_float(px)} target={target}")
        if isinstance(px, torch.Tensor) and loss_name == "mse":
            # d/dp (p-target)^2 = 2(p-target); p = f * g1, so d/df = 2(p-t)*g1 and d/dg1 = 2(p-t)*f
            P = {k_: float(v) for k_, v in enumerate(m.parameters)}
            gi = {0: None}
            fv = P[0]
            g1v = P[1] if mode in (0, 1, 2) else pa
            pv = as_float(px)
            want0 = 2 * (pv - target) * g1v
            have0 = float(m.optimizer._params_grad.get(0, 0.0))
            if abs(want0 - have0) > 1e-4 * (1 + abs(want0)):
                raise Bug(f"gradient of mse wrt f: DeepProbLog {have0} vs analytic {want0}")
        m.optimizer.step()
        for v in m.parameters:
            if math.isnan(v) or v < 0 or v > 1:
                raise Bug(f"parameter left [0,1] or NaN after step: {m.parameters}")
        if mode in (2, 3):
            fixed, learned = (pb, m.parameters[1]) if mode == 2 else (pa, m.parameters[1])
            if fixed + learned > 1 + 1e-9:
                raise Bug(f"annotated disjunction mass {fixed + learned} > 1 after step (fixed {fixed}, learned {learned})")
        for grp in m.parameter_groups:
            s_ = sum(m.parameters[i] for i in grp)
            if abs(s_ - 1) > 1e-6:
                raise Bug(f"AD group no longer sums to 1 after step: {s_}")


# ------------------------------------------------------------------ T5: API boundary
class ListDS(Dataset):
    def __init__(self, n):
        self.qs = [Query(Term("q", Constant(i)), p=1.0) for i in range(n)]

    def __len__(self):
        return len(self.qs)

    def to_query(self, i):
        return self.qs[i]


def t_api(fdp):
    sub = fdp.ConsumeIntInRange(0, 5)
    if sub == 0:     # folds partition the dataset
        n, nf = fdp.ConsumeIntInRange(0, 12), fdp.ConsumeIntInRange(1, 6)
        i = fdp.ConsumeIntInRange(0, nf - 1)
        CTX["txt"] = f"T5 fold n={n} folds={nf} i={i}"
        ds = ListDS(n)
        tr, te = ds.fold(nf, i)
        a = [str(tr.to_query(k).query) for k in range(len(tr))]
        b = [str(te.to_query(k).query) for k in range(len(te))]
        allq = sorted(a + b)
        if allq != sorted(str(q.query) for q in ds.to_queries()):
            raise Bug(f"fold({nf},{i}) of {n} items: train {a} + test {b} is not a partition")
    elif sub == 1:   # subset / extension lengths and content
        n = fdp.ConsumeIntInRange(0, 8)
        i, j = fdp.ConsumeIntInRange(0, n), fdp.ConsumeIntInRange(0, 10)
        i, j = min(i, j), max(i, j)
        CTX["txt"] = f"T5 subset n={n} i={i} j={j}"
        ds = ListDS(n)
        s = ds.subset(i, j)
        want = [str(q.query) for q in ds.to_queries()[i:j]]
        got = [str(s.to_query(k).query) for k in range(len(s))]
        if got != want:
            raise Bug(f"subset({i},{j}) of {n} items gave {got}, expected {want}")
        e = ds + s
        if len(e) != n + len(s) or len(e.to_queries()) != len(e):
            raise Bug("Extension length mismatch")
    elif sub == 2:   # data loader covers every item exactly once
        n, bs = fdp.ConsumeIntInRange(0, 9), fdp.ConsumeIntInRange(0, 5)
        shuffle = fdp.ConsumeBool()
        CTX["txt"] = f"T5 loader n={n} batch_size={bs} shuffle={shuffle}"
        loader = DataLoader(ListDS(n), bs, shuffle=shuffle, seed=1)
        seen, count = [], 0
        for batch in loader:
            count += 1
            if GUARDS and count > n + 5:
                raise Bug(f"DataLoader(batch_size={bs}) over {n} items never stops iterating (yields {count} batches, last={batch})")
            seen += [str(q.query) for q in batch]
        if len(loader) != sum(1 for _ in DataLoader(ListDS(n), bs, shuffle=False)):
            raise Bug(f"len(loader)={len(loader)} differs from the number of batches produced")
        if sorted(seen) != sorted(str(q.query) for q in ListDS(n).to_queries()):
            raise Bug(f"loader yielded {seen}")
    elif sub == 3:   # Query API round trips
        na = fdp.ConsumeIntInRange(1, 3)
        args = [Constant(fdp.ConsumeIntInRange(0, 5)) for _ in range(na)]
        oi = tuple(sorted({fdp.ConsumeIntInRange(-na, na - 1) for _ in range(fdp.ConsumeIntInRange(1, na))}))
        p = prob(fdp)
        subst = {Var("X"): Constant(1)} if fdp.ConsumeBool() else None
        CTX["txt"] = f"T5 query args={args} output_ind={oi} p={p} substitution={subst}"
        q = Query(Term("f", *args), subst, p, oi)
        v = q.variable_output()
        back = v.replace_output(q.output_values())
        if back.query != q.query:
            raise Bug(f"variable_output/replace_output round trip: {q.query} -> {back.query}")
        hash(q)                    # queries must be usable in sets / dict keys
        if q != Query(Term("f", *args), subst, p, oi):
            raise Bug("equal queries compare unequal")
        q.to_text_query()
    elif sub == 4:   # save_state / load_state round trip
        D = 2
        w = [flt(fdp) for _ in range(4)]
        b = [flt(fdp) for _ in range(2)]
        complete = fdp.ConsumeBool()
        pf = prob(fdp)
        prog = f"t({fmt(pf)})::f.\nnn(net,[X],Y,[0,1]) :: d(X,Y).\nq(X) :- f, d(X,1).\n"
        CTX["txt"] = f"T5 save_state complete={complete}\n{prog}"
        mod = DNet(2, 2, w, b, 1.0)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1)
        net = Network(mod, "net", optimizer=opt)
        m = make_model(prog, [net])
        m.add_tensor_source("src", {(Constant(0),): torch.tensor([0.3, 0.7])})
        qq = Term("q", Term("tensor", Term("src", Constant(0))))
        v0, _ = query_value(m, qq)
        form = FORMS[fdp.ConsumeIntInRange(0, 2)]
        CTX["txt"] += f"\npath form = {form}"
        with in_tmp():
            path = form.format("state.zip")
            m.save_state(path, complete=complete)
            with torch.no_grad():
                for p_ in mod.parameters():
                    p_.add_(1.0)
            m.parameters[0] = 0.123
            m.load_state(path)
            m.hyperparameters_to_file(form.format("hyper.json"))
        v1, _ = query_value(m, qq)
        if abs(as_float(v0) - as_float(v1)) > 1e-6:
            raise Bug(f"save_state/load_state round trip changed the answer {as_float(v0)} -> {as_float(v1)}")
    else:            # foreign predicate
        k = fdp.ConsumeIntInRange(0, 5)
        prog = "h(X,Y) :- sq(X,Y).\n"
        CTX["txt"] = f"T5 foreign k={k}\n{prog}"
        m = Model(prog, [], load=False)
        m.set_engine(ExactEngine(m))
        m.register_foreign(lambda x: Constant(int(x) ** 2), "sq", 1, 1)
        want = k * k
        got, present = query_value(m, Term("h", Constant(k), Constant(want)))
        check_close("foreign", got, 1.0)


# ------------------------------------------------------------------ T6: end-to-end training and evaluation
def t_train(fdp):
    D = 2
    w = [flt(fdp) for _ in range(4)]
    b = [flt(fdp) for _ in range(2)]
    scale = [1.0, 10.0][fdp.ConsumeIntInRange(0, 1)]
    mod = DNet(2, D, w, b, scale)
    net = Network(mod, "net", optimizer=torch.optim.SGD(mod.parameters(), lr=[0.0, 0.1, 10.0][fdp.ConsumeIntInRange(0, 2)]),
                  batching=fdp.ConsumeBool())
    prog = "nn(net,[X],Y,[0,1]) :: d(X,Y).\nt(0.5)::w.\nq(X,Y) :- d(X,Y), w.\n"
    nq = fdp.ConsumeIntInRange(0, 4)
    qs = [Query(Term("q", Term("tensor", Term("src", Constant(fdp.ConsumeIntInRange(0, 1)))), Constant(fdp.ConsumeIntInRange(0, 1))),
                p=prob(fdp)) for _ in range(nq)]
    bs = fdp.ConsumeIntInRange(1, 3)
    shuffle = fdp.ConsumeBool()
    loss_name = ["cross_entropy", "mse"][fdp.ConsumeIntInRange(0, 1)]
    neg = fdp.ConsumeBool()
    lr = [0.0, 0.1, 1.0, 1e3][fdp.ConsumeIntInRange(0, 3)]
    stop_kind = fdp.ConsumeIntInRange(0, 3)
    CTX["txt"] = (f"T6 nq={nq} bs={bs} shuffle={shuffle} loss={loss_name} negatives={neg} lr={lr} stop={stop_kind} "
                  f"queries={[(str(q.query), q.p) for q in qs]}\n{prog}")
    m = make_model(prog, [net])
    m.add_tensor_source("src", {(Constant(0),): torch.tensor([0.3, -0.2]), (Constant(1),): torch.tensor([-0.5, 0.9])})
    m.optimizer = SGD(m, lr)
    stop = {0: 1, 1: EpochStop(2), 2: EpochStop(1) | Threshold("loss", 1e9), 3: EpochStop(1) | StopOnPlateau("loss")}[stop_kind]
    loader = DataLoader(QueryDataset(qs), bs, shuffle=shuffle, seed=3)
    kw = {}
    if fdp.ConsumeBool():
        form = FORMS[fdp.ConsumeIntInRange(0, 2)]
        kw["snapshot_name"] = form.format("snap")
        CTX["txt"] += f"\nsnapshot_name={kw['snapshot_name']}"
    with in_tmp():
        train_model(m, loader, stop, verbose=0, loss_function_name=loss_name, with_negatives=neg, log_iter=1,
                    initial_test=False, **kw)
    for v in m.parameters:
        if math.isnan(v) or v < 0 or v > 1:
            raise Bug(f"parameter invalid after training: {m.parameters}")
    for p_ in mod.parameters():
        if torch.isnan(p_).any():
            raise Bug("network weights are NaN after training")
    if nq:
        cm = get_confusion_matrix(m, QueryDataset(qs[:1]), verbose=0)
        if int(cm.matrix.sum()) != 1:
            raise Bug(f"confusion matrix counted {int(cm.matrix.sum())} items for 1 query")


# ------------------------------------------------------------------ T7: data utilities
class CountingRandom(random.Random):
    def choice(self, seq):
        self.n = getattr(self, "n", 0) + 1
        if GUARDS and self.n > 500:
            raise Bug("NumericIncorrectOutputMutator keeps drawing without ever finding a different value (infinite loop)")
        return super().choice(seq)


def t_data(fdp):
    sub = fdp.ConsumeIntInRange(0, 5)
    if sub == 0:     # QueryDataset text round trip
        qs = []
        for _ in range(fdp.ConsumeIntInRange(1, 3)):
            args = [Constant(fdp.ConsumeIntInRange(0, 5)) for _ in range(fdp.ConsumeIntInRange(1, 3))]
            subst = {}
            if fdp.ConsumeBool():
                args[0] = Var("X")
                subst = {Var("X"): Constant(fdp.ConsumeIntInRange(0, 3))}
            qs.append(Query(Term("f", *args), subst, prob(fdp)))
        CTX["txt"] = f"T7 QueryDataset round trip {[(str(q.query), q.substitution, q.p) for q in qs]}"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.txt")
            with open(path, "w") as f:
                QueryDataset(qs).write_to_file(f)
            back = QueryDataset(path)
        if len(back) != len(qs):
            raise Bug(f"QueryDataset wrote {len(qs)} queries, read back {len(back)}")
        for a_, b_ in zip(qs, back.queries):
            if a_.p != b_.p or a_.query != b_.query or a_.substitution != b_.substitution:
                raise Bug(f"QueryDataset round trip changed {(str(a_.query), a_.p, a_.substitution)} -> {(str(b_.query), b_.p, b_.substitution)}")
    elif sub == 1:   # ground_dataset
        cache = fdp.ConsumeBool()
        prog = "0.3::f(1).\n0.4::f(2).\n"
        CTX["txt"] = f"T7 ground_dataset cache={cache}\n{prog}"
        m = Model(prog, [], load=False)
        m.set_engine(ExactEngine(m), cache=cache)
        out = m.ground_dataset(QueryDataset([Query(Term("f", Constant(1))), Query(Term("f", Constant(2)))]))
        if not isinstance(out, dict):
            raise Bug("ground_dataset did not return a dict")
    elif sub == 2:   # mutator terminates and really changes the output
        dom = [fdp.ConsumeIntInRange(0, 3) for _ in range(fdp.ConsumeIntInRange(1, 3))]
        v = fdp.ConsumeIntInRange(0, 3)
        change_p = fdp.ConsumeBool()
        p = prob(fdp)
        CTX["txt"] = f"T7 mutator domain={dom} value={v} change_p={change_p} p={p}"
        mut = NumericIncorrectOutputMutator(dom, change_p, seed=fdp.ConsumeIntInRange(0, 100))
        import deepproblog.dataset as dd
        orig = dd.random.Random
        dd.random.Random = CountingRandom
        try:
            out = mut(0, Query(Term("f", Constant(v)), p=p))
        finally:
            dd.random.Random = orig
        nv = out.query.args[-1].value
        if nv == v or nv not in dom:
            raise Bug(f"mutated output {nv} (original {v}, domain {dom})")
        if change_p and abs(out.p - (1 - p)) > 1e-12:
            raise Bug("change_p did not set p to 1-p")
    elif sub == 3:   # Logger write/read round trip
        lg = Logger()
        names = ["loss", "acc", "time"]
        for _ in range(fdp.ConsumeIntInRange(0, 4)):
            lg.log(names[fdp.ConsumeIntInRange(0, 2)], fdp.ConsumeIntInRange(0, 20), fdp.ConsumeFloatInRange(-5, 5))
        for _ in range(fdp.ConsumeIntInRange(0, 2)):
            lg.comment(["hello", "", "a\nb"][fdp.ConsumeIntInRange(0, 2)])
        CTX["txt"] = f"T7 Logger data={lg.log_data} comments={lg.comments}"
        form = FORMS[fdp.ConsumeIntInRange(0, 2)]
        CTX["txt"] += f"\npath form = {form}"
        with in_tmp():
            lg.write_to_file(form.format("x"))
            back = Logger()
            back.read_from_file(form.format("x") + ".log")
        for n_ in lg.log_data:
            if back.get_attribute(n_, True) != lg.get_attribute(n_, True):
                raise Bug(f"Logger round trip changed attribute {n_}: {lg.get_attribute(n_, True)} -> {back.get_attribute(n_, True)}")
    elif sub == 4:   # confusion matrix
        given = fdp.ConsumeBool()
        start = ["a", "b"]
        cm = ConfusionMatrix(start) if given else ConfusionMatrix(fdp.ConsumeIntInRange(0, 3))
        items = [(fdp.ConsumeIntInRange(0, 4), fdp.ConsumeIntInRange(0, 4)) for _ in range(fdp.ConsumeIntInRange(0, 6))]
        if given:
            items = [("abcde"[x], "abcde"[y]) for x, y in items]
        CTX["txt"] = f"T7 ConfusionMatrix given_list={given} items={items}"
        for pr, ac in items:
            cm.add_item(pr, ac)
        if int(cm.matrix.sum()) != len(items):
            raise Bug(f"confusion matrix holds {int(cm.matrix.sum())} items but {len(items)} were added")
        if given and start != ["a", "b"]:
            raise Bug(f"ConfusionMatrix modified the caller's class list: {start}")
        if items:
            acc = cm.accuracy()
            if not (0 <= acc <= 1):
                raise Bug(f"accuracy {acc} outside [0,1]")
    else:            # stop conditions
        lg = Logger()
        dur = fdp.ConsumeIntInRange(1, 3)
        cond = [Threshold("loss", fdp.ConsumeFloatInRange(0, 2), duration=dur),
                StopOnPlateau("loss", patience=fdp.ConsumeIntInRange(1, 3), warm_up=fdp.ConsumeIntInRange(0, 3)),
                StopOnNoChange("loss", patience=fdp.ConsumeIntInRange(1, 3), warm_up=fdp.ConsumeIntInRange(0, 3))][fdp.ConsumeIntInRange(0, 2)]
        tobj = types.SimpleNamespace(logger=lg, epoch=0)
        CTX["txt"] = f"T7 stop condition {type(cond).__name__} {vars(cond)}"
        for i in range(fdp.ConsumeIntInRange(0, 6)):
            if fdp.ConsumeBool():
                lg.log("loss", i, fdp.ConsumeFloatInRange(0, 2))
            with contextlib.redirect_stdout(io.StringIO()):
                cond.is_stop(tobj)


# ------------------------------------------------------------------ T8: user predicates named like built-in tensor predicates
BUILTINS = [("embed", 2), ("rbf", 3), ("add", 3), ("mul", 3), ("dot", 3), ("max", 2), ("sigmoid", 2), ("mean", 2),
            ("stack", 2), ("cat", 2), ("one_hot", 3), ("list_to_tensor", 2), ("tensor_index", 3), ("foo", 2), ("bar", 3)]


def t_names(fdp):
    name, ar = BUILTINS[fdp.ConsumeIntInRange(0, len(BUILTINS) - 1)]
    p = prob(fdp)
    args = ",".join(str(fdp.ConsumeIntInRange(0, 3)) for _ in range(ar))
    prog = f"{fmt(p)}::{name}({args}).\n"
    CTX["txt"] = f"T8 user-defined {name}/{ar}\n{prog}"
    m = make_model(prog)
    q = Term(name, *[Constant(int(x)) for x in args.split(",")])
    got, present = query_value(m, q)
    if not present and p > TOL:
        raise Bug(f"user fact {name}/{ar} with probability {p} gets no answer")
    check_close(f"user fact {name}/{ar}", got, p)


TESTS = [t_prop, t_graph, t_neural, t_learn, t_api, t_train, t_data, t_names]
_SKIP = set(filter(None, os.environ.get("DPL_SKIP", "").split(",")))     # dimension ablation: leave whole test families out
TESTS = [t for t in TESTS if t.__name__ not in _SKIP] or TESTS


def TestOneInput(data):
    CTX["data"] = bytes(data)
    fdp = atheris.FuzzedDataProvider(data)
    t = TESTS[fdp.ConsumeIntInRange(0, len(TESTS) - 1)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t(fdp)
    except Bug as e:
        record("oracle:" + t.__name__, str(e))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        record(f"exception:{t.__name__}:{type(e).__name__}", str(e).replace("\n", " "))


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
