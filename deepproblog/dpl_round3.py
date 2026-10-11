"""Round 2/3 harness for DeepProbLog: metamorphic oracles and first-order programs.
Same engine (libFuzzer via Atheris), coverage-only guidance, same record()/replay as dpl_fuzz."""
import os, sys, re, io, math, itertools, contextlib, random
import dpl_fuzz as H
from dpl_fuzz import *     # Bug, record, CTX, gen_prop, prop_oracle, make_model, query_value, check_close, DNet, flt, prob, fmt, Term, Constant, Var, Query, Model, Network, torch, atheris ...

def shuffled(lines, fdp):
    lines = list(lines)
    for i in range(len(lines) - 1, 0, -1):
        j = fdp.ConsumeIntInRange(0, i)
        lines[i], lines[j] = lines[j], lines[i]
    return lines

def t_meta_prog(fdp):
    prog, n, facts, ads, rules, q = H.gen_prop(fdp)
    k = fdp.ConsumeIntInRange(0, 4)
    want = H.prop_oracle(n, facts, ads, rules, q)
    lines = prog.split("\n")
    qname = f"a{q}"
    if k == 0:                                   # clause order does not matter
        new = shuffled(lines, fdp)
    elif k == 1:                                 # rename atoms a<i> to p(<i>)
        new = [re.sub(r"\ba(\d+)\b", r"p(\1)", l) for l in lines]; qname = f"p({q})"
    elif k == 2:                                 # unused probabilistic facts do not change the answer
        new = lines + [f"{fmt(H.prob(fdp))}::z{i}." for i in range(fdp.ConsumeIntInRange(1, 3))]
    elif k == 3:                                 # a duplicated deterministic clause does not change the answer
        det = [l for l in lines if ":-" in l and "::" not in l]
        new = lines + (det[:1] if det else [])
    else:                                        # a trivially-true extra rule for a fresh predicate
        new = lines + ["extra :- true.", f"other :- {qname}, extra."]
    text = "\n".join(new)
    CTX["txt"] = f"META k={k} query={qname}\n{text}"
    m = make_model(text)
    got, present = query_value(m, Term(qname) if k != 1 else Term("p", Constant(q)))
    if not present and want > TOL:
        raise Bug(f"metamorphic k={k}: no answer but oracle {want}")
    check_close(f"metamorphic k={k} {qname}", got, want)

def t_fo(fdp):
    nc = fdp.ConsumeIntInRange(2, 3)
    cs = [f"c{i}" for i in range(nc)]
    pf = [H.prob(fdp) for _ in cs]; pg = [H.prob(fdp) for _ in cs]
    rule = fdp.ConsumeIntInRange(0, 3)
    body = {0: "f(X), \\+ g(X)", 1: "f(X), g(X)", 2: "f(X)", 3: "g(X), \\+ f(X)"}[rule]
    extra = "h(X) :- g(X).\n" if fdp.ConsumeBool() and rule != 3 else ""
    prog = "".join(f"{fmt(p)}::f({c}).\n{fmt(g)}::g({c}).\n" for c, p, g in zip(cs, pf, pg)) + f"h(X) :- {body}.\n{extra}some :- h(_).\n"
    CTX["txt"] = f"FO rule={rule}\n{prog}"
    # oracle: enumerate worlds of the 2*nc facts
    def hv(i, F_, G_):
        r = {0: F_[i] and not G_[i], 1: F_[i] and G_[i], 2: F_[i], 3: (not F_[i]) and G_[i]}[rule]
        return r or (bool(extra) and G_[i])
    pa = [0.0] * nc; psome = 0.0
    for fw in itertools.product([0, 1], repeat=nc):
        for gw in itertools.product([0, 1], repeat=nc):
            w = 1.0
            for i in range(nc):
                w *= (pf[i] if fw[i] else 1 - pf[i]) * (pg[i] if gw[i] else 1 - pg[i])
            hs = [hv(i, fw, gw) for i in range(nc)]
            for i in range(nc):
                if hs[i]: pa[i] += w
            if any(hs): psome += w
    m = make_model(prog, cache=fdp.ConsumeBool())
    for i, c in enumerate(cs):
        got, present = query_value(m, Term("h", Term(c)))
        if not present and pa[i] > TOL: raise Bug(f"h({c}) returned no answer; oracle {pa[i]}")
        check_close(f"h({c})", got, pa[i])
    got, present = query_value(m, Term("some"))
    if not present and psome > TOL: raise Bug(f"some returned no answer; oracle {psome}")
    check_close("some", got, psome)
    # non-ground query: every answer must equal the ground query
    res = m.solve([Query(Term("h", Var("X")))])[0].result
    for term, val in res.items():
        c = str(term.args[0])
        if c in cs and abs(as_float(val) - pa[cs.index(c)]) > TOL:
            raise Bug(f"non-ground query h(X): {term} -> {as_float(val)} but ground query gives {pa[cs.index(c)]}")
    for i, c in enumerate(cs):
        if pa[i] > TOL and not any(str(t.args[0]) == c for t in res):
            raise Bug(f"non-ground query h(X) lacks the answer h({c}) (probability {pa[i]})")

def t_batch(fdp):
    prog, n, facts, ads, rules, q = H.gen_prop(fdp)
    cache = fdp.ConsumeBool()
    qs = [fdp.ConsumeIntInRange(0, n - 1) for _ in range(fdp.ConsumeIntInRange(1, 4))]
    CTX["txt"] = f"BATCH cache={cache} queries={qs}\n{prog}"
    m = make_model(prog, cache=cache)
    batch = m.solve([Query(Term(f"a{i}")) for i in qs])
    m2 = make_model(prog, cache=not cache)
    for i, r in zip(qs, batch):
        single, present = query_value(m2, Term(f"a{i}"))
        bv = r.result.get(Term(f"a{i}"))
        bvf = as_float(bv) if bv is not None else 0.0
        if abs(bvf - as_float(single)) > 1e-9:
            raise Bug(f"batch answer {bvf} differs from single-query answer {as_float(single)} for a{i}")
    again = m.solve([Query(Term(f"a{i}")) for i in qs])
    for r1, r2 in zip(batch, again):
        v1 = [as_float(v) for v in r1.result.values()]; v2 = [as_float(v) for v in r2.result.values()]
        if v1 != v2: raise Bug(f"repeated batch changed the answers: {v1} vs {v2}")

def t_domain_perm(fdp):
    D = fdp.ConsumeIntInRange(2, 4)
    w = [flt(fdp) for _ in range(D * 2)]; b = [flt(fdp) for _ in range(D)]
    x = torch.tensor([flt(fdp, -1, 1), flt(fdp, -1, 1)])
    perm = list(range(D))
    for i in range(D - 1, 0, -1):
        j = fdp.ConsumeIntInRange(0, i); perm[i], perm[j] = perm[j], perm[i]
    target = fdp.ConsumeIntInRange(0, D - 1)
    base = DNet(2, D, w, b, 1.0)
    W = torch.tensor(w, dtype=torch.float32).reshape(D, 2); B = torch.tensor(b, dtype=torch.float32)
    # the same network with output columns permuted, and the domain listed in the matching permuted order
    permuted = DNet(2, D, W[perm].flatten().tolist(), B[perm].tolist(), 1.0)
    dom1 = ",".join(map(str, range(D))); dom2 = ",".join(str(p) for p in perm)
    def prob_of(net_mod, dom):
        net = Network(net_mod, "net", batching=False)
        m = make_model(f"nn(net,[X],Y,[{dom}]) :: d(X,Y).\nq(X) :- d(X,{target}).\n", [net], cache=False)
        m.add_tensor_source("src", {(Constant(0),): x})
        return query_value(m, Term("q", Term("tensor", Term("src", Constant(0)))))[0]
    CTX["txt"] = f"DOMPERM D={D} perm={perm} target={target}"
    a, c = as_float(prob_of(base, dom1)), as_float(prob_of(permuted, dom2))
    if abs(a - c) > 1e-6:
        raise Bug(f"permuting the output domain and the network columns together changed P(d(x,{target})): {a} vs {c}")


def t_fo_ad(fdp):
    """annotated disjunctions and probabilistic rules with variables, over 2-3 constants"""
    nc = fdp.ConsumeIntInRange(2, 3)
    cs = [f"c{i}" for i in range(nc)]
    cols = ["red", "green", "blue"][:fdp.ConsumeIntInRange(2, 3)]
    ps, left = [], 1.0
    for _ in cols[:-1]:
        p = min(left * H.prob(fdp), left); ps.append(p); left = max(left - p, 0.0)
    ps.append(max(0.0, min(left * H.prob(fdp), left)))
    while sum(ps) > 1.0: ps[-1] = math.nextafter(ps[-1], 0.0)
    pr = H.prob(fdp)                                   # probabilistic rule
    obj = "".join(f"obj({c}).\n" for c in cs)
    ad = "; ".join(f"{fmt(p)}::color(X,{c})" for p, c in zip(ps, cols)) + " :- obj(X).\n"
    kind = fdp.ConsumeIntInRange(0, 2)
    if kind == 0:
        rule = "same :- color(c0,C), color(c1,C).\n"
    elif kind == 1:
        rule = f"{fmt(pr)}::pick(X) :- color(X,{cols[0]}).\nsome :- pick(_).\n"
    else:
        rule = f"nored(X) :- obj(X), \\+ color(X,{cols[0]}).\nall :- nored(c0), nored(c1).\n"
    prog = obj + ad + rule
    CTX["txt"] = f"FOAD kind={kind}\n{prog}"
    probs = ps + [max(0.0, 1 - sum(ps))]               # last outcome: no colour
    outcomes = cols + [None]
    names = ["same"] if kind == 0 else ["some"] if kind == 1 else ["all"]
    want = 0.0
    for combo in itertools.product(range(len(outcomes)), repeat=nc):
        w = 1.0
        for k_ in combo: w *= probs[k_]
        col = [outcomes[k_] for k_ in combo]
        if kind == 0:
            hit = col[0] is not None and col[0] == col[1]
            want += w if hit else 0.0
        elif kind == 1:
            n_red = sum(1 for c_ in col if c_ == cols[0])
            want += w * (1 - (1 - pr) ** n_red)
        else:
            want += w if (col[0] != cols[0] and col[1] != cols[0]) else 0.0
    m = make_model(prog, cache=fdp.ConsumeBool())
    got, present = query_value(m, Term(names[0]))
    if not present and want > TOL: raise Bug(f"{names[0]}: no answer but oracle {want}")
    check_close(names[0], got, want)

TESTS3 = [t_meta_prog, t_fo, t_batch, t_domain_perm, t_fo_ad]

def TestOneInput(data):
    CTX["data"] = bytes(data)
    fdp = atheris.FuzzedDataProvider(data)
    t = TESTS3[fdp.ConsumeIntInRange(0, len(TESTS3) - 1)]
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
