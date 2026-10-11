import warnings, random, io, contextlib, sys; warnings.filterwarnings("ignore")
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.query import Query
rng=random.Random(int(sys.argv[1]) if len(sys.argv)>1 else 1)
def prog():
    n=rng.randint(2,4); L=[]
    for i in range(n): L.append(f"{rng.choice([0.1,0.5,0.9,rng.random()]):.6f}::num({i},{rng.randint(0,3)}).")
    L.append("big(X) :- num(X,V), V > 1.")
    L.append("pair(X,Y) :- num(X,A), num(Y,B), X < Y, A =:= B.")
    L.append("sumv(S) :- num(0,A), num(1,B), S is A+B.")
    L.append("bothbig :- big(0), big(1).")
    L.append("nobig :- \\+ big(0), \\+ big(1).")
    L.append("either :- big(0) ; big(1).")
    L.append("cnt(N) :- findall(X, big(X), Xs), length(Xs, N).")
    return "\n".join(L), n
bad=0
for it in range(int(sys.argv[2]) if len(sys.argv)>2 else 60):
    text,n=prog()
    qs=["bothbig","nobig","either","pair(0,1)","sumv(2)","sumv(3)","cnt(0)","cnt(1)","cnt(2)"]
    ref=get_evaluatable().create_from(PrologString(text+"\n"+"\n".join(f"query({q})." for q in qs))).evaluate()
    ref={str(k):v for k,v in ref.items()}
    try:
        m=Model(text,[],load=False); m.set_engine(ExactEngine(m),cache=False)
        for q in qs:
            from problog.parser import PrologParser
            t=PrologString(q+".")
            term=list(t)[0]
            r=m.solve([Query(term)])[0].result
            got=float(r.get(term,0.0)) if len(r) else 0.0
            want=float(ref.get(str(term),0.0))
            if abs(got-want)>1e-6:
                bad+=1; print("MISMATCH",q,got,want); print(text); break
    except Exception as e:
        bad+=1; print("EXC",type(e).__name__,str(e)[:100]); print(text); break
print("done, mismatches:",bad)
