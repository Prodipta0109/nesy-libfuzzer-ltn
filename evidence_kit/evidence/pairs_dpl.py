"""DeepProbLog: for every dimension, a control input and the fuzzer-chosen input that differ ONLY in that dimension.
  python3 pairs_dpl.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("DPL_DIR", "/home/claude/dpl"))
from evidence_core import run_pairs
from check_dpl_bugs import *          # model(), Numbers, torch, nn, Term, Constant, Query, SGD, ...

def fact_name(name):
    def f():
        m = model("0.5::%s(1,2).\n" % name); r = list(m.solve([Query(Term(name, Constant(1), Constant(2)))])[0].result.values())
        return "P=%s" % r
    return f
def tiny_prob(p):
    def f():
        m = model("%s::a.\n" % p); got = list(m.solve([Query(Term("a"))])[0].result.values())
        assert got and abs(got[0] - float(p)) < float(p) * 1e-6, "P(a)=%s but the program says %s" % (got, p)
    return f
def topk(logits):
    def f():
        class Net(nn.Module):
            def forward(self, x): return torch.softmax(torch.tensor(logits), -1)
        net = Network(Net(), "net", k=2)
        m = Model("nn(net,[X],Y,[0,1]) :: d(X,Y).\ns(X,Y) :- d(X,Y).\n", [net], load=False); m.set_engine(ExactEngine(m))
        m.add_tensor_source("src", {(Constant(0),): torch.tensor([1.0, 0.0])})
        m.solve([Query(Term("s", Term("tensor", Term("src", Constant(0))), Constant(0)))])
    return f
def hashing(use_query):
    def f():
        hash(Query(Term("f", Constant(1)))) if use_query else hash(Term("f", Constant(1)))
    return f
def ground(cache):
    def f():
        m = model("0.3::f(1).\n", cache=cache); m.ground_dataset(QueryDataset([Query(Term("f", Constant(1)))]))
    return f
def lr_step(lr):
    def f():
        m = model("t(0.1)::a; t(0.1)::b.\nq :- a.\nq :- b.\n"); m.optimizer = SGD(m, lr); q = Term("q")
        r = m.solve([Query(q)])[0]; GraphSemiring.cross_entropy(r, 0.0, 1.0, q=q); m.optimizer.step()
    return f
def save_as(name):
    def f():
        m = model("0.3::a.\n"); old = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            try: m.save_state(name)
            finally: os.chdir(old)
    return f
def loader(bs):
    def f():
        for c, _ in enumerate(DataLoader(Numbers(3), bs, shuffle=False), 1): assert c <= 100, "still yielding after %d batches" % c
    return f
PAIRS = [
 ("P1 program / term names", "D11", "fact name foo -> max (a torch built-in)", fact_name("foo"), fact_name("max")),
 ("P2 probability values", "D12", "1e-10::a -> 1e-20::a", tiny_prob("1e-10"), tiny_prob("1e-20")),
 ("P3 neural predicate outputs", "D7", "softmax [0.9,0.1] -> [1.0,0.0]", topk([2.0, 0.0]), topk([200.0, 0.0])),
 ("P4 queries", "D1", "hash(Term) -> hash(Query)", hashing(False), hashing(True)),
 ("P5 solver settings", "D3", "ground_dataset cache=True -> default cache=False", ground(True), ground(False)),
 ("P6 learning", "D6", "SGD learning rate 0.01 -> 10", lr_step(0.01), lr_step(10.0)),
 ("P7a API: file name form", "D4", "save_state('./state.zip') -> 'state.zip'", save_as("./state.zip"), save_as("state.zip")),
 ("P7b API: loader arguments", "D2", "DataLoader batch_size 1 -> 0", loader(1), loader(0)),
]
if __name__ == "__main__": run_pairs("PAIRS_dpl", PAIRS)
