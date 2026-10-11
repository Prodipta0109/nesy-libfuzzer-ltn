"""LNN: for every dimension, a control input and the fuzzer-chosen input that differ ONLY in that dimension.
  python3 pairs_lnn.py"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("LNN_DIR", "/home/claude/lnn") + "/src")
from evidence_core import run_pairs
from lnn import *

def and_infer(weights, bdata):
    def f():
        A, B = Propositions("A", "B")
        g = And(A, B, activation={"weights": weights, "bias": 1.0, "weights_learning": False, "bias_learning": False})
        m = Model(); m.add_knowledge(g); m.add_data({B: bdata})
        steps, _ = m.infer(max_steps=60)
        assert steps < 60, "infer() used all 60 steps: never converges"
        return "%d steps" % steps
    return f

def forall_data(sb):
    def f():
        x = Variable("x"); S = Predicate("S"); g = Forall(x, S(x)); m = Model(); m.add_knowledge(g)
        m.add_data({S: {"a": Fact.TRUE, "b": sb}}); m.infer(max_steps=5)
        assert not m.has_contradiction(), "consistent data reported as contradiction"
    return f

def nest(inner_exists):
    def f():
        x, y = Variables("x", "y"); R = Predicate("R", arity=2); S = Predicate("S"); T = Predicate("T")
        Forall(x, Implies(S(x), Exists(y, R(x, y)) if inner_exists else T(x)))
    return f

def loss(connective):
    def f():
        A, B = Propositions("A", "B"); g = connective(A, B); m = Model(); m.add_knowledge(g)
        m.add_data({A: Fact.TRUE, B: Fact.FALSE}); m.add_labels({g: Fact.TRUE}); m.train(losses=[Loss.LOGICAL], epochs=2, pbar=False)
    return f

def second_infer(quant):
    def f():
        if quant:
            x = Variable("x"); S = Predicate("S"); g = Exists(x, S(x)); m = Model(); m.add_knowledge(g); m.add_data({S: {"a": Fact.TRUE}})
        else:
            A, B = Propositions("A", "B"); g = And(A, B); m = Model(); m.add_knowledge(g); m.add_data({A: Fact.TRUE, B: Fact.TRUE})
        m.infer(max_steps=5); _, upd = m.infer(max_steps=5)
        assert float(upd) == 0.0, "second infer() on an unchanged model still reports updates (%s)" % float(upd)
    return f

def reflexive(refl):
    def f():
        x, y = Variables("x", "y"); R = Predicate("R", arity=2); S = Predicate("S")
        g = Or(S(x), R(x, x) if refl else R(x, y)); m = Model(); m.add_knowledge(g)
        m.add_data({S: {"a": Fact.TRUE}, R: {("a", "a"): Fact.FALSE}}); m.infer(max_steps=5)
    return f

PAIRS = [
 ("L1 formula structure", "L7", "quantifier nested under a connective", nest(False), nest(True)),
 ("L2 truth bounds / data", "L1", "operand interval (0.3,0.5) -> exact point (0.4,0.4)", and_infer((1.0, 0.5), (0.3, 0.5)), and_infer((1.0, 0.5), (0.4, 0.4))),
 ("L3 first-order structure", "L6", "S(b)=true -> S(b)=false", forall_data(Fact.TRUE), forall_data(Fact.FALSE)),
 ("L4 neuron configuration", "L1", "weights (1,1) -> (1,0.5)", and_infer((1.0, 1.0), (0.4, 0.4)), and_infer((1.0, 0.5), (0.4, 0.4))),
 ("L5 inference control", "L2", "And -> Exists, then infer() twice", second_infer(False), second_infer(True)),
 ("L6 learning", "L4", "loss on And -> loss on Iff", loss(And), loss(Iff)),
 ("L7 API / grounding table", "L8", "R(x,y) -> reflexive R(x,x)", reflexive(False), reflexive(True)),
]
if __name__ == "__main__": run_pairs("PAIRS_lnn", PAIRS)
