"""NAIVE DeepProbLog fuzz target for A3: one fixed program, bytes -> one probability, crash-only."""
import sys, os, warnings
warnings.filterwarnings("ignore")
import atheris
with atheris.instrument_imports(include=["deepproblog", "problog"]):
    import deepproblog
    from deepproblog.model import Model
    from deepproblog.engines import ExactEngine
    from deepproblog.query import Query
    from problog.logic import Term, Constant
import torch

def one(data):
    fdp = atheris.FuzzedDataProvider(data)
    p = fdp.ConsumeProbability()
    m = Model("%.6f::a. %.6f::b. q :- a, b." % (p, fdp.ConsumeProbability()), [], load=False)
    m.set_engine(ExactEngine(m))
    m.solve([Query(Term("q"))])

def TestOneInput(data):
    one(data)

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput); atheris.Fuzz()
