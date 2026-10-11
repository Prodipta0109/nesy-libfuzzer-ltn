"""A deliberately NAIVE LNN fuzz target for the A3/A4 experiments: one fixed formula And(A,B), bytes -> two truth
values, no structure decoding, no oracle (it only notices crashes)."""
import sys, os, warnings
warnings.filterwarnings("ignore")
import atheris
with atheris.instrument_imports(include=["lnn"]):
    import lnn
    from lnn import *
import torch

def one(data):
    fdp = atheris.FuzzedDataProvider(data)
    A, B = Propositions("A", "B"); f = And(A, B)
    m = Model(); m.add_knowledge(f)
    lo = fdp.ConsumeProbability(); hi = fdp.ConsumeProbability()
    m.add_data({A: (min(lo, hi), max(lo, hi))})
    m.infer(max_steps=60)

def TestOneInput(data):
    try: one(data)
    except Exception: raise

if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput); atheris.Fuzz()
