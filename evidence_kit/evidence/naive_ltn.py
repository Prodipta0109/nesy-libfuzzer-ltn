"""NAIVE LTN fuzz target for A3: one fixed formula Forall x P(x), bytes -> 3 numbers, crash-only."""
import sys, os, warnings
warnings.filterwarnings("ignore")
import atheris
sys.path.insert(0, os.environ.get("LTN_SRC", "/tmp/rv/ltn_torch"))
import torch
with atheris.instrument_imports(include=["ltn"]):
    import ltn
def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    x = ltn.Variable("x", torch.tensor([[fdp.ConsumeFloatInRange(-10, 10)] for _ in range(3)]))
    P = ltn.Predicate(func=lambda v: torch.sigmoid(v.sum(dim=1)))
    ltn.Connective(ltn.fuzzy_ops.NotStandard())(P(x))
    ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")(x, P(x))
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput); atheris.Fuzz()
