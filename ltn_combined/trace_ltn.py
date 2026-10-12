"""pytest plugin: record the inputs every fuzzy operator / aggregator receives while LTNtorch's own tests run.

Usage: (cd frameworks/LTNtorch && PYTHONPATH=../../fuzz:. pytest tests/tests.py -p trace_ltn)  -> results/seeds/ltn_seeds.json
"""
import json
import os
import pytest

OUT = os.environ.get("NESY_LTN_SEED_OUT") or os.path.join(os.path.dirname(__file__), "..", "results", "seeds", "ltn_seeds.json")
RECORDS = []
CURRENT = {"test": None}


def _tolist(x):
    try:
        return x.detach().cpu().tolist()
    except Exception:
        return None


def _install():
    import torch
    import ltn.fuzzy_ops as F
    classes = [c for n, c in vars(F).items() if isinstance(c, type) and issubclass(c, (F.ConnectiveOperator, F.AggregationOperator))
               and c not in (F.ConnectiveOperator, F.UnaryConnectiveOperator, F.BinaryConnectiveOperator, F.AggregationOperator)
               and "__call__" in c.__dict__ and c.__name__ != "SatAgg"]
    for cls in classes:
        orig = cls.__call__

        def make(orig, cls):
            def wrapped(self, *args, **kwargs):
                rec = {"test": CURRENT["test"], "op": cls.__name__,
                       "ctor": {k: getattr(self, k) for k in ("stable", "p") if hasattr(self, k)},
                       "args": [_tolist(a) if isinstance(a, torch.Tensor) else a for a in args],
                       "dtype": str(args[0].dtype) if args and isinstance(args[0], torch.Tensor) else None,
                       "kwargs": {k: (_tolist(v) if isinstance(v, torch.Tensor) else (list(v) if isinstance(v, tuple) else v))
                                  for k, v in kwargs.items()}}
                RECORDS.append(rec)
                return orig(self, *args, **kwargs)
            return wrapped
        cls.__call__ = make(orig, cls)


def pytest_configure(config):
    _install()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    CURRENT["test"] = item.nodeid
    yield


def pytest_sessionfinish(session, exitstatus):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seen, out = set(), []
    for r in RECORDS:
        key = json.dumps([r["op"], r["ctor"], r["args"], r["kwargs"]], sort_keys=True, default=str)
        if key not in seen and all(a is not None for a in r["args"]):
            seen.add(key)
            out.append(r)
    json.dump(out, open(OUT, "w"))
