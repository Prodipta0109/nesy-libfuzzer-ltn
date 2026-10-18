#!/usr/bin/env python3
"""
Raw libFuzzer (Atheris) run on LTNtorch with NO dimensions and NO program builder.

The bytes libFuzzer produces are used as they are: they are reinterpreted as float32 numbers and handed
straight to LTN's public operators. Nothing chooses a shape, a value list, a formula, an aggregator or a
p value. libFuzzer's only guide is code coverage of the `ltn` package; the only oracle is "an exception that
is not one of LTN's own validation errors".

What still has to exist (it cannot be avoided): something must call LTN. Here that is a loop over every
public operator and aggregator, each called with the raw tensor. No choice is read from the bytes except
one byte at the front that selects which operator to call.

Run:   python ltn_raw_fuzz.py corpus -max_total_time=300 -rss_limit_mb=4000
"""
import hashlib
import json
import os
import re
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "deps"))      # atheris installed with --target deps (if used)
sys.path.insert(0, os.path.join(HERE, "ltn_torch"))

import atheris  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

with atheris.instrument_imports(include=["ltn"]):
    import ltn  # noqa: E402
    import ltn.fuzzy_ops as fo  # noqa: E402

FINDINGS = os.environ.get("RAW_FINDINGS", os.path.join(HERE, "findings"))
os.makedirs(FINDINGS, exist_ok=True)
SEEN = set()

# LTN's own validation messages: LTN correctly saying "no". Same list as the structured harness.
EXPECTED_PREFIXES = (
    "Expected the output of a predicate", "Expected inputs of connectives and quantifiers",
    "Expected one operand", "Expected two operands", "Expected parameter",
    "Expected the given LTN variables", "LTNObject()", "Variable()", "Predicate()", "Function()",
    "Connective()", "Quantifier()", "SatAgg()", "Labels starting with 'diag_'", "Both model and func",
    "Since 'cond_", "'xs' and 'mask' must have the same shape",
)


def expected(exc):
    return isinstance(exc, (ValueError, TypeError)) and str(exc).startswith(EXPECTED_PREFIXES)


def signature(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if os.sep + "ltn" + os.sep in f.filename]
    fr = frames[-1] if frames else (tb[-1] if tb else None)
    where = f"{os.path.basename(fr.filename)}:{fr.lineno}" if fr else "?"
    return f"{type(exc).__name__}@{where}: " + re.sub(r"[-+]?\d[\d.e+-]*", "N", str(exc))[:120]


def save(sig, data, label):
    if sig in SEEN:
        return
    SEEN.add(sig)
    d = os.path.join(FINDINGS, hashlib.sha1(sig.encode()).hexdigest()[:10])
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "input.bin"), "wb").write(data)
    json.dump({"signature": sig, "call": label}, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"\n[NEW FINDING] {label}: {sig}", file=sys.stderr, flush=True)


def build_calls():
    """Every public operator class of fuzzy_ops, constructed with its default arguments."""
    calls = []
    for name in dir(fo):
        cls = getattr(fo, name)
        if not (isinstance(cls, type) and cls.__module__ == fo.__name__):
            continue
        # if name.startswith("_") or name in ("ConnectiveOperator", "UnaryConnectiveOperator",
        #                                     "BinaryConnectiveOperator", "AggregationOperator"):
        #     continue
        try:
            op = cls()
        except TypeError:
            continue                      # needs arguments (e.g. Equiv, SatAgg); handled below
        calls.append((name, op))
    return calls


CALLS = build_calls()


def TestOneInput(data):
    if len(data) < 5:
        return
    sel, raw = data[0], data[1:]
    t = torch.from_numpy(np.frombuffer(raw[: len(raw) // 4 * 4], dtype=np.float32).copy())
    if t.numel() == 0:
        return
    half = t.numel() // 2
    a, b = (t[:half], t[half:2 * half]) if half else (t, t)

    name, op = CALLS[sel % len(CALLS)]
    attempts = [(f"{name}(a)", lambda: op(a)),
                (f"{name}(a, b)", lambda: op(a, b)),
                (f"{name}(a, dim=0)", lambda: op(a, dim=0)),
                (f"SatAgg({name})(a)", lambda: fo.SatAgg(op)(a)),
                (f"Variable(a)", lambda: ltn.Variable("x", t))]
    for label, fn in attempts:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            if not expected(exc):
                save(signature(exc), data, label)


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
