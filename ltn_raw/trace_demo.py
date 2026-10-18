#!/usr/bin/env python3
"""Visual trace of ONE fuzzer input and ONE mutation, using the real harness (ltn_fuzz/ltn_fuzz.py) and real LTN.

It shows: the bytes, which byte answered which question, the LTN program that was built, the values handed to
LTN, which LTN lines ran, and what changed when ONE byte was mutated.

Not real Atheris: a small transparent reader replaces atheris.FuzzedDataProvider so each byte read can be printed.
Atheris maps bytes to numbers differently (it reads some values from the end of the buffer and uses several bytes
for a float), so the exact byte -> number mapping differs; the mechanism is the same.

run:  python trace_demo.py            (needs torch, numpy, coverage; no atheris)
"""
import inspect
import os
import random
import sys
import types
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ltn_torch"))
sys.path.insert(0, os.path.join(HERE, "..", "ltn_fuzz"))

READS = []                      # (position, byte, harness function, harness line, question, decision)


class TracingReader:
    """Reads the input left to right, one byte per question, and remembers every read."""

    def __init__(self, data):
        self.d, self.p = bytes(data), 0

    def _next(self, what, fmt):
        b = self.d[self.p] if self.p < len(self.d) else 0
        pos, self.p = self.p, self.p + 1
        fr = next(f for f in inspect.stack()[2:] if f.filename.endswith("ltn_fuzz.py"))
        return b, pos, fr.function, fr.lineno, what, fmt

    def ConsumeIntInRange(self, lo, hi):
        b, pos, fn, ln, what, _ = self._next("int", None)
        v = lo + b % (hi - lo + 1)
        READS.append((pos, b, fn, ln, f"integer in [{lo},{hi}]", v))
        return v

    def ConsumeBool(self):
        b, pos, fn, ln, what, _ = self._next("bool", None)
        v = bool(b & 1)
        READS.append((pos, b, fn, ln, "yes/no", v))
        return v

    def ConsumeFloatInRange(self, lo, hi):
        b, pos, fn, ln, what, _ = self._next("float", None)
        v = lo + (hi - lo) * b / 255.0
        READS.append((pos, b, fn, ln, f"decimal in [{lo},{hi}]", round(v, 3)))
        return v


fake = types.ModuleType("atheris")
fake.FuzzedDataProvider = TracingReader
fake.instrument_imports = lambda include=None: __import__("contextlib").nullcontext()
fake.Setup = fake.Fuzz = lambda *a, **k: None
sys.modules["atheris"] = fake
os.environ["LTNFUZZ_NO_INSTRUMENT"] = "1"
os.environ["LTN_SRC"] = os.path.join(HERE, "ltn_torch")

import coverage  # noqa: E402

LTN_DIR = os.path.join(HERE, "ltn_torch", "ltn")


def run(data):
    """Run the real Builder on `data`. Returns (reads, program lines, outcome, executed LTN lines)."""
    import importlib
    F = sys.modules.get("ltn_fuzz") or importlib.import_module("ltn_fuzz")
    READS.clear()
    cov = coverage.Coverage(data_file=None, source=[LTN_DIR])
    cov.start()
    b = F.Builder(data)
    try:
        b.run()
        outcome = "ran clean"
    except Exception as e:  # noqa: BLE001
        outcome = ("EXPECTED REJECTION by LTN" if F.is_expected(e) else "FINDING  ") + f" {type(e).__name__}: {str(e)[:90]}"
    cov.stop()
    d = cov.get_data()
    lines = {(os.path.basename(f), n) for f in d.measured_files() for n in (d.lines(f) or [])}
    return list(READS), list(b.trace), outcome, lines


def show(title, data, reads, trace, outcome):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print("input bytes (position: value):")
    print("  " + " ".join(f"{i}:{x:02x}" for i, x in enumerate(data[:max(r[0] for r in reads) + 1])))
    print("\nhow the harness used them (each byte answers one question):")
    print(f"  {'pos':>3} {'byte':>4}  {'asked by (ltn_fuzz.py)':28} {'question':26} answer")
    for pos, b, fn, ln, q, v in reads:
        print(f"  {pos:>3} {b:>4}  {fn + ':' + str(ln):28} {q:26} {v}")
    print("\nthe LTN program that was built from those answers:")
    for t in trace:
        print("  " + t)
    print(f"\nresult: {outcome}")


def find_pair():
    """Find an input that runs clean and a ONE-byte change that makes a variable empty and breaks LTN."""
    import importlib
    F = importlib.import_module("ltn_fuzz")
    rng = random.Random(7)
    for _ in range(4000):
        data = bytes(rng.randrange(256) for _ in range(90))
        reads, trace, out, lines = run(data)
        if "ran clean" not in out or not trace or not trace[0].startswith("x = "):
            continue
        # the read that decided the number of individuals of the first variable is the 2nd read (after n_vars)
        n_reads = [r for r in reads if r[2] == "make_domain" and "[0,5]" in r[4]]
        if not n_reads or n_reads[0][5] == 0:
            continue
        pos = n_reads[0][0]
        mutated = bytearray(data)
        mutated[pos] = 0                                  # the ONE-byte mutation: 0 individuals
        r2, t2, out2, lines2 = run(bytes(mutated))
        if out2.startswith("FINDING") and "outside [0,1]" in out2:
            return data, pos, bytes(mutated), (reads, trace, out, lines), (r2, t2, out2, lines2)
    raise SystemExit("no suitable pair found in 4000 tries")


def source_line(fname, n):
    path = os.path.join(LTN_DIR, fname)
    return open(path).read().splitlines()[n - 1].strip()[:90]


if __name__ == "__main__":
    data, pos, mutated, A, B = find_pair()
    show("INPUT A  (a saved input that ran fine)", data, *A[:3])
    print(f"\n{'#' * 100}\nMUTATION: libFuzzer changes ONE byte: position {pos}: "
          f"0x{data[pos]:02x} ({data[pos]}) -> 0x00 (0)\n{'#' * 100}")
    show("INPUT B  (A with that one byte changed)", mutated, *B[:3])

    only_b = sorted(B[3] - A[3])
    only_a = sorted(A[3] - B[3])
    print(f"\n{'=' * 100}\nWHICH LTN LINES CHANGED BETWEEN THE TWO RUNS\n{'=' * 100}")
    print(f"LTN lines executed in A: {len(A[3])}   in B: {len(B[3])}")
    print(f"\nlines that ran ONLY in B  ({len(only_b)})  <- new coverage: libFuzzer would KEEP input B")
    for f, n in only_b[:14]:
        print(f"  {f}:{n:<5} {source_line(f, n)}")
    print(f"\nlines that ran ONLY in A  ({len(only_a)})")
    for f, n in only_a[:8]:
        print(f"  {f}:{n:<5} {source_line(f, n)}")
