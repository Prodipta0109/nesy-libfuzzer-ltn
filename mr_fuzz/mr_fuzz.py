"""
Metamorphic-relation guided fuzzing of LTN's fuzzy operators.

Each fuzz input (bytes) is decoded into
    byte 0              which MR to check
    next 3*N*2 bytes    three vectors x, y, z of N values in [0, 1]   (16 bits per value)
and the chosen MR (see mrs.py) is evaluated. Its violation margin is fed back to libFuzzer
(mrs.bucket), so the fuzzer is steered towards inputs that nearly break, or break, an MR.

A violated MR is written once to findings/<mr name>/  (input.bin + info.txt) and fuzzing goes on.

Run with Atheris (libFuzzer):   python mr_fuzz.py corpus -max_total_time=60
Run without Atheris (random):   python mr_fuzz.py --random 20000
"""
import os
import random
import sys

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.environ.get("LTN_SRC", os.path.join(HERE, "..", "ltn_torch"))))
FINDINGS = os.environ.get("MR_FINDINGS", os.path.join(HERE, "findings"))
N = 4                                           # values per vector

try:
    import atheris
except ImportError:
    atheris = None

if atheris:                                     # instrument only our own code, not torch
    with atheris.instrument_imports(include=["ltn", "mrs"]):
        import ltn.fuzzy_ops
        import mrs
else:
    import mrs

seen = set()                                    # MRs already reported


def decode(data):
    """bytes -> (mr, x, y, z)"""
    data = data + bytes(1 + 6 * N - len(data))  # pad short inputs with zeros
    mr = mrs.MRS[data[0] % len(mrs.MRS)]
    vals = [int.from_bytes(data[1 + 2 * i: 3 + 2 * i], "big") / 65535 for i in range(3 * N)]
    t = torch.tensor(vals)
    return mr, t[:N], t[N:2 * N], t[2 * N:]


def report(mr, data, margin, x, y, z):
    d = os.path.join(FINDINGS, mr.__name__)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "input.bin"), "wb") as f:
        f.write(data)
    with open(os.path.join(d, "info.txt"), "w") as f:
        f.write(f"{mr.__name__}  margin={margin}\nx={x.tolist()}\ny={y.tolist()}\nz={z.tolist()}\n")
    print(f"[VIOLATION] {mr.__name__}  margin={margin:.3g}", file=sys.stderr, flush=True)


def TestOneInput(data):
    mr, x, y, z = decode(bytes(data))
    try:
        margin = mr(x, y, z)
    except Exception as e:                      # an operator crashed on a valid input in [0, 1]
        margin = float("inf")
        print(f"[CRASH] {mr.__name__}: {e!r}", file=sys.stderr)
    if margin != margin:                        # NaN
        margin = float("inf")
    mrs.bucket(margin)                          # <- the guidance
    if margin > mrs.TOL and mr.__name__ not in seen:
        seen.add(mr.__name__)
        report(mr, bytes(data), margin, x, y, z)


def main():
    if "--random" in sys.argv:                  # no Atheris: plain random inputs
        for _ in range(int(sys.argv[sys.argv.index("--random") + 1])):
            TestOneInput(random.randbytes(1 + 6 * N) if hasattr(random, "randbytes")
                         else bytes(random.getrandbits(8) for _ in range(1 + 6 * N)))
        print(f"MRs violated: {sorted(seen) or 'none'}")
        return
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
