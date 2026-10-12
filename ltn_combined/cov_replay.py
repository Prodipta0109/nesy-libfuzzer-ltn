#!/usr/bin/env python3
"""Replay one corpus folder through the combined harness (no libFuzzer) so coverage.py can measure `ltn`.

usage (valid corpus):   coverage run -p --branch --source=<ltn dir> cov_replay.py corpus_valid
usage (misuse corpus):  LTNFUZZ_MISUSE=1 coverage run -p --branch --source=<ltn dir> cov_replay.py corpus_misuse

The misuse corpus must be replayed with LTNFUZZ_MISUSE=1, because the harness decodes the same bytes into a
different program in that mode.
"""
import glob
import os
import sys

os.environ["LTNFUZZ_NO_INSTRUMENT"] = "1"        # plain import of ltn: coverage.py does the measuring
corpus = sys.argv[1]
sys.argv = [sys.argv[0]]                          # keep atheris/argparse out of the way
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ltn_fuzz_combined as F  # noqa: E402

files = sorted(glob.glob(os.path.join(corpus, "*")))
ok = failed = 0
for p in files:
    try:
        F.make_runner(open(p, "rb").read()).run()
        ok += 1
    except Exception:  # noqa: BLE001  (findings and expected rejections both count as executed code)
        failed += 1
print(f"replayed {len(files)} inputs from {corpus}: {ok} ran clean, {failed} raised", file=sys.stderr)
