#!/usr/bin/env python3
"""Replay a corpus without libFuzzer and measure line coverage of the ltn package."""
import os, sys, glob
sys.argv = [sys.argv[0]]
import ltn_fuzz as F
corpus = os.environ["CORPUS"]
files = sorted(glob.glob(os.path.join(corpus, "*")))
for p in files:
    try:
        F.Builder(open(p, "rb").read()).run()
    except Exception:
        pass
print(f"replayed {len(files)} inputs", file=sys.stderr)
