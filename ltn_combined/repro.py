#!/usr/bin/env python3
"""Replay one fuzzer input: print the LTN program it decodes to and the outcome."""
import os
import sys
import traceback

os.environ.setdefault("LTNFUZZ_COLLECT", "0")
sys.argv = [sys.argv[0]] + sys.argv[1:]
import ltn_fuzz_combined as F  # noqa: E402

for path in sys.argv[1:]:
    data = open(path, "rb").read()
    b = F.make_runner(data)
    print(f"=== {path} ({len(data)} bytes)")
    try:
        b.run()
        outcome = "OK"
    except Exception as e:  # noqa: BLE001
        outcome = ("EXPECTED REJECTION: " if F.is_expected(e) else "FINDING: ") + F.signature(e)
        tb = traceback.format_exc()
    print("\n".join(b.trace))
    print("--> " + outcome)
    if outcome != "OK" and not outcome.startswith("EXPECTED"):
        print(tb)
