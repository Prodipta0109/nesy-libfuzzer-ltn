"""Replay one fuzz input (spec Task 8): decode it exactly as the harness does, run the same checks, print what it finds.
Run: python -m mr_unified.replay <input.bin> --lib ltn|lnn|ref|all  (use the --lib recorded in the finding's meta.json).
"""
import argparse
import sys
from typing import List, Optional

from mr_unified import harness
from mr_unified.oracle import Violation


def replay(path: str, lib: str) -> List[Violation]:
    """The violations that the input file at `path` produces on the adapters selected by `lib`."""
    with open(path, "rb") as handle:
        data = handle.read()
    return harness.run_input(data, harness.make_adapters(lib))


def main(argv: Optional[List[str]] = None) -> int:
    """Print each violation the input reproduces, or "no violation". Returns 1 if there is one, else 0."""
    parser = argparse.ArgumentParser(description="Replay one harness input.")
    parser.add_argument("input", help="path to input.bin")
    parser.add_argument("--lib", default="ltn", help="ltn, lnn, ref or all, as recorded in meta.json")
    args = parser.parse_args(argv)
    violations = replay(args.input, args.lib)
    if not violations:
        print("no violation")
        return 0
    for violation in violations:
        print("[VIOLATION] " + harness.describe(violation))
    return 1


if __name__ == "__main__":
    sys.exit(main())
