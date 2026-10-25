"""Test setup: put the repo root, ltn_torch and lnn/src on sys.path so pytest can import mr_unified and ltn."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (ROOT, os.path.join(ROOT, "ltn_torch"), os.path.join(ROOT, "lnn", "src")):
    if path not in sys.path:
        sys.path.insert(0, path)
