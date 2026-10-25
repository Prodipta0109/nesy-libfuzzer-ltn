"""Findings store for the byte-driven harness (spec Task 8): one folder per (lib, rule) with input.bin and meta.json.
The root is $MR_UNIFIED_FINDINGS when set, else mr_unified/findings; it is read on every call.
"""
import json
import os
from typing import Any, Dict

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "findings")


def findings_root() -> str:
    """The folder that findings are written under."""
    return os.environ.get("MR_UNIFIED_FINDINGS", DEFAULT_ROOT)


def write_finding(lib: str, rule: str, data: bytes, meta: Dict[str, Any]) -> str:
    """Write input.bin and meta.json under <root>/<lib>/<rule>/ and return that folder."""
    folder = os.path.join(findings_root(), lib, rule)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "input.bin"), "wb") as handle:
        handle.write(data)
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    return folder
