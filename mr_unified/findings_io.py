"""Findings store for the byte-driven harness (spec Task 8): one folder per (lib, rule) with input.bin and meta.json.
The root is $MR_UNIFIED_FINDINGS when set, else mr_unified/findings; it is read on every call. meta.json is strict JSON.
"""
import json
import math
import os
from typing import Any, Dict

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "findings")


def findings_root() -> str:
    """The folder that findings are written under."""
    return os.environ.get("MR_UNIFIED_FINDINGS", DEFAULT_ROOT)


def strict_value(value: Any) -> Any:
    """Copy of `value` with every float inf, -inf and nan replaced by the string "inf", "-inf" or "nan"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: strict_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(item) for item in value]
    return value


def write_finding(lib: str, rule: str, data: bytes, meta: Dict[str, Any]) -> str:
    """Write input.bin and meta.json under <root>/<lib>/<rule>/ and return that folder. An empty lib or rule is refused."""
    if not lib or not rule:
        raise ValueError("finding needs a lib and a rule label, got %r and %r" % (lib, rule))
    folder = os.path.join(findings_root(), lib, rule)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "input.bin"), "wb") as handle:
        handle.write(data)
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(strict_value(meta), handle, indent=2, sort_keys=True, allow_nan=False)
    return folder
