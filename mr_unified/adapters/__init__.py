"""Library adapters for the unified metamorphic oracle (spec Task 5).
Exports the Adapter protocol and the Unsupported error; LTNAdapter lives in adapters/ltn_adapter.py.
"""
from mr_unified.adapters.base import Adapter, Unsupported

__all__ = ["Adapter", "Unsupported"]
