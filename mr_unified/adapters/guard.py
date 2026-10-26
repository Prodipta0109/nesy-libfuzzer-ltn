"""Adapter guard shared by the harness and the mutation study (spec Tasks 8 and 9): a library exception becomes
AdapterCrash, so that a crash is reported as a violation and not as a harness error. Unsupported passes through.
"""
from typing import Dict

from mr_unified.adapters.base import Adapter, Unsupported
from mr_unified.formula import Env, Node


class AdapterCrash(Exception):
    """An exception other than Unsupported raised by an adapter's evaluate: a library crash, not a harness error."""

    def __init__(self, original: Exception) -> None:
        super().__init__(original)
        self.original = original


class GuardedAdapter:
    """Thin proxy: forwards every call to the adapter; evaluate re-raises any non-Unsupported exception as AdapterCrash.
    Only exceptions raised inside evaluate are library crashes, so harness exceptions elsewhere still propagate.
    oracle.rewrite_gap and oracle._compare also convert exceptions (to an infinite gap), so rewrite and env-rule crashes
    reach the harness as margin-inf violations and are labelled `<rule>:crash` by harness._crash_mark.
    """

    def __init__(self, adapter: Adapter) -> None:
        self._adapter = adapter
        self.name = adapter.name
        self.dtype = adapter.dtype

    def evaluate(self, node: Node, env: Env) -> float:
        """The adapter's value; an exception other than Unsupported becomes AdapterCrash(original)."""
        try:
            return self._adapter.evaluate(node, env)
        except Unsupported:
            raise
        except Exception as err:
            raise AdapterCrash(err) from err

    def declared_quantifiers(self) -> Dict[str, str]:
        """The wrapped adapter's declared quantifiers."""
        return self._adapter.declared_quantifiers()

    def dtype_for(self, node: Node) -> str:
        """The wrapped adapter's dtype for `node`."""
        return self._adapter.dtype_for(node)
