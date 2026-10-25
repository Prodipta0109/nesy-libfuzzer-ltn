"""Adapter protocol and the Unsupported error for the unified metamorphic oracle (spec Task 5).
Every library adapter (LTN, LNN, NeuPSL) turns a Node and an Env into one float, matching ref_eval.
"""
from typing import Dict, Protocol

from mr_unified.formula import Env, Node


class Unsupported(Exception):
    """Raised by an adapter for a formula it cannot express; this is not a truth-value mismatch."""


class Adapter(Protocol):
    """A library-specific evaluator of formulas over an Env."""

    name: str
    dtype: str

    def evaluate(self, node: Node, env: Env) -> float:
        """Return the truth value of `node` in `env` as computed by the library."""

    def declared_quantifiers(self) -> Dict[str, str]:
        """Map "forall" and "exists" to the aggregator name the library uses ("min", "max", "fold")."""
