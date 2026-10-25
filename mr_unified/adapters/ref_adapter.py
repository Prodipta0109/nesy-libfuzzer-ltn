"""Reference adapter for the unified metamorphic oracle (spec Task 7): ref_eval in float64 with min/max quantifiers.
Used as the trusted side of every check and as the baseline in tests of later tasks.
"""
from typing import Dict

from mr_unified.formula import Env, Node, ref_eval


class RefAdapter:
    """Evaluates a Node with the float64 reference semantics, forall as min and exists as max."""

    name = "ref"
    dtype = "float64"

    def evaluate(self, node: Node, env: Env) -> float:
        """ref_eval(node, env) with the min/max quantifiers."""
        return ref_eval(node, env, "minmax")

    def declared_quantifiers(self) -> Dict[str, str]:
        """forall is min and exists is max."""
        return {"forall": "min", "exists": "max"}

    def dtype_for(self, node: Node) -> str:
        """Always float64: the reference computes in double precision."""
        return "float64"
