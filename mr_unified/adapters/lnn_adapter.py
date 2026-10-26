"""LNN adapter for the unified metamorphic oracle (spec Task 6): translates a Node into LNN formulas and evaluates it.
Strong connectives only (weak and/or raise Unsupported). Quantifier finding: default Forall/Exists are open world
(an unseen grounding: forall lower 0, exists upper 1); fully_grounded=True is closed world, the strong fold, and is used here.
Point data P={a:.9,b:.8,c:.7}: default Forall (0.0, 0.4); fully_grounded Forall (0.4, 0.4) == fold. See test_lnn_quantifier_semantics.
"""
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Tuple

# LNN logs and warns while it runs; silence them as lnn/lnn_fuzz.py does, so test output stays pristine.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from lnn import And, Direction, Exists, Forall, Implies, Model, Not, Or, Predicate, Propositions, Variable
logging.disable(logging.CRITICAL)

from mr_unified.adapters.base import Unsupported
from mr_unified.formula import Env, Node

MAX_STEPS = 60
POINT_TOLERANCE = 1e-6
STRONG = {"and_s": And, "or_s": Or, "imp": Implies, "not": Not}
WEAK_OPS = ("and_w", "or_w")
QUANT_CLASSES = {"forall": Forall, "exists": Exists}


@dataclass
class _Build:
    """The LNN objects made for one Node: formulas keyed by printed name, predicates, and the data for the Model."""

    formulas: Dict[str, object] = field(default_factory=dict)
    preds: Dict[str, object] = field(default_factory=dict)
    data: Dict[object, object] = field(default_factory=dict)


class LNNAdapter:
    """Evaluates a Node with LNN: strong connectives and closed-world forall/exists over unary predicate atoms."""

    name = "lnn"
    dtype = "float32"

    def declared_quantifiers(self) -> Dict[str, str]:
        """forall is the strong fold of and, exists is the strong fold of or (closed world)."""
        return {"forall": "fold", "exists": "fold"}

    def dtype_for(self, node: Node) -> str:
        """LNN computes in float32 for every formula."""
        return "float32"

    def evaluate(self, node: Node, env: Env) -> float:
        """Build the LNN model for `node` over `env`, infer upward, and return the root's truth value."""
        build = _Build()
        root = _translate(node, env, build)
        model = Model()
        model.add_knowledge(root, *build.formulas.values())
        model.add_data(build.data)
        model.infer(Direction.UPWARD, max_steps=MAX_STEPS)
        lower, upper = _bounds_in(model, root)
        if abs(lower - upper) > POINT_TOLERANCE:
            raise AssertionError("root is not a point: lower %r upper %r for %r" % (lower, upper, node))
        return lower


def _translate(node: Node, env: Env, build: _Build) -> object:
    """Dispatch on the node kind to its LNN formula."""
    if node.op == "atom":
        return _atom(node, env, build)
    if node.op == "const":
        return _proposition(build, "const_%s" % node.value, node.value)
    if node.op in QUANT_CLASSES:
        return _quantify(node, env, build)
    if node.op in WEAK_OPS:
        raise Unsupported("LNN has no weak connective for %s" % node.op)
    return _connect(node, env, build)


def _atom(node: Node, env: Env, build: _Build) -> object:
    """A propositional atom is a proposition with point bounds; a predicate atom outside a quantifier is free."""
    if node.args:
        return _ground_atom(node, env, build)
    return _proposition(build, node.name, env.atoms[node.name])


def _ground_atom(node: Node, env: Env, build: _Build) -> object:
    """A predicate atom over domain objects only, e.g. P(o0): a proposition with the value Env.preds[P][(o0,)]."""
    unbound = [arg for arg in node.args if arg not in env.domain]
    if unbound:
        raise ValueError("formula has free variables: %s" % (unbound,))
    key = tuple(node.args)
    return _proposition(build, "%s(%s)" % (node.name, ",".join(key)), env.preds[node.name][key])


def _proposition(build: _Build, name: str, value: float) -> object:
    """Laws: a proposition p has point bounds (v, v) with v its truth value; one object per printed name."""
    if name not in build.formulas:
        leaf = Propositions(name)
        build.formulas[name] = leaf
        build.data[leaf] = (value, value)
    return build.formulas[name]


def _connect(node: Node, env: Env, build: _Build) -> object:
    """Laws: not a == 1 - a; a -> b == min(1 - a + b, 1); a (x) b == max(a + b - 1, 0); a (+) b == min(a + b, 1)."""
    if node.op not in STRONG:
        raise ValueError("unknown op: %s" % node.op)
    kids = [_translate(kid, env, build) for kid in node.kids]
    return _remember(build, STRONG[node.op](*kids))


def _quantify(node: Node, env: Env, build: _Build) -> object:
    """Laws: forall x. P(x) == fold of strong and over the domain; exists x. P(x) == fold of strong or; empty domain: 1 / 0."""
    body = node.kids[0]
    if body.op != "atom" or body.args != (node.name,):
        raise Unsupported("LNN quantifier body must be a unary predicate atom over %s" % node.name)
    if not env.domain:
        return _proposition(build, "const_%s" % _identity(node.op), _identity(node.op))
    variable = Variable(node.name)
    predicate = _predicate(body.name, env, build)
    return _remember(build, QUANT_CLASSES[node.op](variable, predicate(variable), fully_grounded=True))


def _identity(op: str) -> float:
    """Laws: forall over an empty set is 1.0 (identity of min and of the strong fold); exists is 0.0."""
    return 1.0 if op == "forall" else 0.0


def _predicate(name: str, env: Env, build: _Build) -> object:
    """A unary predicate P with one grounding per domain object, each with point bounds (v, v) from Env.preds."""
    if name not in build.preds:
        predicate = Predicate(name, arity=1)
        build.preds[name] = predicate
        build.data[predicate] = {obj: _point(env.preds[name][(obj,)]) for obj in env.domain}
    return build.preds[name]


def _point(value: float) -> Tuple[float, float]:
    """The point bounds (v, v) of a truth value v."""
    return value, value


def _remember(build: _Build, formula: object) -> object:
    """Keep the first formula with each printed name, so every distinct subformula is built once."""
    if formula.name not in build.formulas:
        build.formulas[formula.name] = formula
    return build.formulas[formula.name]


def _bounds_in(model: Model, formula: object) -> Tuple[float, float]:
    """The bounds the model holds for `formula`, found by printed name (LNN keys nodes by name)."""
    stored = next((n for n in model.nodes.values() if n.name == formula.name), None)
    if stored is None:
        raise ValueError("LNN model holds no node named %r" % formula.name)
    bounds = stored.get_data().detach().flatten().tolist()
    return bounds[0], bounds[1]
