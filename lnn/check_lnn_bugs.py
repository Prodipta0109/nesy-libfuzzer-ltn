"""Run any LNN bug found in this project as plain LNN code (IBM LNN 1.1.0, commit c88f513).

  python3 check_lnn_bugs.py            run every bug, then print a summary table
  python3 check_lnn_bugs.py list       list the bugs
  python3 check_lnn_bugs.py L1         show the code of one bug and RUN IT RAW:
                                       nothing is caught, so you see the library's own traceback
                                       or the failed assertion

Every bug is a small function written the way a bug report is: it uses LNN the documented way and states
what should happen. When the bug is present the function fails: it crashes with the exception shown, or an
assertion states the wrong value. If LNN is ever fixed, the function returns normally and the bug is
reported as NOT REPRODUCED.  A call that should return but does not is stopped after a few seconds and reported
as an assertion ("did not return").
"""
import contextlib
import inspect
import os
import signal
import sys
import textwrap
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("LNN_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")))
from lnn import *          # noqa: E402,F403


@contextlib.contextmanager
def returns_within(seconds, what):
    """Fail with an AssertionError when the block does not finish in time (the call never returns)."""
    def stop(*_):
        raise AssertionError(f"{what} did not return within {seconds} s")
    old = signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ----------------------------------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------------------------------

def bug_L1():
    """Weighted And with unequal weights.  Expected: infer() returns (it stops when nothing changes any more).
    Bug: with the lighter-weight operand known exactly, every step still changes the bounds by one float32 step,
    which is above the 1e-7 stopping threshold, so infer() without max_steps never returns."""
    A, B = Propositions("A", "B")
    f = And(A, B, activation={"weights": (1.0, 0.5), "bias": 1.0, "weights_learning": False, "bias_learning": False})
    m = Model()
    m.add_knowledge(f)
    m.add_data({B: (0.4, 0.4)})                      # B is known exactly, A is unknown
    steps, updated = m.infer(max_steps=200)          # capped, so this demo ends: it uses all 200 steps
    print("steps used:", steps, "| bounds of A after 200 steps:", A.get_data().tolist())
    assert steps < 200, f"inference did not converge: it used all {steps} steps and A is still shrinking (upper {A.get_data().flatten()[1]:.7f})"


def bug_L1b():
    """Same root cause as L1, seen through training.  Expected: Model.train() returns.
    Bug: after the first parameter update the And weights are unequal, the final inference never converges,
    and train() (which calls infer() without a step limit) never returns."""
    A, B = Propositions("A", "B")
    f = And(A, B)
    m = Model()
    m.add_knowledge(f)
    m.add_data({B: (0.4, 0.4)})
    m.add_labels({f: Fact.UNKNOWN})
    with returns_within(8, "Model.train(epochs=1)"):
        m.train(losses=[Loss.SUPERVISED], epochs=1, pbar=False)


def bug_L2():
    """Quantifier over a predicate with one known grounding.  Expected: after the first inference the bounds are
    final and a further infer() reports no new information.
    Bug: the quantifier rebuilds its neuron on every step (unary_operator.py:248), so each step 'learns' the same
    thing again and infer() without max_steps never returns."""
    x = Variable("x")
    S = Predicate("S")
    f = Exists(x, S(x))
    m = Model()
    m.add_knowledge(f)
    m.add_data({S: {"a": Fact.TRUE}})
    m.infer(max_steps=5)
    print("Exists x S(x) after inference:", f.get_data().tolist())
    steps, updated = m.infer(max_steps=5)            # nothing has changed since: no new information expected
    assert float(updated) == 0.0, f"a second inference on an unchanged model reports {float(updated)} new information in {steps} steps"


def bug_L3():
    """Choose the Goedel (or Frechet) activation, which the NeuralActivation enum offers.
    Bug: the module that implements it does not exist in the package (lnn/neural/methods has only Lukasiewicz)."""
    A, B = Propositions("A", "B")
    And(A, B, activation={"type": NeuralActivation.Godel})


def bug_L4():
    """Train with Loss.LOGICAL on a model that contains an Iff (or XOr).  Expected: a loss value.
    Bug: _logical_loss has branches for And, Or and Implies only, so true_hinge is never assigned."""
    A, B = Propositions("A", "B")
    f = Iff(A, B)
    m = Model()
    m.add_knowledge(f)
    m.add_data({A: Fact.TRUE, B: Fact.FALSE})
    m.add_labels({f: Fact.TRUE})
    m.train(losses=[Loss.LOGICAL], epochs=2, pbar=False)


def bug_L5():
    """The same formula built twice.  Expected: the answer does not depend on the order in which formulas are added.
    Bug: a model identifies a formula by its printed name; Iff(P,Q) creates its own Implies(P,Q) inside, and the
    later explicit Implies(P,Q) object is then never updated.  Adding the formulas in the other order is correct."""
    def answer(order):
        P, Q, R = Propositions("P", "Q", "R")
        k = Iff(P, Q)
        g = Or(Implies(P, Q), R)
        m = Model()
        m.add_knowledge(*([k, g] if order == "Iff first" else [g, k]))
        m.add_data({P: Fact.TRUE, Q: Fact.FALSE, R: Fact.FALSE})
        m.infer(max_steps=20)
        return g.get_data().tolist()
    a, b = answer("Iff first"), answer("Or first")
    print("Or(P->Q, R) with P=true, Q=false, R=false:   Iff added first ->", a, "|  Or added first ->", b)
    assert a == b, f"result depends on the order of add_knowledge: {a} versus {b}"


def bug_L6():
    """Quantified sentence over consistent classical data.  Forall x S(x) with S(a) true and S(b) false is simply a
    false sentence.  Expected: Forall is false and there is no contradiction (same for Exists x S(x) with one true
    and one false grounding, which is true).
    Bug: downward inference from the quantifier's result is applied to every grounding, so S(a) (or S(b)) is
    pushed to the impossible bounds (1, 0) and the model reports a contradiction."""
    for quantifier in (Forall, Exists):
        x = Variable("x")
        S = Predicate("S")
        f = quantifier(x, S(x))
        m = Model()
        m.add_knowledge(f)
        m.add_data({S: {"a": Fact.TRUE, "b": Fact.FALSE}})
        m.infer(max_steps=5)
        print(quantifier.__name__, "x S(x), S(a)=true, S(b)=false ->", f.get_data().tolist(),
              "| groundings of S:", S.get_data().tolist(), "| contradiction:", m.has_contradiction())
        assert not m.has_contradiction(), f"{quantifier.__name__}: consistent classical data reported as a contradiction"


def bug_L7():
    """The sentence 'every S has an R-successor': Forall x (S(x) -> Exists y R(x,y)).  Expected: a model.
    Bug: a quantifier applied directly to a predicate that has another free variable fails while the formula is
    built, with a raw ValueError (the library's own tests for this case are commented out)."""
    x, y = Variables("x", "y")
    R = Predicate("R", arity=2)
    S = Predicate("S")
    Forall(x, Implies(S(x), Exists(y, R(x, y))))


def bug_L8():
    """A predicate with a repeated variable (reflexive R(x,x)) combined with another formula.  Expected: inference.
    Bug: ValueError 'Length mismatch' from pandas, because the grounding table of R(x,x) has one column for two
    argument slots.  R(x,x) alone works."""
    x = Variable("x")
    R = Predicate("R", arity=2)
    S = Predicate("S")
    f = Or(S(x), R(x, x))
    m = Model()
    m.add_knowledge(f)
    m.add_data({S: {"a": Fact.TRUE}, R: {("a", "a"): Fact.FALSE}})
    m.infer(max_steps=5)


# id -> (function, expected exception type, severity, found by, where, expected, what is wrong)
FUZZ = "libFuzzer (Atheris)"
BUGS = {
    "L1": (bug_L1, AssertionError, "high", f"{FUZZ}, family t_weights", "model.py:507 (_infer stop test) with the weighted And/Or/Implies update",
           "infer() returns", "inference never converges when operand weights differ"),
    "L1b": (bug_L1b, AssertionError, "high", f"{FUZZ}, family t_learn (timeouts)", "model.py:619/635 (train calls infer without max_steps)",
            "train() returns", "train() hangs once learned weights differ (same root cause as L1)"),
    "L2": (bug_L2, AssertionError, "high", f"{FUZZ}, family t_conv", "symbolic/logic/unary_operator.py:248 (_fully_quantified_upward)",
           "second inference reports 0", "inference over a quantifier never converges"),
    "L3": (bug_L3, ModuleNotFoundError, "medium", f"{FUZZ}, family t_upward", "symbolic/logic/neural_activation.py (module import)",
           "a Goedel neuron", "NeuralActivation.Godel / Frechet are offered but not implemented"),
    "L4": (bug_L4, UnboundLocalError, "medium", f"{FUZZ}, family t_learn", "symbolic/logic/connective_neuron.py:170 (_logical_loss)",
           "a loss value", "Loss.LOGICAL crashes for Iff and XOr"),
    "L5": (bug_L5, AssertionError, "medium", f"{FUZZ}, family t_upward / t_classical", "model.py:275 (formulas identified by printed name)",
           "same answer in either order", "a formula built twice is half-orphaned; result depends on add_knowledge order"),
    "L6": (bug_L6, AssertionError, "high", f"{FUZZ} round 3, family t_fol2", "symbolic/logic/unary_operator.py:267-272 (_fully_quantified_downward)",
           "no contradiction", "consistent data on a quantified sentence is reported as a contradiction"),
    "L7": (bug_L7, ValueError, "medium", f"{FUZZ} round 3, family t_fol_shapes", "symbolic/logic/unary_operator.py:83 (_Quantifier.__init__)",
           "a model", "quantifier over a predicate with another free variable crashes while building the formula"),
    "L8": (bug_L8, ValueError, "medium", f"{FUZZ} round 3, family t_fol_shapes", "pandas set_axis in the grounding table of R(x,x)",
           "inference", "a predicate with a repeated variable cannot be combined with another formula"),
}


def show_code(bug_id):
    fn = BUGS[bug_id][0]
    print(f"{bug_id}: {BUGS[bug_id][6]}\n")
    print(textwrap.dedent(inspect.getsource(fn)))
    print("-" * 100)


def main(argv):
    if len(argv) > 1 and argv[1] == "list":
        for bug_id, b in BUGS.items():
            print(f"  {bug_id:4} {b[2]:7} {b[6]}")
        return 0
    if len(argv) > 1:
        bug_id = argv[1]
        if bug_id not in BUGS:
            print("unknown bug id; use 'list'")
            return 2
        show_code(bug_id)
        BUGS[bug_id][0]()
        print("\nNO FAILURE: LNN behaved as expected, so this bug is NOT present in this version.")
        return 0

    results = {}
    for bug_id, (fn, exc, *_rest) in BUGS.items():
        print(f"\n{'#' * 100}\n# {bug_id}: {BUGS[bug_id][6]}\n{'#' * 100}")
        print(textwrap.indent(textwrap.dedent(inspect.getsource(fn)), "  | "))
        try:
            fn()
            results[bug_id] = ("NOT REPRODUCED", "ran without failing")
            print("  -> ran without failing: NOT REPRODUCED")
        except Exception as e:
            kind = type(e).__name__
            ok = isinstance(e, exc)
            results[bug_id] = ("REPRODUCED" if ok else "DIFFERENT FAILURE", f"{kind}: {str(e)[:110]}")
            last = traceback.extract_tb(e.__traceback__)[-1]
            print(f"  -> FAILED as the bug describes: {kind}: {str(e)[:150]}")
            print(f"     raised at {last.filename.split('/')[-1]}:{last.lineno}  ({last.name})")
            if not ok:
                print(f"     (expected a {exc.__name__})")

    print(f"\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    print(f"  {'ID':4} {'status':16} {'severity':8} what happens")
    for bug_id, (status, shown) in results.items():
        b = BUGS[bug_id]
        print(f"  {bug_id:4} {status:16} {b[2]:8} {b[6]}")
        print(f"  {'':4} {'':16} {'':8} found by: {b[3]};  where: {b[4]};  expected: {b[5]}")
    n = sum(1 for s, _ in results.values() if s == "REPRODUCED")
    print(f"\n  {n} of {len(BUGS)} bug demonstrations fail exactly as described")
    return 0 if n == len(BUGS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
