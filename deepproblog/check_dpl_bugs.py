"""Run any DeepProbLog bug found in this project as plain DeepProbLog code (DeepProbLog 2.1.0, ProbLog 2.3.0).

  python3 check_dpl_bugs.py            run every bug, then print a summary table
  python3 check_dpl_bugs.py list       list the bugs
  python3 check_dpl_bugs.py D7         show the code of one bug and RUN IT RAW:
                                       nothing is caught, so you see the library's own traceback
                                       or the failed assertion

Every bug is a small function written the way a bug report is: it uses DeepProbLog the documented way and states
what should happen. When the bug is present the function fails: it crashes with the exception shown, or an
assertion states the wrong value. If DeepProbLog is ever fixed, the function returns normally and the bug is
reported as NOT REPRODUCED.
"""
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import traceback
import types
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from problog.logic import Term, Constant
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.query import Query
from deepproblog.dataset import Dataset, DataLoader, QueryDataset, NumericIncorrectOutputMutator
from deepproblog.optimizer import SGD
from deepproblog.semiring.graph_semiring import GraphSemiring
from deepproblog.utils.logger import Logger
from deepproblog.utils.confusion_matrix import ConfusionMatrix
from deepproblog.utils.stop_condition import Threshold


def model(program, cache=False):
    """A DeepProbLog model for a program given as text, with the standard exact engine."""
    m = Model(program, [], load=False)
    m.set_engine(ExactEngine(m), cache=cache)
    return m


class Numbers(Dataset):
    """A dataset of n queries q(0), q(1), ..."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def to_query(self, i):
        return Query(Term("q", Constant(i)))


# ----------------------------------------------------------------------------------------------------------------
# Learning the probabilities and the networks
# ----------------------------------------------------------------------------------------------------------------

def bug_D5():
    """Annotated disjunction with one learnable and one fixed member.  Expected: total probability stays <= 1.
    Bug: the optimizer step rescales the learnable member to 1.0 and ignores the fixed member (even with lr=0)."""
    m = model("t(0.5)::a; 0.4::b.\nq :- a.\n")        # a is learnable, b is fixed at 0.4: total mass 0.9
    m.optimizer = SGD(m, 0.0)                          # learning rate ZERO: nothing should change
    m.solve([Query(Term("q"))])
    m.optimizer.step()
    mass = m.parameters[0] + 0.4
    print("learnable member a:", m.parameters[0], "| fixed member b: 0.4 | total mass:", mass)
    assert mass <= 1.0 + 1e-9, f"annotated disjunction now has total probability {mass} > 1 after a step with lr=0"


def bug_D6():
    """Learning two members of an annotated disjunction downwards.  Expected: training continues.
    Bug: both parameters reach 0, the group is divided by its sum 0, ZeroDivisionError."""
    m = model("t(0.1)::a; t(0.1)::b.\nq :- a.\nq :- b.\n")
    m.optimizer = SGD(m, 10.0)
    q = Term("q")
    r = m.solve([Query(q)])[0]
    GraphSemiring.cross_entropy(r, 0.0, 1.0, q=q)      # target probability 0: gradient pushes a and b down
    m.optimizer.step()
    print("parameters after the step:", m.parameters)


def bug_D7():
    """Neural predicate with k=2 (top-k) in training mode, network confident about one class.
    Expected: a probability.  Bug: numpy ValueError 'Fewer non-zero entries in p than size'."""
    class Confident(nn.Module):
        def forward(self, x):
            return torch.softmax(torch.tensor([200.0, 0.0]), -1)       # = [1.0, 0.0] exactly in float32

    net = Network(Confident(), "net", k=2)
    m = Model("nn(net,[X],Y,[0,1]) :: d(X,Y).\ns(X,Y) :- d(X,Y).\n", [net], load=False)
    m.set_engine(ExactEngine(m))                      # training mode is the default
    m.add_tensor_source("src", {(Constant(0),): torch.tensor([1.0, 0.0])})
    x = Term("tensor", Term("src", Constant(0)))
    print(m.solve([Query(Term("s", x, Constant(0)))]))


# ----------------------------------------------------------------------------------------------------------------
# Files and state
# ----------------------------------------------------------------------------------------------------------------

def bug_D4():
    """Saving to a file name without a directory.  Expected: file 'state.zip' in the current directory.
    Bug: FileNotFoundError: ''.  Same for Logger.write_to_file('log') and train_model(snapshot_name='snap')."""
    m = model("0.3::a.\n")
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            m.save_state("./state.zip")               # works
            print("./state.zip: ok")
            m.save_state("state.zip")                 # same file, no './'
            print("state.zip: ok")
        finally:
            os.chdir(old)


def bug_D13():
    """Persistent circuit cache (Solver cache_root).  Expected: second run finds the circuit on disk.
    Bug: the file name is str(hash(term)); Python salts string hashes per process, so every run writes a new file
    and never reads the old one."""
    code = textwrap.dedent("""
        import warnings, io, contextlib, os, sys
        warnings.filterwarnings("ignore")
        from problog.logic import Term
        from deepproblog.model import Model
        from deepproblog.engines import ExactEngine
        from deepproblog.query import Query
        m = Model("0.3::a.\\nq :- a.\\n", [], load=False)
        with contextlib.redirect_stdout(io.StringIO()):
            m.set_engine(ExactEngine(m), cache=False, cache_root=sys.argv[1])
            m.solve([Query(Term("q"))])
    """)
    with tempfile.TemporaryDirectory() as td:
        for run in (1, 2):                            # two separate processes, same cache directory
            subprocess.run([sys.executable, "-c", code, td], check=True, capture_output=True)
            print(f"after run {run}: {len(os.listdir(td))} file(s) in the cache directory")
        n = len(os.listdir(td))
    assert n == 1, f"the same query was compiled and stored {n} times: the persistent cache is never hit"


# ----------------------------------------------------------------------------------------------------------------
# Queries, datasets and utilities
# ----------------------------------------------------------------------------------------------------------------

def bug_D1():
    """Using a Query as a set member / dict key.  Expected: a hash.  Bug: TypeError: unhashable type: 'dict'."""
    q = Query(Term("f", Constant(1)))
    print(hash(q))


def bug_D2():
    """DataLoader with batch_size=0.  Expected: an error, or no batches.  Bug: it yields empty batches forever."""
    loader = DataLoader(Numbers(3), 0, shuffle=False)
    for count, batch in enumerate(loader, 1):
        assert count <= 100, f"the loader is still yielding after {count} batches (the last one is {batch})"


def bug_D3():
    """Model.ground_dataset with the default solver settings.  Expected: timing dictionary.
    Bug: AttributeError 'ArithmeticCircuit' object has no attribute 'from_cache' (works only with cache=True)."""
    m = model("0.3::f(1).\n")
    print(m.ground_dataset(QueryDataset([Query(Term("f", Constant(1)))])))


def bug_D8():
    """Threshold(attribute, maximum=1.0, duration=3) when only one value (already above the limit) is logged.
    Expected: False, not enough history to stop yet.  Bug: IndexError: list index out of range."""
    logger = Logger()
    logger.log("accuracy", 1, 2.0)                    # one logged value, already >= the limit 1.0
    Threshold("accuracy", 1.0, duration=3).is_stop(types.SimpleNamespace(logger=logger))


def bug_D9():
    """NumericIncorrectOutputMutator whose domain contains only the current value.  Expected: error or same value.
    Bug: infinite loop (the demo gives up after 2 seconds)."""
    mutator = NumericIncorrectOutputMutator([3], seed=1)
    done = threading.Event()

    def run():
        mutator(0, Query(Term("f", Constant(3))))
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(2.0), "the mutator is still looping after 2 seconds: it searches forever for a value different from 3"


def bug_D10():
    """ConfusionMatrix built from the caller's list of classes.  Expected: the caller's list is untouched.
    Bug: the matrix appends new classes to that very list."""
    classes = ["a", "b"]
    ConfusionMatrix(classes).add_item("c", "a")
    print("caller's list:", classes)
    assert classes == ["a", "b"], f"the caller's list was changed to {classes}"


def bug_D14():
    """Dataset.subset(i, j) with i beyond the end of the dataset.  Expected: empty subset, like a Python slice.
    Bug: ValueError: __len__() should return >= 0."""
    sub = Numbers(7).subset(8, 12)                    # list(range(7))[8:12] == []
    print("subset size:", len(sub))


# ----------------------------------------------------------------------------------------------------------------
# Programs and probabilities
# ----------------------------------------------------------------------------------------------------------------

def bug_D11():
    """A user predicate named like one of the 13 built-in tensor predicates (max/2, stack/2, cat/2, dot/3, ...).
    Expected: {max(1,2): 0.5}.  Bug: the call goes to the built-in torch function and crashes inside torch."""
    m = model("0.5::max(1,2).\n")                     # an ordinary probabilistic fact
    print(m.solve([Query(Term("max", Constant(1), Constant(2)))]))


def bug_D12():
    """Probabilities below 5e-16 in a program or in a query file.  Expected: kept.  Bug: ProbLog rounds every
    float to 15 decimal places, so 1e-20 becomes 0.0 and 8.6e-16 becomes 1e-15."""
    m = model("1e-20::a.\n")
    got = list(m.solve([Query(Term("a"))])[0].result.values())
    print("P(a) for '1e-20::a.' ->", got)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "q.txt")
        with open(path, "w") as f:
            QueryDataset([Query(Term("f", Constant(3)), p=8.628622989725887e-16)]).write_to_file(f)
        back = QueryDataset(path).queries[0].p
    print("query probability 8.628622989725887e-16 written to a file and read back ->", back)
    assert got and got[0] == 1e-20, f"P(a)={got}; the program said 1e-20"


# ----------------------------------------------------------------------------------------------------------------
# id -> (function, expected exception type, severity, found by, where, expected, what is wrong)
# ----------------------------------------------------------------------------------------------------------------
FUZZ = "libFuzzer (Atheris)"
BUGS = {
    "D7": (bug_D7, ValueError, "high", f"{FUZZ}, family t_neural", "exact_engine.py:74 (get_predicate)",
           "a probability", "top-k network crashes in training mode as soon as its softmax has exact zeros"),
    "D5": (bug_D5, AssertionError, "medium", f"{FUZZ}, family t_learn", "optimizer.py:92-96 (SGD.step)",
           "mass of the disjunction stays <= 1", "partly learnable disjunction: learnable member forced to 1.0"),
    "D6": (bug_D6, ZeroDivisionError, "medium", f"{FUZZ}, family t_learn", "optimizer.py:95 (SGD.step)",
           "training continues", "ZeroDivisionError when all members of a learnable disjunction reach 0"),
    "D4": (bug_D4, FileNotFoundError, "medium", f"{FUZZ}, families t_api / t_data / t_train", "utils/__init__.py:28 (check_path)",
           "file written to the current directory", "save_state / write_to_file / snapshot_name fail for a bare file name"),
    "D13": (bug_D13, AssertionError, "medium", "code reading + targeted check (needs two processes, outside the fuzzer)",
            "utils/cache.py:73 (get_from_file)", "second run reuses the stored circuit", "persistent cache is never hit; one new file per run"),
    "D3": (bug_D3, AttributeError, "medium", f"{FUZZ}, family t_data", "model.py:127 (ground_dataset)",
           "timing dictionary", "ground_dataset crashes unless the solver was created with cache=True"),
    "D11": (bug_D11, TypeError, "medium", f"{FUZZ}, family t_names", "engines/builtins.py (register_tensor_predicates)",
            "{max(1,2): 0.5}", "user predicates named like a built-in tensor predicate (max, stack, cat, dot, ...) are captured by it and crash in torch"),
    "D1": (bug_D1, TypeError, "low", f"{FUZZ}, family t_api", "query.py:94 (Query.__hash__)",
           "an integer hash", "Query.__hash__ always raises: it hashes a dict"),
    "D2": (bug_D2, AssertionError, "low", f"{FUZZ}, family t_api", "dataset.py:137 (DataLoader.__next__)",
           "an error or zero batches", "DataLoader(batch_size=0) iterates forever"),
    "D8": (bug_D8, IndexError, "low", f"{FUZZ}, family t_data", "utils/stop_condition.py:132 (Threshold.is_stop)",
           "keep training", "Threshold(duration=n) crashes until n values are logged"),
    "D9": (bug_D9, AssertionError, "low", f"{FUZZ}, family t_data", "dataset.py:326 (NumericIncorrectOutputMutator)",
           "an error or the same value", "mutator loops forever if the domain holds only the current value"),
    "D10": (bug_D10, AssertionError, "low", f"{FUZZ}, family t_data", "utils/confusion_matrix.py (ConfusionMatrix.__init__)",
            "caller's list unchanged", "ConfusionMatrix modifies the list of classes it was given"),
    "D14": (bug_D14, ValueError, "low", f"{FUZZ}, family t_api (first run)", "dataset.py (Subset.__len__)",
            "empty subset", "subset(i, j) with i beyond the end gives a negative length"),
    "D12": (bug_D12, AssertionError, "low", f"{FUZZ}, family t_data (round trip); cause is in ProbLog", "problog/logic.py:901 (Constant)",
            "1e-20 stays 1e-20", "probabilities below 5e-16 are rounded to 0 / 1e-15 (15 decimal places)"),
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
    if len(argv) > 1:                      # run one bug raw: the traceback or the failed assertion is the evidence
        bug_id = argv[1]
        if bug_id not in BUGS:
            print("unknown bug id; use 'list'")
            return 2
        show_code(bug_id)
        BUGS[bug_id][0]()
        print("\nNO FAILURE: DeepProbLog behaved as expected, so this bug is NOT present in this version.")
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
