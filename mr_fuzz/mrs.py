"""
Metamorphic relations (MRs) for LTN's fuzzy operators.

Source: Metcalfe, "Fundamentals of Fuzzy Logics" (definition / proposition numbers in comments).

Every MR is a function  mr(x, y, z) -> margin
  x, y, z : float64 tensors with values in [0, 1]
  margin  : how badly the relation is broken (0 = holds). A violation is margin > TOL.

For an inequality "a <= b" the margin is relu(a - b). For an equality "a == b" it is |a - b|.
"""
import torch
import ltn.fuzzy_ops as fo

TOL = 1e-9

# ---- the operators under test. stable=False: the stable variants nudge values on purpose,
# ---- so they are not expected to obey exact algebraic laws.
NOT = fo.NotStandard()
AND = {"min": fo.AndMin(), "prod": fo.AndProd(stable=False), "luk": fo.AndLuk()}
OR = {"max": fo.OrMax(), "probsum": fo.OrProbSum(stable=False), "luk": fo.OrLuk()}
IMP = {"kd": fo.ImpliesKleeneDienes(), "godel": fo.ImpliesGodel(),
       "reich": fo.ImpliesReichenbach(stable=False), "goguen": fo.ImpliesGoguen(stable=False),
       "luk": fo.ImpliesLuk()}

# t-norm and its residuum (Def 24): these two form an adjoint pair
RESIDUATED = [("min", "godel"), ("prod", "goguen"), ("luk", "luk")]
# t-norm and its dual t-conorm (Prop 3)
DUAL = [("min", "max"), ("prod", "probsum"), ("luk", "luk")]


def err(a, b):
    """margin of the equality a == b"""
    return (a - b).abs().max().item()


def below(a, b):
    """margin of the inequality a <= b"""
    return torch.relu(a - b).max().item()


def worst(*margins):
    return max(margins)


# ------------------------------------------------------------ "and" (Def 9-11, 16, Lemma 1)
def and_commutative(x, y, z):
    return worst(*[err(f(x, y), f(y, x)) for f in AND.values()])


def and_associative(x, y, z):
    return worst(*[err(f(f(x, y), z), f(x, f(y, z))) for f in AND.values()])


def and_identity(x, y, z):                      # 1 is the identity of a t-norm
    one = torch.ones_like(x)
    return worst(*[err(f(x, one), x) for f in AND.values()])


def and_annihilator(x, y, z):                   # 0 annihilates a t-norm
    zero = torch.zeros_like(x)
    return worst(*[err(f(x, zero), zero) for f in AND.values()])


def and_monotone(x, y, z):                      # lo <= hi  =>  and(lo, z) <= and(hi, z)
    lo, hi = torch.minimum(x, y), torch.maximum(x, y)
    return worst(*[below(f(lo, z), f(hi, z)) for f in AND.values()])


# ------------------------------------------------------------ "or" (Def 20, Prop 3)
def or_identity(x, y, z):                       # 0 is the identity of a t-conorm
    zero = torch.zeros_like(x)
    return worst(*[err(f(x, zero), x) for f in OR.values()])


def or_commutative_associative(x, y, z):
    return worst(*[worst(err(f(x, y), f(y, x)), err(f(f(x, y), z), f(x, f(y, z))))
                   for f in OR.values()])


def de_morgan(x, y, z):                         # or(x,y) == 1 - and(1-x, 1-y)
    return worst(*[err(OR[o](x, y), NOT(AND[a](NOT(x), NOT(y)))) for a, o in DUAL])


# ------------------------------------------------------------ ordering of the operators (Prop 2)
def and_order(x, y, z):                         # luk <= prod <= min
    return worst(below(AND["luk"](x, y), AND["prod"](x, y)),
                 below(AND["prod"](x, y), AND["min"](x, y)))


def or_order(x, y, z):                          # max <= probsum <= luk
    return worst(below(OR["max"](x, y), OR["probsum"](x, y)),
                 below(OR["probsum"](x, y), OR["luk"](x, y)))


def idempotent_min_max(x, y, z):                # only min / max are idempotent (Prop 1)
    return worst(err(AND["min"](x, x), x), err(OR["max"](x, x), x))


# ------------------------------------------------------------ "implies" (Def 23, Lemma 6)
def modus_ponens(x, y, z):                      # and(x, x => y) <= y
    return worst(*[below(AND[a](x, IMP[i](x, y)), y) for a, i in RESIDUATED])


def adjunction(x, y, z):                        # and(x,y) <= z  iff  x <= (y => z)   (Def 24)
    m = 0.0
    for a, i in RESIDUATED:
        lhs = AND[a](x, y) <= z + TOL
        rhs = x <= IMP[i](y, z) + TOL
        m = max(m, (lhs != rhs).any().item() * 1.0)
    return m


def implies_boundary(x, y, z):                  # x=>1 == 1,  0=>x == 1,  1=>x == x
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    return worst(*[worst(err(f(x, one), one), err(f(zero, x), one), err(f(one, x), x))
                   for f in IMP.values()])


def implies_monotone(x, y, z):                  # decreasing in 1st arg, increasing in 2nd
    lo, hi = torch.minimum(x, y), torch.maximum(x, y)
    return worst(*[worst(below(f(hi, z), f(lo, z)), below(f(z, lo), f(z, hi)))
                   for f in IMP.values()])


def contraposition(x, y, z):                    # x=>y == not(y)=>not(x); holds for these three only
    return worst(*[err(IMP[i](x, y), IMP[i](NOT(y), NOT(x))) for i in ("luk", "kd", "reich")])


def not_involution(x, y, z):                    # strong negation: n(n(x)) == x
    return err(NOT(NOT(x)), x)


def equiv_symmetric(x, y, z):
    return worst(*[err(fo.Equiv(AND[a], IMP[i])(x, y), fo.Equiv(AND[a], IMP[i])(y, x))
                   for a, i in RESIDUATED])


# ------------------------------------------------------------ range and classical collapse
def stays_in_unit_interval(x, y, z):            # every connective maps [0,1] into [0,1]
    outs = [f(x, y) for f in (*AND.values(), *OR.values(), *IMP.values())] + [NOT(x)]
    return worst(*[worst(torch.relu(o - 1).max().item(), torch.relu(-o).max().item(),
                         float("inf") if not torch.isfinite(o).all() else 0.0) for o in outs])


def classical_collapse(x, y, z):                # on {0,1} every operator is the Boolean one
    a, b = (x > 0.5).double(), (y > 0.5).double()
    boolean = {"and": a * b, "or": torch.maximum(a, b), "imp": torch.maximum(1 - a, b)}
    return worst(*[err(f(a, b), boolean["and"]) for f in AND.values()],
                 *[err(f(a, b), boolean["or"]) for f in OR.values()],
                 *[err(f(a, b), boolean["imp"]) for f in IMP.values()])


# ------------------------------------------------------------ aggregators (the "forall")
AGG = {"min": fo.AggregMin(), "mean": fo.AggregMean(),
       "pmean": fo.AggregPMean(p=2, stable=False), "pmeanerr": fo.AggregPMeanError(p=2, stable=False)}


def agg_permutation(x, y, z):                   # order of the elements must not matter
    perm = torch.flip(x, dims=[0])
    return worst(*[err(f(x), f(perm)) for f in AGG.values()])


def agg_bounds(x, y, z):                        # min(xs) <= aggregate(xs) <= max(xs)
    return worst(*[worst(below(x.min(), f(x)), below(f(x), x.max())) for f in AGG.values()])


def agg_monotone(x, y, z):                      # raising every element cannot lower the aggregate
    hi = torch.maximum(x, y)
    return worst(*[below(f(x), f(hi)) for f in AGG.values()])


MRS = [and_commutative, and_associative, and_identity, and_annihilator, and_monotone,
       or_identity, or_commutative_associative, de_morgan,
       and_order, or_order, idempotent_min_max,
       modus_ponens, adjunction, implies_boundary, implies_monotone, contraposition,
       not_involution, equiv_symmetric,
       stays_in_unit_interval, classical_collapse,
       agg_permutation, agg_bounds, agg_monotone]


# ------------------------------------------------------------ the "guidance"
def bucket(margin):
    """
    Turn the violation margin into coverage. libFuzzer only learns from new code edges, so every
    line below is a new edge that is reached only when the margin is large enough. Inputs that
    push an MR closer to (or further into) a violation therefore look "interesting" and are kept
    in the corpus and mutated further.
    """
    if margin > 1e-15: pass
    if margin > 1e-12: pass
    if margin > 1e-9: pass
    if margin > 1e-6: pass
    if margin > 1e-3: pass
    if margin > 1e-1: pass
    if margin > 5e-1: pass
