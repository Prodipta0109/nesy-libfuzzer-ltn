# LTN in 8 small programs. In Colab, put each "# %%" block in its own cell.
# Cell 0 (setup) is below; everything after it needs only `import ltn`.

# %% Cell 0: setup (Colab only; locally ltn_torch is already cloned)
# !git clone -q https://github.com/tommasocarraro/LTNtorch.git ltn_torch
# !git -C ltn_torch checkout -q d1bd981
# import sys; sys.path.insert(0, "ltn_torch")

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "ltn_torch")
import torch
import ltn
fo = ltn.fuzzy_ops


# %% Program 1: a Variable is a table of things (rows = things, columns = facts)
animals = ltn.Variable("animals", torch.tensor([[0.9, 0.1],
                                                [0.2, 0.8],
                                                [0.7, 0.3]]))
print("value:\n", animals.value)
print("free_vars:", animals.free_vars)       # the variables a formula depends on
print("rows:", animals.value.shape[0], " columns:", animals.value.shape[1])


# %% Program 2: a Predicate gives each row a score between 0 and 1
IsCat = ltn.Predicate(func=lambda a: torch.sigmoid(a[:, 0] * 4 - 2))    # column 0 = "catness"
IsSmall = ltn.Predicate(func=lambda a: torch.sigmoid(a[:, 1] * 4 - 2))  # column 1 = "smallness"

cat = IsCat(animals)
small = IsSmall(animals)
print("IsCat(animals)   :", cat.value)       # one score per animal
print("IsSmall(animals) :", small.value)
print("free_vars        :", cat.free_vars)   # still depends on 'animals'


# %% Program 3: a Connective joins scores (AND, OR, NOT, IF-THEN)
And = ltn.Connective(fo.AndProd())
Or = ltn.Connective(fo.OrProbSum())
Not = ltn.Connective(fo.NotStandard())
Implies = ltn.Connective(fo.ImpliesReichenbach())

print("cat AND small   :", And(cat, small).value)
print("cat OR small    :", Or(cat, small).value)
print("NOT cat         :", Not(cat).value)
print("cat IMPLIES small:", Implies(cat, small).value)
# Same words, different maths: three kinds of AND on the same two scores
a, b = torch.tensor([0.9]), torch.tensor([0.4])
print("AndMin:", fo.AndMin()(a, b).item(), " AndProd:", fo.AndProd()(a, b).item(),
      " AndLuk:", fo.AndLuk()(a, b).item())


# %% Program 4: a Quantifier squashes many scores into ONE ("for all", "exists")
rule = Implies(IsCat(animals), IsSmall(animals))          # still one score per animal
print("rule per animal:", rule.value, " free_vars:", rule.free_vars)

Forall = ltn.Quantifier(fo.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(fo.AggregPMean(p=2), quantifier="e")
print("FORALL animals: rule =", Forall(animals, rule).value.item())
print("EXISTS animals: rule =", Exists(animals, rule).value.item())
print("free_vars after quantifying:", Forall(animals, rule).free_vars)   # [] = closed formula


# %% Program 5: different aggregators give different answers on the same scores
scores = torch.tensor([0.9, 0.8, 0.1])
for name, agg in [("Min", fo.AggregMin()), ("Mean", fo.AggregMean()),
                  ("PMeanError p=2", fo.AggregPMeanError(p=2)),
                  ("PMeanError p=6", fo.AggregPMeanError(p=6))]:
    print(f"{name:16}", agg(scores, dim=0).item())
# p larger -> closer to Min (stricter "for all")


# %% Program 6: SatAgg = how satisfied are ALL my rules together
x = ltn.Variable("x", torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]))
rule1 = Forall(x, Implies(IsCat(x), IsSmall(x)))
rule2 = Forall(x, Or(IsCat(x), IsSmall(x)))
sat = fo.SatAgg(fo.AggregPMeanError(p=2))(rule1, rule2)
print("rule1:", rule1.value.item(), " rule2:", rule2.value.item(), " overall:", sat.item())


# %% Program 7: LEARNING. Train a predicate so the rules become true
torch.manual_seed(0)
data = ltn.Variable("data", torch.rand(30, 2))             # 30 random points
net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())
P = ltn.Predicate(model=torch.nn.Sequential(net, torch.nn.Flatten(0)))
# Rule we want TRUE: "P(data) holds when the first feature is big"
hi = ltn.Predicate(func=lambda v: (v[:, 0] > 0.5).float())
Q = ltn.Quantifier(fo.AggregPMeanError(p=2), quantifier="f")
opt = torch.optim.Adam(net.parameters(), lr=0.05)
for step in range(201):
    opt.zero_grad()
    sat = fo.SatAgg(fo.AggregPMeanError(p=2))(Q(data, Implies(hi(data), P(data))),
                                              Q(data, Implies(Not(hi(data)), Not(P(data)))))
    (1 - sat).backward()          # loss = 1 - satisfaction ; this is where gradients flow
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3}  satisfaction = {sat.item():.4f}")


# %% Program 8: bug F1. "for all x" over ZERO things. Should be 1.0 (nothing can break the rule)
empty = ltn.Variable("empty", torch.empty(0, 2))
print("rows:", empty.value.shape[0])
for name, agg in [("Mean", fo.AggregMean()), ("PMeanError", fo.AggregPMeanError(p=2)), ("Min", fo.AggregMin())]:
    try:
        r = ltn.Quantifier(agg, quantifier="f")(empty, IsCat(empty)).value.item()
        print(f"{name:11} -> {r}")
    except Exception as e:
        print(f"{name:11} -> CRASH {type(e).__name__}: {str(e)[:50]}")


# %% Program 9: bug F2. p=0 is accepted, then crashes
agg = fo.AggregPMean(p=0)                      # no complaint here
print("constructor accepted p=0")
try:
    ltn.Quantifier(agg, quantifier="f")(animals, IsCat(animals))
except Exception as e:
    print("using it ->", type(e).__name__, ":", e)


# %% Program 10: bug F24. "false IMPLIES false" must be TRUE (1.0)
z = torch.tensor([0.0])
print("stable=True :", fo.ImpliesGoguen(stable=True)(z, z).item(), " (should be 1.0)")
print("stable=False:", fo.ImpliesGoguen(stable=False)(z, z).item())
