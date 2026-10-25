# Mutation table (Task 9)

Command: `PYTHONPATH=ltn_torch:lnn/src python -m mr_unified.mutation_study --lib all --cases 2000`
Seed: 0. Each oracle runs up to 2000 cases per mutant and stops at its first firing.
Baseline (unmutated LTN and LNN, same seed and cases): all three oracles silent; the run exited 0, which the CLI returns only when the baseline is silent.

mutant                                        per_operator  full_reference  mr    kind
ImpliesLuk without min(.,1)                   yes           yes             yes   all three
AndLuk uses min                               yes           yes             yes   all three
OrLuk without clamp                           yes           yes             yes   all three
NotStandard clipped at 0.5                    yes           yes             yes   all three
AggregMin ignores last element                no            yes             yes   MR + full-reference only
AndLuk off by 1e-3                            yes           yes             yes   all three
AggregMin wrong axis with two free variables  no            yes             yes   MR + full-reference only
AndLuk stale result per operator id           yes           yes             yes   all three
LNN And upward computed as min                yes           yes             yes   all three
LNN Implies arguments swapped                 yes           yes             yes   all three
LNN Not is identity                           yes           yes             yes   all three
LNN Or upward raised by 0.1                   yes           yes             yes   all three
full_reference needs a reference evaluator; MR does not
