# Mutation table (Task 9)

Command: `PYTHONPATH=ltn_torch:lnn/src python -m mr_unified.mutation_study --lib all --cases 2000`
Seed: 0. Each oracle runs up to 2000 cases per mutant and stops at its first firing. "yes, case N" is the case number (1-based) at which that oracle first fired. Two separate processes gave identical output.
Baseline (unmutated LTN and LNN, same seed and cases): all three oracles silent; the run exited 0, which the CLI returns only when the baseline is silent.

mutant                                        per_operator  full_reference  mr              kind
ImpliesLuk without min(.,1)                   yes, case 2   yes, case 24    yes, case 5     all three
AndLuk uses min                               yes, case 15  yes, case 34    yes, case 1     all three
OrLuk without clamp                           yes, case 16  yes, case 2     yes, case 2     all three
NotStandard clipped at 0.5                    yes, case 14  yes, case 9     yes, case 6     all three
AggregMin ignores last element                no            yes, case 26    yes, case 33    MR + full-reference only
AndLuk off by 1e-3                            yes, case 15  yes, case 4     yes, case 1     all three
AggregMin wrong axis with two free variables  no            yes, case 26    yes, case 1216  MR + full-reference only
AndLuk memo by input shape                    yes, case 17  yes, case 4     yes, case 1     all three
LNN And upward computed as min                yes, case 15  yes, case 34    yes, case 396   all three
LNN Implies arguments swapped                 yes, case 1   yes, case 29    yes, case 52    all three
LNN Not is identity                           yes, case 6   yes, case 29    yes, case 6     all three
LNN Or upward raised by 0.1                   yes, case 13  yes, case 5     yes, case 31    all three
full_reference needs a reference evaluator; MR does not

Notes:

- AggregMin wrong axis: the MR column first fires at case 1216 (missed at 200); full_reference fires at 26. Random formulas rarely produce a two-dimensional grounding with one reduced axis.
- Only the two AggregMin rows discriminate between oracles: per_operator misses both; MR and full_reference both catch them. The four LNN rows are caught by all three oracles and show only that the patching works.
- E3 negative result: no mutant is caught by MR alone (MR-only = 0). Whenever a reference evaluator exists, full_reference catches every value bug MR catches here. MR does not need a reference evaluator; this study does not test that advantage (it needs a library without a reference, e.g. NeuPSL).
