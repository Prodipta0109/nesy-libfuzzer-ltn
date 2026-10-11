# DeepProbLog 2.1.0 libFuzzer study (Pro-libfuzz, second library)

Date: 2026-10-11. Method identical to the LTN study: Atheris/libFuzzer, coverage-only guidance, dimensions P1-P7 derived by classify_dims_dpl.py (333 parameters, 317 unvalidated).

- Campaign: 34,292 executions (8 chunks x ~170 s, 2 cores), coverage 2,911 / 2,916 edges, ~25 exec/s.
- Adaptations: A1-A8 reused; A9 instrument deepproblog+problog, A10 -len_control=0 -max_len=200, A11 -timeout + loop counters, A12 temp-dir isolation and file-name forms.
- Oracles: possible-worlds enumeration, torch-autograd gradient, invariants. Validated by 3 planted bugs (mutation_check.py).
- Result: no wrong probability or gradient in inference; 14 bugs D1-D14 around it (13 by the fuzzer, D13 persistent-cache by code reading). All reproduce with check_dpl_bugs.py (14 of 14).
- Most serious: D7 top-k network crashes in training mode when softmax has exact zeros; D5 partly learnable AD forced to mass > 1; D6 ZeroDivisionError in SGD.step; D4 bare file name fails in save_state/Logger/snapshot_name.
- Not covered: ApproximateEngine (needs SWI-Prolog), image datasets, TermEmbedder, example apps. PySDD native code unobserved.
- Next: report upstream (D7, D5, D6 first); then NeurASP or LNN.

## Round 2/3 (11 Oct 2026): metamorphic and first-order oracles
dpl_round3.py: five families (t_meta_prog: clause shuffle / atom renaming / unused facts / duplicated clause / extra rule; t_fo: variables, constants, existential and non-ground queries; t_fo_ad: annotated disjunctions and probabilistic rules with variables; t_batch: batch vs single query, cache on/off; t_domain_perm: permuted domain and network columns). 17,257 executions, 601 inputs kept, 2,193 edges in this harness. No failure in the real library; planted bugs (negate, plus, times) caught by the possible-worlds families (21-40 of 40 inputs), unmodified library 0. Informal differential probe against ProbLog on 60 programs with arithmetic, findall, negation, disjunction: no difference. Observation: negation with an unbound variable is evaluated left to right (Prolog semantics), a trap, not a bug.

Evidence kit (11 Oct 2026): runnable with/without experiments and minimal pairs for every dimension and adaptation (evidence_kit.zip).
