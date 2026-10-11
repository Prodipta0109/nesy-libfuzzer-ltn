# Evidence kit: why these dimensions and why these adaptations

Every claim is a command you can run in front of your supervisor. Works for LNN, LTN and DeepProbLog.

## Setup (once)
Unzip next to your three project folders, then tell the scripts where they are:
```
export LNN_DIR=~/nesy_libfuzzer_ltn/lnn            # folder with lnn_fuzz.py and src/
export DPL_DIR=~/nesy_libfuzzer_ltn/deepproblog    # folder with dpl_fuzz.py
export LTN_DIR=~/nesy_libfuzzer_ltn/ltn_combined   # folder with ltn_fuzz_combined.py
export LTN_SRC=~/nesy_libfuzzer_ltn/ltn_torch
```
First copy the three files in `harness_updates/` over the same-named files in those folders (they only add switches; default behaviour is unchanged).

## Question 1: "Why this dimension?"  (seconds)
```
python3 pairs_lnn.py      python3 pairs_ltn.py      python3 pairs_dpl.py
```
Each row is the same library call twice. A normal value passes (control). The value the fuzzer chooses, which is the only change, fails with a named bug.

## Question 2: "Does the fuzzer really need it?"  (about 2-3 min per row)
```
python3 evidence_lnn.py dim      python3 evidence_ltn.py dim      python3 evidence_dpl.py dim
```
Runs the real fuzzer with the full harness and with one dimension removed (same seed, same number of executions) and lists the bugs that disappear.

## Question 3: "Why adaptation A1, A2, A3 ...?"
```
python3 evidence_lnn.py adapt    python3 evidence_ltn.py adapt    python3 evidence_dpl.py adapt
python3 evidence_dpl.py A11      # one adaptation only
python3 evidence_lnn.py list     # all ids
```
Each adaptation is switched off and compared with the full setup: A1 no Python instrumentation, A2 / A9 instrument scope, A3 fixed program instead of decoded structure, A4 oracle versus crash-only, A5 de-duplication, A6 harness false alarm, A7 memory growth, A10 length control, A11 timeout and loop guards, A12 temp directory, A13 step cap (LNN), A8 extra dimensions (LTN).

## Visual summary
`python3 make_report.py` rebuilds `evidence_report.html` from `results/*.json`. Open it in a browser.

## Honest reading
Some rows show a small or no difference in a short budget (for example DeepProbLog A1: the structured harness alone finds most bugs, coverage feedback adds corpus growth and a few extra findings; LTN/LNN A2: only the coverage number is not comparable). Say so; the minimal pairs are the proof of why each dimension exists, the on/off runs show its effect on the search. Budget: LNN 5000, LTN 20000, DeepProbLog 3000 executions, seed 1. Raise it with `--runs N`.
