# libFuzzer for DeepProbLog 2.1.0

Same method as the LTN study (`../ltn_combined`): libFuzzer (Atheris) guided by code coverage only, with the bytes it mutates
decoded into the choices of seven dimensions P1–P7 of DeepProbLog's API, and oracles that are independent of the library.

| File | What it is |
|---|---|
| `setup_deepproblog.sh` | `pip install deepproblog==2.1.0` into the active venv (the LTN venv already has torch and atheris) |
| `dpl_fuzz.py` | The libFuzzer harness: eight test families, possible-worlds oracle, gradient oracle, invariants |
| `run_chunks.sh` | Campaign driver: `./run_chunks.sh 4 170` = 4 chunks of 170 s on each of 2 cores |
| `classify_dims_dpl.py`, `deepproblog_dimensions.csv` | Static extraction of the dimensions P1–P7 from DeepProbLog's source (333 parameters) |
| `check_dpl_bugs.py` | Every bug (D1–D14) as a short piece of plain DeepProbLog code that fails the way the bug describes |
| `mutation_check.py` | Plants three bugs in DeepProbLog (in memory) and shows that the oracles detect them |
| `dpl_round3.py`, `run_round3.sh`, `mutation_check_round3.py`, `diff_probe.py` | Round 2/3: metamorphic and first-order oracles (Section 12 of the report), their campaign driver, planted-bug check, and a differential probe against ProbLog |
| `findings/<id>/` | One folder per failure signature of the campaign: `input.bin` (replayable) and `meta.json` |
| `logs_round3/` | libFuzzer output of the 11 round-3 chunks (`b_` = second batch) |
| `logs/` | libFuzzer output of the eight chunks |
| `docs/DeepProbLog_libFuzzer_Report.md` | The report |

## Quick start

```bash
cd ~/nesy_libfuzzer_ltn && source venv/bin/activate
cd deepproblog
bash setup_deepproblog.sh                          # once
python3 classify_dims_dpl.py $(python3 -c "import deepproblog,os;print(os.path.dirname(deepproblog.__file__))")
python3 check_dpl_bugs.py list                     # the bugs
python3 check_dpl_bugs.py D7                       # one bug: its code, run raw (real traceback)
python3 check_dpl_bugs.py                          # all bugs and a summary table (last line: 14 of 14 ...)
python3 dpl_fuzz.py findings/425f75f82c/input.bin  # replay a saved fuzzer input (bug D7)
python3 mutation_check.py                          # do the oracles catch planted bugs?
./run_chunks.sh 1 60                               # 60-second campaign; findings in campaign/findings1 and findings2
```

A full campaign as in the report: `./run_chunks.sh 4 170` (about 12 minutes). The harness prints `WARNING` lines about missing
`__sanitizer_*` functions; they are expected and harmless.

Round 2/3: `./run_round3.sh 4 150` (findings in `campaign3/findings1`); `python3 mutation_check_round3.py`; `python3 diff_probe.py 1 60`. Replay of a finding works as before with `python3 dpl_round3.py <dir>/input.bin`. No round-3 finding exists in the real library.

## Why these dimensions and adaptations (evidence)
The harness has optional switches used by the evidence kit (evidence_kit.zip): see its README. Defaults are unchanged.
