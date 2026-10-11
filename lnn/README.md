# libFuzzer for LNN 1.1.0

Same method as the LTN and DeepProbLog studies: libFuzzer (Atheris) guided by code coverage only, with the bytes it mutates
decoded into the choices of seven dimensions L1–L7 of LNN's API, and oracles that are independent of the library.

| File | What it is |
|---|---|
| `setup_lnn.sh` | Clones IBM/LNN at commit c88f513 into `./src` and installs its small dependencies (LNN is used from source, not pip-installed) |
| `lnn_fuzz.py` | The libFuzzer harness: nine test families, closed-form Łukasiewicz oracles, classical and downward soundness, convergence oracle |
| `run_chunks.sh` | Campaign driver: `./run_chunks.sh 4 170` = 4 chunks of 170 s on each of 2 cores |
| `classify_dims_lnn.py`, `lnn_dimensions.csv` | Static extraction of the dimensions L1–L7 from LNN's source (173 parameters) |
| `check_lnn_bugs.py` | Every bug (L1–L8, plus L1b) as a short piece of plain LNN code that fails the way the bug describes |
| `mutation_check.py` | Plants three bugs in LNN (in memory) and shows that the oracles detect them |
| `lnn_round3.py`, `run_round3.sh`, `mutation_check_round3.py` | Round 2/3: metamorphic oracles, binary predicates, nested quantifiers (Section 12 of the report) |
| `findings_round3/<id>/`, `logs_round3/` | Replayable inputs (bugs L6–L8) and libFuzzer output of the 11 round-3 chunks (`b_` = second batch) |
| `findings/<id>/` | One folder per failure signature of the campaign: `input.bin` (replayable) and `meta.json` |
| `logs/` | libFuzzer output of the 16 chunks of the campaign |
| `phase1_hangs/` | The first, interrupted campaign (no step limit): libFuzzer timeout inputs and logs |
| `docs/LNN_libFuzzer_Report.md` | The report |

## Quick start

```bash
cd ~/nesy_libfuzzer_ltn && source venv/bin/activate
cd lnn
bash setup_lnn.sh                                  # once
python3 classify_dims_lnn.py src/lnn               # the dimensions
python3 check_lnn_bugs.py list                     # the bugs
python3 check_lnn_bugs.py L2                       # one bug: its code, run raw (real traceback / failed assertion)
python3 check_lnn_bugs.py | tail -1                # all bugs (last line: 9 of 9 ...)
python3 lnn_fuzz.py findings/7afeae3fbc/input.bin  # replay a saved fuzzer input (bug L4)
python3 mutation_check.py                          # do the oracles catch planted bugs?
./run_chunks.sh 1 60                               # 60-second campaign; findings in campaign/findings1 and findings2
```

A full campaign as in the report: `./run_chunks.sh 8 170` (about 25 minutes). The harness prints `WARNING` lines about missing
`__sanitizer_*` functions; they are expected and harmless. The inputs in `phase1_hangs/` are from the first harness version and
no longer hang the current harness (it limits every inference); the bug they exposed is demonstrated by `check_lnn_bugs.py L1b`.

Round 2/3: `./run_round3.sh 4 150` (findings in `campaign3/findings1`); `python3 lnn_round3.py findings_round3/a43a34cab0/input.bin` replays bug L6; `python3 mutation_check_round3.py`.

## Why these dimensions and adaptations (evidence)
The harness has optional switches used by the evidence kit (evidence_kit.zip): see its README. Defaults are unchanged.
