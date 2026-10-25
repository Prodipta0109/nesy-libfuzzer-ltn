# libFuzzer for LTNtorch: code, results and setup

| Folder / file | What it is |
|---|---|
| `setup.sh` | One-time setup: venv, Atheris, PyTorch (CPU), LTNtorch cloned at commit `d1bd981` |
| `ltn_fuzz/` | Round 1: libFuzzer harness with the basic oracle (crash, NaN, range, gradient) |
| `ltn_combined/` | Round 2: same harness plus the oracles and seeds of the manual fuzzers in neys_fuzz. Also holds the Round 3 harness `ltn_round3.py` |
| `check_all_bugs.py` | Every bug (F1–F3, F24–F31, F33) as a short piece of plain LTN code that fails the way the bug describes. `python3 check_all_bugs.py F29` shows the code and runs it raw; no argument runs all |
| `check_f1_f2_f3.py` | The first manual check of F1, F2, F3 only |
| `classify_dims.py`, `ltn_dimensions.csv` | Static extraction of the fuzzing dimensions D1–D7 from LTN's source, and its output |
| `docs/` | Final report (`LTN_libFuzzer_Final_Report.md`) and the step-by-step verification files `step1`–`step7` |
| `mr_unified/` | Metamorphic-relation oracle shared by LTN, LNN and PSL, with thin adapters per library (see its `README.md`) |

Each folder has its own README, the corpora libFuzzer kept, the findings (`findings_*/<id>/input.bin`,
`program.txt`, `meta.json`) and the campaign logs.

## Quick start (Ubuntu or WSL Ubuntu)

```bash
tar xzf nesy_libfuzzer_ltn.tar.gz && cd nesy_libfuzzer_ltn
bash setup.sh                                   # once
. venv/bin/activate
PYTHONPATH=ltn_torch python3 check_all_bugs.py list      # the bugs
PYTHONPATH=ltn_torch python3 check_all_bugs.py F29       # one bug: its code, run raw (real traceback)
PYTHONPATH=ltn_torch python3 check_all_bugs.py           # all bugs and a summary table
./ltn_combined/run_campaign.sh quick 0 60 1     # 60-second combined campaign
ls ltn_combined/findings_quick                   # one folder per distinct failure
cd ltn_combined && LTNFUZZ_NO_INSTRUMENT=1 python repro.py findings_quick/<id>/input.bin
```

Full campaigns used in the report: `./ltn_combined/run_chunks.sh valid 0 4 300` and
`./ltn_combined/run_chunks.sh misuse 1 4 300` (20 minutes each).

Round 3 (new dimension choices and oracles, 4 minutes):

```bash
cd ltn_combined
PYTHONPATH=../ltn_torch python3 ltn_round3.py -max_total_time=240 -rss_limit_mb=3000
ls findings_r3                                   # one folder per distinct failure signature
```

Copy files into this folder with `cp` inside WSL. Dragging them in through Explorer creates `:Zone.Identifier`
marker files, which are ignored by git (`.gitignore`) and can be removed with
`find . -name "*:Zone.Identifier" -delete`.
