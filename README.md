# libFuzzer for LTNtorch: code, results and setup

| Folder / file | What it is |
|---|---|
| `setup.sh` | One-time setup: venv, Atheris, PyTorch (CPU), LTNtorch cloned at commit `d1bd981` |
| `ltn_fuzz/` | Round 1: libFuzzer harness with the basic oracle (crash, NaN, range, gradient) |
| `ltn_combined/` | Round 2: same harness plus the oracles and seeds of the manual fuzzers in neys_fuzz |

Each folder has its own README, the corpora libFuzzer kept, the findings (`findings_*/<id>/input.bin`,
`program.txt`, `meta.json`) and the campaign logs.

## Quick start (Ubuntu or WSL Ubuntu)

```bash
tar xzf nesy_libfuzzer_ltn.tar.gz && cd nesy_libfuzzer_ltn
bash setup.sh                                   # once
. venv/bin/activate
python -W ignore ltn_fuzz/confirm_findings.py   # F1-F3 without the fuzzer
./ltn_combined/run_campaign.sh quick 0 60 1     # 60-second combined campaign
ls ltn_combined/findings_quick                   # one folder per distinct failure
cd ltn_combined && LTNFUZZ_NO_INSTRUMENT=1 python repro.py findings_quick/<id>/input.bin
```

Full campaigns used in the report: `./ltn_combined/run_chunks.sh valid 0 4 300` and
`./ltn_combined/run_chunks.sh misuse 1 4 300` (20 minutes each).
