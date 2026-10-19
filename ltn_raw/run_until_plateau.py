#!/usr/bin/env python3
"""Keep fuzzing ltn_hybrid_fuzz.py in rounds until coverage stops growing.

One round  = libFuzzer for --round-seconds on the SAME corpus folder (so every round starts where the last ended),
             then cov_hybrid.py replays the whole corpus and reports how many lines of `ltn` it covers.
Stop when  = the last --same rounds all report exactly the same covered-line count (default 3), or a safety cap hits.

Each round uses a different seed, otherwise libFuzzer would retry the same mutations.

Run:   python run_until_plateau.py corpus_hybrid
       python run_until_plateau.py corpus_hybrid --round-seconds 120 --same 3 --max-rounds 40

Writes in --out (default ./plateau_out):
   plateau_log.csv      one row per round: seconds, corpus size, lines covered, percent, libFuzzer cov, findings
   round_N/             the full cov_hybrid.py output of each round (cov_functions.csv, cov_growth.csv, ...)
   round_N.fuzz.log     the libFuzzer output of each round

Honest limit: "no new lines for 3 rounds" means the fuzzer is stuck for this harness and these round lengths, not
that nothing more exists. Longer rounds or --same 5 make it harder to stop early by bad luck.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("corpus")
ap.add_argument("--round-seconds", type=int, default=60)
ap.add_argument("--same", type=int, default=3, help="stop when this many consecutive rounds have equal coverage")
ap.add_argument("--max-rounds", type=int, default=30, help="safety cap")
ap.add_argument("--max-hours", type=float, default=3.0, help="safety cap on total wall time")
ap.add_argument("--out", default="plateau_out")
ap.add_argument("--seed0", type=int, default=1)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(args.corpus, exist_ok=True)
os.makedirs(args.out, exist_ok=True)
findings_dir = os.environ.get("RAW_FINDINGS", os.path.join(HERE, "findings_hybrid"))
log_path = os.path.join(args.out, "plateau_log.csv")
FIELDS = ["round", "seed", "fuzz_seconds", "corpus_size", "lines_covered", "new_lines", "percent",
          "functions_called", "libfuzzer_cov", "findings_total", "fuzz_exit_code"]


def count_findings():
    return len([d for d in os.listdir(findings_dir)]) if os.path.isdir(findings_dir) else 0


history, t_start = [], time.time()
with open(log_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for rnd in range(1, args.max_rounds + 1):
        seed = args.seed0 + rnd
        t0 = time.time()
        fuzz = subprocess.run(
            [sys.executable, os.path.join(HERE, "ltn_hybrid_fuzz.py"), args.corpus,
             f"-max_total_time={args.round_seconds}", f"-seed={seed}", "-rss_limit_mb=4000"],
            capture_output=True, text=True)
        fuzz_log = os.path.join(args.out, f"round_{rnd}.fuzz.log")
        open(fuzz_log, "w").write((fuzz.stdout or "") + (fuzz.stderr or ""))
        covs = re.findall(r"cov: (\d+)", (fuzz.stderr or "") + (fuzz.stdout or ""))
        fuzz_secs = round(time.time() - t0)

        round_out = os.path.join(args.out, f"round_{rnd}")
        cv = subprocess.run([sys.executable, os.path.join(HERE, "cov_hybrid.py"), args.corpus, "--out", round_out],
                            capture_output=True, text=True)
        try:
            summ = json.load(open(os.path.join(round_out, "cov_summary.json")))
        except Exception:
            sys.exit(f"round {rnd}: coverage step failed.\n{cv.stdout[-1500:]}\n{cv.stderr[-1500:]}")

        prev = history[-1] if history else 0
        history.append(summ["lines_covered"])
        row = dict(round=rnd, seed=seed, fuzz_seconds=fuzz_secs, corpus_size=summ["inputs"],
                   lines_covered=summ["lines_covered"], new_lines=summ["lines_covered"] - prev,
                   percent=summ["percent"], functions_called=f'{summ["functions_called"]}/{summ["functions_total"]}',
                   libfuzzer_cov=covs[-1] if covs else "", findings_total=count_findings(),
                   fuzz_exit_code=fuzz.returncode)
        w.writerow(row)
        fh.flush()
        print(f"round {rnd:>2} | corpus {row['corpus_size']:>5} | lines {row['lines_covered']:>5} "
              f"(+{row['new_lines']:>3}) | {row['percent']:>5}% | libFuzzer cov {row['libfuzzer_cov'] or '-':>6} | "
              f"findings {row['findings_total']} | exit {fuzz.returncode}", flush=True)

        if len(history) >= args.same and len(set(history[-args.same:])) == 1:
            print(f"\nSTOP: coverage unchanged for {args.same} consecutive rounds "
                  f"({history[-1]} lines). Total {round((time.time() - t_start) / 60, 1)} min, {rnd} rounds.")
            break
        if (time.time() - t_start) / 3600 > args.max_hours:
            print(f"\nSTOP: hit --max-hours {args.max_hours} before a plateau.")
            break
    else:
        print(f"\nSTOP: hit --max-rounds {args.max_rounds} before a plateau.")

print(f"log: {log_path}")
