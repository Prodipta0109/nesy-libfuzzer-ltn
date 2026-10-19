#!/usr/bin/env python3
"""Keep running ltn_fuzz.py in rounds until coverage stops growing.

One round  = ltn_fuzz.py for --round-seconds on the SAME corpus folder (each round continues from the last),
             then cov_replay.py replays the whole corpus and counts the covered lines of `ltn`.
Stop when  = the last --same rounds all report exactly the same covered-line count (default 3),
             or a safety cap is hit (--max-rounds, --max-hours).

ltn_fuzz.py itself is not modified: this wrapper only decides how long to keep it running. Each round gets its own seed,
otherwise libFuzzer would retry the same mutations.

Run:   LTN_SRC=/path/to/ltn_torch python run_until_plateau.py corpus_valid
       LTNFUZZ_MISUSE=1 python run_until_plateau.py corpus_misuse --out plateau_misuse     (include dimension D7)

Environment passed through to ltn_fuzz.py: LTN_SRC, LTNFUZZ_MISUSE, LTNFUZZ_FINDINGS (default ./findings_plateau).
LTNFUZZ_COLLECT is always set to 1 so a finding is saved and fuzzing continues.

Writes in --out (default ./plateau_out):
    plateau_log.csv     one row per round: seconds, corpus size, lines covered, lines added, percent, libFuzzer cov, findings
    round_N/            the full cov_replay.py output of each round
    round_N.fuzz.log    the libFuzzer output of each round

A plateau means the fuzzer is stuck for this harness and round length. It does not mean nothing more exists.
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
env = dict(os.environ)
env["LTNFUZZ_COLLECT"] = "1"
findings_dir = env.setdefault("LTNFUZZ_FINDINGS", os.path.join(os.getcwd(), "findings_plateau"))
os.makedirs(args.corpus, exist_ok=True)
os.makedirs(args.out, exist_ok=True)
FIELDS = ["round", "seed", "fuzz_seconds", "corpus_size", "lines_covered", "new_lines", "percent",
          "functions_called", "libfuzzer_cov", "findings_total", "fuzz_exit_code"]


def count_findings():
    return len(os.listdir(findings_dir)) if os.path.isdir(findings_dir) else 0


history, t_start = [], time.time()
with open(os.path.join(args.out, "plateau_log.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for rnd in range(1, args.max_rounds + 1):
        seed = args.seed0 + rnd
        t0 = time.time()
        fuzz = subprocess.run(
            [sys.executable, os.path.join(HERE, "ltn_fuzz.py"), args.corpus,
             f"-max_total_time={args.round_seconds}", f"-seed={seed}", "-rss_limit_mb=6000"],
            capture_output=True, text=True, env=env)
        text = (fuzz.stdout or "") + (fuzz.stderr or "")
        open(os.path.join(args.out, f"round_{rnd}.fuzz.log"), "w").write(text)
        covs = re.findall(r"cov: (\d+)", text)
        fuzz_secs = round(time.time() - t0)

        if not [f for f in os.listdir(args.corpus) if os.path.isfile(os.path.join(args.corpus, f))]:
            sys.exit(f"round {rnd}: the corpus is empty, so there is nothing to measure. "
                     f"Look at {args.out}/round_{rnd}.fuzz.log for the reason.\n{text[-1200:]}")

        round_out = os.path.join(args.out, f"round_{rnd}")
        cv = subprocess.run([sys.executable, os.path.join(HERE, "cov_replay.py"), args.corpus, "--out", round_out],
                            capture_output=True, text=True, env=env)
        try:
            summ = json.load(open(os.path.join(round_out, "cov_summary.json")))
        except Exception:
            sys.exit(f"round {rnd}: the coverage step failed.\n{cv.stdout[-1500:]}\n{cv.stderr[-1500:]}")

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
            print(f"\nSTOP: coverage unchanged for {args.same} consecutive rounds ({history[-1]} lines). "
                  f"Total {round((time.time() - t_start) / 60, 1)} min, {rnd} rounds.")
            break
        if (time.time() - t_start) / 3600 > args.max_hours:
            print(f"\nSTOP: hit --max-hours {args.max_hours} before a plateau.")
            break
    else:
        print(f"\nSTOP: hit --max-rounds {args.max_rounds} before a plateau.")

print(f"log: {os.path.join(args.out, 'plateau_log.csv')}   findings: {findings_dir}")
