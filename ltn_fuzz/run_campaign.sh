#!/bin/bash
# usage: run_campaign.sh <name> <misuse 0|1> <seconds> <seed>
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"; . "$D/../venv/bin/activate"; mkdir -p "$D/corpus_$1" "$D/logs"
LTNFUZZ_COLLECT=1 LTNFUZZ_MISUSE=$2 LTNFUZZ_FINDINGS=$D/findings_$1 \
  python $D/ltn_fuzz.py $D/corpus_$1 -max_total_time=$3 -rss_limit_mb=6000 -print_final_stats=1 -seed=$4 > $D/logs/$1.log 2>&1
