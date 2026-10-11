#!/bin/bash
# Round 2/3 campaign, one worker: ./run_round3.sh <chunks> <seconds-per-chunk> [worker-number]
CH=${1:-4}; SEC=${2:-170}; W=${3:-1}
mkdir -p campaign3/corpus$W
for c in $(seq 1 $CH); do
  DPL_OUT=campaign3/findings$W python3 dpl_round3.py campaign3/corpus$W -seed=$((W*1000+c)) -max_total_time=$SEC -timeout=25 \
     -len_control=0 -max_len=200 -rss_limit_mb=3500 -print_final_stats=1 > campaign3/w${W}_c${c}.log 2>&1
done
