#!/bin/bash
# Campaign: N chunks of libFuzzer per core (Atheris leaks ~10 KB per execution, so we restart; the corpus is kept).
# usage: ./run_chunks.sh <chunks> <seconds-per-chunk>
CH=${1:-4}; SEC=${2:-170}
for w in 1 2; do
  ( for c in $(seq 1 $CH); do
      mkdir -p campaign/corpus$w
      DPL_OUT=campaign/findings$w python3 dpl_fuzz.py campaign/corpus$w -seed=$((w*100+c)) -max_total_time=$SEC -timeout=25 \
         -len_control=0 -max_len=200 -rss_limit_mb=3500 -print_final_stats=1 > campaign/w${w}_c${c}.log 2>&1
    done ) &
done
wait
