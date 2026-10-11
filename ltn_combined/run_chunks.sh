#!/bin/bash
# usage: run_chunks.sh <name> <misuse 0|1> <chunks> <secs-per-chunk>
# libFuzzer resumes from the corpus each chunk; restarting avoids Atheris' memory growth.
D="$(cd "$(dirname "$0")" && pwd)"; mkdir -p $D/corpus_$1 $D/logs
for c in $(seq 1 $3); do
  ORACLES=${ORACLES:-all} $D/run_campaign.sh $1 $2 $4 $((100+c))
  cp $D/logs/$1.log $D/logs/$1.chunk$c.log
done
