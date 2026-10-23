#!/bin/bash
# usage: ./run.sh [seconds]      MR-guided fuzzing with Atheris (needs ../venv from setup.sh)
D="$(cd "$(dirname "$0")" && pwd)"; . "$D/../venv/bin/activate"; mkdir -p "$D/corpus"
python "$D/mr_fuzz.py" "$D/corpus" -max_total_time=${1:-60} -rss_limit_mb=3000
