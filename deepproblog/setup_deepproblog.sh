#!/bin/bash
# Install DeepProbLog 2.1.0 into the active virtual environment (the LTN venv already has torch and atheris).
#   source ../venv/bin/activate && bash setup_deepproblog.sh
set -e
pip install "deepproblog==2.1.0"
python3 -c "import deepproblog, problog, pysdd; print('deepproblog ok')"
