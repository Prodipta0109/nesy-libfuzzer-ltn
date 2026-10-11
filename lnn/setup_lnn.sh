#!/bin/bash
# Fetch IBM LNN (commit c88f513, version 1.1.0) next to the scripts and install its small dependencies into the active venv.
#   source ../venv/bin/activate && bash setup_lnn.sh
# LNN needs Python >= 3.11 and torch >= 2.13 (the LTN/DeepProbLog venv already has torch and atheris).
set -e
[ -d src/.git ] || git clone https://github.com/IBM/LNN src
git -C src checkout c88f513
pip install matplotlib networkx pandas tabulate tqdm
python3 -c "import sys; sys.path.insert(0,'src'); import lnn; print('lnn ok')"
