#!/bin/bash
# One-time setup for the LTN libFuzzer experiments (Ubuntu 22.04/24.04 or WSL Ubuntu).
# Run from the folder that contains ltn_fuzz/ and ltn_combined/:   bash setup.sh
set -e
cd "$(dirname "$0")"

# 1. system packages (Python venv support, git, clang for building Atheris if no wheel fits)
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev git clang

# 2. virtual environment next to the harness folders (the run scripts expect ../venv)
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip

# 3. Python packages. CPU-only PyTorch is enough and much smaller than the CUDA build.
pip install atheris==3.1.0 numpy coverage
pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch

# 4. the fuzzing target, pinned to the commit used in the experiments
if [ ! -d ltn_torch ]; then
  git clone https://github.com/tommasocarraro/LTNtorch.git ltn_torch
fi
git -C ltn_torch checkout d1bd981

# 5. sanity check
python -c "import atheris, torch, sys; sys.path.insert(0, 'ltn_torch'); import ltn; print('OK: atheris', atheris.__version__ if hasattr(atheris, '__version__') else '', 'torch', torch.__version__)"
