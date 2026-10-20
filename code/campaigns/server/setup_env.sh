#!/usr/bin/env bash
# Create the server venv with the packages the native experiments need. Run on the server.
set -euo pipefail
cd ~/commitjl
command -v python3 >/dev/null
python3 -m venv .venv 2>/dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv .venv; }
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy scipy scikit-learn pandas threadpoolctl pycryptodome pytest
.venv/bin/python -c "import numpy, scipy, sklearn, pandas; print('env ok', numpy.__version__, sklearn.__version__)"
