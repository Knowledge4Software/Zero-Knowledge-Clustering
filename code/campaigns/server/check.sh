#!/usr/bin/env bash
# Server sanity check: tests for the new modules and one tiny experiment case.
set -euo pipefail
cd ~/commitjl
.venv/bin/pip install -q pytest
.venv/bin/python -m pytest tests/test_certified_head.py tests/test_residual_maps.py -q 2>&1 | tail -2
.venv/bin/python -c "
from data.loaders import load_dataset
for n,kw in [('mnist',dict(n_max=50)),('gisette',dict(n_max=50)),('20ng_tfidf',dict(n_max=50)),('fashion_mnist',dict(n_max=50))]:
    print(n, load_dataset(n, **kw)['X'].shape)
" 2>&1 | grep -v -i warn
