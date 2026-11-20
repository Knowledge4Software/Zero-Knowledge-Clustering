"""Budget and disjoint-input protections for the optional large-circuit queue."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux server-only coordinator')


def test_no_large_cell_can_spend_the_proof_drain_reserve():
    from campaigns.server_scale_20261101 import admission
    assert admission(20*3600, 5*3600)
    assert not admission(21*3600, 5*3600)
    assert not admission(26.5*3600, 1)


def test_extra_audit_rows_are_disjoint_from_every_primary_source_row():
    from campaigns.published_baselines_laptop import sha
    from commitjl.pilot_io import array_sha
    repo = Path(__file__).resolve().parents[2]
    root = repo/'server_inputs/scale_20261101'
    if not root.exists():
        pytest.skip('Frozen server scale input package not installed')
    pin = json.loads((root/'pin.json').read_text())
    assert sha(root/'fashion_extra.npz') == pin['archive_sha256']
    with np.load(root/'fashion_extra.npz') as data:
        pixels, ids = data['pixels'], data['raw_source_ids']
    canonical = np.random.default_rng(0).choice(70000,5000,replace=False)
    assert len(np.unique(ids)) == 3072 and len(np.intersect1d(ids,canonical)) == 0
    assert ids.tolist() == pin['raw_source_ids']
    assert pixels.shape == (3072,784) and array_sha(pixels) == pin['pixels_sha256']


def test_versioned_adapter_changes_only_size_admission_bounds():
    repo = Path(__file__).resolve().parents[2]
    old = (repo/'code/zk_external/zkml/prototype_audit.rs').read_text()
    new = (repo/'code/zk_external/zkml/prototype_audit_scale_20261101.rs').read_text()
    assert new == old.replace('shape.n<=1024 && shape.d<=10000 && shape.centers>=2 && shape.k<=20',
                              'shape.n<=4096 && shape.d<=10000 && shape.centers>=2 && shape.k<=23')
