"""Evidence preservation and input protections for the maximum-RAM probe."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform!='linux',reason='Linux-only remote coordinator')


def test_retiring_private_key_preserves_inputs_and_public_evidence(tmp_path):
    from campaigns.server_capacity_20261101 import retire_private_key
    from commitjl.pilot_io import sha
    case = tmp_path/'case'
    private = case/'zkml_private'
    private.mkdir(parents=True)
    for name in ['pk','params','vk','proof','input.json']:
        (private/name).write_bytes(name.encode())
    digest = sha(private/'pk')
    retire_private_key(tmp_path,case)
    assert not (private/'pk').exists()
    assert all((private/name).exists() for name in ['params','vk','proof','input.json'])
    receipt = json.loads((case/'private_key_retirement.json').read_text())
    assert receipt['sha256']==digest and receipt['bytes']==2


def test_retirement_refuses_paths_outside_its_owned_output(tmp_path):
    from campaigns.server_capacity_20261101 import retire_private_key
    root = tmp_path/'output'
    root.mkdir()
    case = tmp_path/'outside'
    (case/'zkml_private').mkdir(parents=True)
    key = case/'zkml_private/pk'
    key.write_bytes(b'preserve')
    with pytest.raises(AssertionError):
        retire_private_key(root,case)
    assert key.read_bytes()==b'preserve'


def test_capacity_inputs_preserve_previous_prefix_and_exclude_training_source():
    from commitjl.pilot_io import array_sha,sha
    repo = Path(__file__).resolve().parents[2]
    root = repo/'server_inputs/capacity_20261101'
    previous = repo/'server_inputs/scale_20261101'
    if not root.exists():
        pytest.skip('Frozen input package not installed')
    pin = json.loads((root/'pin.json').read_text())
    assert sha(root/'fashion_extra.npz')==pin['archive_sha256']
    with np.load(root/'fashion_extra.npz') as data:
        pixels,ids = data['pixels'],data['raw_source_ids']
    with np.load(previous/'fashion_extra.npz') as old:
        assert np.array_equal(pixels[:3072],old['pixels'])
        assert np.array_equal(ids[:3072],old['raw_source_ids'])
    canonical = np.random.default_rng(0).choice(70000,5000,replace=False)
    assert len(np.unique(ids))==7168 and not np.intersect1d(ids,canonical).size
    assert ids.tolist()==pin['raw_source_ids']
    assert pixels.shape==(7168,784) and array_sha(pixels)==pin['pixels_sha256']


def test_capacity_adapter_changes_only_the_size_admission_bounds():
    repo = Path(__file__).resolve().parents[2]
    old = (repo/'code/zk_external/zkml/prototype_audit.rs').read_text()
    new = (repo/'code/zk_external/zkml/prototype_audit_capacity_20261101.rs').read_text()
    assert new==old.replace('shape.n<=1024 && shape.d<=10000 && shape.centers>=2 && shape.k<=20',
                            'shape.n<=8192 && shape.d<=10000 && shape.centers>=2 && shape.k<=24')
