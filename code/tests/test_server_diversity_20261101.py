"""Data separation, matched task identity and immutable-resume protections."""
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux server coordinator')


def test_har_respects_official_subject_partitions():
    from campaigns.server_diversity_20261101 import source_rows
    train, audit, pin, k = source_rows('har', 701)
    assert train.shape == audit.shape == (1024, 561) and k == 6
    assert not set(pin['training_subjects']) & set(pin['audit_subjects'])
    again = source_rows('har', 701)
    assert np.array_equal(train, again[0]) and np.array_equal(audit, again[1])


def test_mnist_training_and_audit_are_disjoint():
    from campaigns.server_diversity_20261101 import source_rows
    train, audit, pin, k = source_rows('mnist', 709)
    assert train.shape == audit.shape == (1024, 784) and k == 5
    assert not set(pin['training_ids']) & set(pin['audit_ids'])


def test_audit_and_incoming_rows_cannot_change_quantization_scale():
    from campaigns.server_diversity_20261101 import quantize
    training = np.ones((1024, 3))
    audit = np.ones((1024, 3))
    original = quantize(training, audit)
    training[512:] = 1000
    audit[:] = 2000
    changed = quantize(training, audit)
    assert original[2] == changed[2] == 4095
    assert np.array_equal(original[0][:512], changed[0][:512])
    assert np.max(changed[1]) == 4095


def test_matched_inputs_keep_six_centers_and_reject_modified_inputs(tmp_path, monkeypatch):
    import campaigns.server_diversity_20261101 as module
    from campaigns.published_baselines_laptop import save, read, sha
    from commitjl.pilot_io import array_sha
    monkeypatch.setattr(module, 'server_profile', lambda: {'workers':12})
    fitted = tmp_path/'fitted/har_seed701'
    fitted.mkdir(parents=True)
    rng = np.random.default_rng(37)
    data = dict(training=rng.integers(-20,20,(512,84)), audit=rng.integers(-20,20,(1024,84)),
                centers=rng.integers(-20,20,(5,6,84)))
    for method in ['ordered_head','pca']:
        data[method+'_basis'] = rng.integers(-100,100,(63,84))
        data[method+'_auxiliary'] = rng.integers(-100,100,(79,84))
    np.savez_compressed(fitted/'arrays.npz', **data)
    save(fitted/'pin.json', dict(dataset='har', seed=701, k=6, scale=1.,
         arrays_sha256=sha(fitted/'arrays.npz'), training_sha256=array_sha(data['training']),
         basis_fitting_seconds={'ordered_head':1., 'pca':1.},
         proposal_recipe=[{'batch':0,'rate':r} for r in [.125,.25,.5,1.]]))
    save(tmp_path/'plan.json', dict(selection='First four training-only proposals'))
    case = module.materialize(tmp_path, 'har', 701)
    left = read(case/'ordered_head/full/input.json')
    right = read(case/'pca/full/input.json')
    assert left == right and left['model']['k'] == 6
    assert len(left['reuse_centers']) == 5 and len(left['x']) == 128
    assert all(len(bank) == 6 for bank in left['reuse_centers'])
    assert read(case/'approved_task.json')['reuse_centers'] == left['reuse_centers']
    assert module.materialize(tmp_path, 'har', 701) == case
    left['x'][0][0] += 1
    save(case/'ordered_head/full/input.json', left)
    with pytest.raises(AssertionError):
        module.materialize(tmp_path, 'har', 701)


def test_unknown_sources_do_not_silently_become_text():
    from campaigns.server_diversity_20261101 import source_rows
    with pytest.raises(ValueError):
        source_rows('unknown', 701)
