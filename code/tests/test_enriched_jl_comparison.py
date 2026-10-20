import json
from pathlib import Path

import pytest

from experiments.enriched_jl_comparison import pca_spec, summarize, METRICS


def test_pca_spec_always_fits_its_actual_dimension():
    for method in ['pca_full', 'pca_randomized', 'pca_randomized_fast']:
        for m in [24, 128, 256]:
            spec = pca_spec(method, m, 2)
            assert spec['n_components'] == m
            assert spec['svd_solver'] == ('full' if method == 'pca_full' else 'randomized')
    assert pca_spec('pca_randomized_fast', 128, 0)['iterated_power'] == 2
    with pytest.raises(ValueError): pca_spec('unknown', 128, 0)


def test_paired_screen_does_not_count_capped_or_fixed_only_ari_as_win(tmp_path):
    import pandas as pd
    folder = tmp_path/'rows'; folder.mkdir()
    for method, seeds in [('protected_plus12', [0, 1]), ('pca_full', [-1])]:
        for seed in seeds:
            for start in [0, 1]:
                row = dict(dataset='fixture', method=method, seed=seed, start=start, m=128,
                           training_hit_cap=False, fresh_hit_cap=False, heldout_hit_cap=False)
                row.update({m: .5 for m in METRICS})
                row.update(fixed_ari=1., fresh_ari=.5, validation_fresh_ari=.5, test_fresh_ari=.5)
                if method == 'protected_plus12':
                    row.update(cold_fresh_seconds=.25, fixed_ari=1.)
                else:
                    row.update(fixed_ari=.8)
                (folder/f'{method}-{seed}-{start}.json').write_text(json.dumps(row))
    summarize(tmp_path, bootstrap=50)
    comparison = pd.read_csv(tmp_path/'comparisons.csv').iloc[0]
    assert comparison['cold_fresh_speedup_mean'] == 2
    assert not comparison['observed_speed_and_fresh_ari_win']
    for p in folder.glob('protected_plus12*'):
        row=json.loads(p.read_text()); row.update(fresh_ari=.8, test_fresh_ari=.8)
        p.write_text(json.dumps(row))
    summarize(tmp_path, bootstrap=50)
    assert pd.read_csv(tmp_path/'comparisons.csv').iloc[0]['ci_speed_and_fresh_ari_win']
    p=next(folder.glob('protected_plus12*')); row=json.loads(p.read_text());row['fresh_hit_cap']=True
    p.write_text(json.dumps(row)); summarize(tmp_path, bootstrap=50)
    assert not pd.read_csv(tmp_path/'comparisons.csv').iloc[0]['ci_speed_and_fresh_ari_win']
