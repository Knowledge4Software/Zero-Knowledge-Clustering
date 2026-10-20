import numpy as np

from experiments.shift_owner_reuse import run_job


def test_both_settings_run_on_toy_data():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(6), 80)
    x = rng.normal(size=(len(y), 150)) + 4*np.eye(6, 150)[y]*8
    for setting in ('shift', 'owners'):
        rows = run_job('toy', x, y, setting, 5, 0)
        methods = {r['method'] for r in rows}
        assert {'pca_energy', 'pca_energy_oracle', 'learned_energy', 'ours_cross', 'jl'} <= methods
