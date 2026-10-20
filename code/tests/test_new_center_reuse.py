import numpy as np
import pandas as pd

from analysis.new_center_reuse_report import matched_comparisons
from commitjl.protected_jl import centroid_basis
from experiments.new_center_reuse import reuse_distances, run_job, split_residual, sqdist


def test_decomposition_is_exact_with_true_residuals():
    rng = np.random.default_rng(0)
    x, c0, cn = rng.normal(size=(40, 30)), rng.normal(size=(5, 30)), rng.normal(size=(7, 30))
    u = centroid_basis(c0)
    h, r, e = split_residual(x, c0[0], u)
    hc, rc, ec = split_residual(cn, c0[0], u)
    np.testing.assert_allclose(reuse_distances(h, e, r, hc, ec, rc), sqdist(x, cn), rtol=1e-9, atol=1e-9)


def test_run_job_same_centres_exact():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(loc=5*j, size=(60, 40)) for j in range(5)])
    rows = run_job('toy', x, 5, 0)
    same = [r for r in rows if r['scenario'] == 'same' and r['method'] in ('ours_cross', 'ours_nocross')]
    assert same and all(r['agreement'] == 1.0 for r in same)
    assert all(abs(r['psi_rel_error']) < 1e-8 for r in same if r['method'] == 'ours_nocross')


def test_report_compares_only_common_cells_at_each_budget():
    rows = []
    # PCA is unavailable on the second corpus at width 64. Its missing cell
    # must not change the corpus mix used to compare it with the JL estimator.
    for dataset, ours in [('shared', .9), ('pca_rank_excluded', .1)]:
        for label, score in [('ours_cross_q1', ours), ('pca', .8)]:
            if dataset == 'pca_rank_excluded' and label == 'pca':
                continue
            rows.append(dict(dataset=dataset, k=5, start=0, scenario='refit1',
                             budget=64, label=label, agreement=score,
                             ari=score, r_phi=1., abs_psi=0.))
    result = matched_comparisons(pd.DataFrame(rows)).iloc[0]
    assert result.cells == 1
    assert np.isclose(result.agreement_ours, .9)
    assert np.isclose(result.agreement_other, .8)
    assert np.isclose(result.agreement_delta, .1)
