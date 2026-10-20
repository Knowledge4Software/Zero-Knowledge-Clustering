import json

import numpy as np
import pytest

from experiments.rq1_complete import bootstrap_crossed, read_rows, row_key, worker
from commitjl.sparse_gen import sparse_matrix_gen


def test_sparse_execution_matches_canonical_dense():
    from scipy.sparse import csr_matrix
    X = np.random.default_rng(7).normal(size=(13, 9))
    _, A = sparse_matrix_gen(b'fixture', 4, 9, 2)
    np.testing.assert_allclose((csr_matrix(A / np.sqrt(2)) @ X.T).T,
                               X @ (A / np.sqrt(2)).T, atol=1e-14)


def test_crossed_bootstrap_preserves_bounded_constant():
    assert bootstrap_crossed(np.ones((3, 4)), draws=20) == (1., 1., 1.)
    with pytest.raises(ValueError):
        bootstrap_crossed([[1., float('nan')]])


def test_export_refuses_active_writer_and_releases_lock_on_error(tmp_path):
    from experiments.common import campaign_lock
    from analysis.rq1_export import export
    with campaign_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another campaign/export'):
            export(tmp_path)
    # Export errors must release the lock so a later operation can proceed.
    with pytest.raises(ValueError,match='No dataset manifests'):
        export(tmp_path)
    with campaign_lock(tmp_path):
        pass


def test_rq1_cli_refuses_locked_output_before_starting_workers(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from experiments.common import campaign_lock
    code=Path(__file__).resolve().parents[1]
    with campaign_lock(tmp_path):
        result=subprocess.run([sys.executable,'-m','experiments.rq1_complete','--run',
            '--datasets','gaussian_easy','--out',str(tmp_path),'--seeds','1','--starts','1',
            '--methods','sparse_jl','--k','5','--m','16','--q','1'],
            cwd=code,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'Another campaign/export' in result.stderr
    assert not (tmp_path/'config.json').exists()


def test_complete_worker_axes_resume_and_provenance(tmp_path, monkeypatch):
    import experiments.rq1_complete as module
    monkeypatch.setitem(module.DATASETS, 'tiny', ('gaussian_mixture',
        dict(n=24, d=8, k=2, gamma=3., seed=123)))
    cfg = dict(scope='paper', starts=2, seeds=3, k=[2], m=[4], q=[2],
               eps=.15, pairs=20, threads=1, bootstrap=30,
               tau_ari=.8, tau_phi=1.1, source_sha256='fixture')
    worker('tiny', cfg, tmp_path)
    rows = read_rows(tmp_path / 'rows.jsonl')
    assert len(rows) == 2 + 3 * 2
    assert len({row_key(row) for row in rows}) == len(rows)
    sparse = [r for r in rows if r['method'] == 'sparse_jl']
    assert {(r['projection_seed'], r['clustering_start']) for r in sparse} == {
        (s, c) for s in range(3) for c in range(2)}
    assert all(r['qualified_violations'] == 0 for r in sparse)
    assert all(r['r_phi'] > 0 for r in sparse)
    worker('tiny', cfg, tmp_path)
    assert len(read_rows(tmp_path / 'rows.jsonl')) == len(rows)
    with pytest.raises(ValueError, match='differs'):
        worker('tiny', {**cfg, 'seeds': 4}, tmp_path)


def test_interrupted_last_line_is_recoverable(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text(json.dumps({'ok': 1}) + '\n{"partial":')
    assert read_rows(path) == [{'ok': 1}]
    assert path.read_bytes().endswith(b'\n')


def test_all_methods_artifacts_and_export(tmp_path, monkeypatch):
    import experiments.rq1_complete as module
    import pandas as pd
    from analysis.rq1_export import export
    monkeypatch.setitem(module.DATASETS, 'tiny', ('gaussian_mixture',
        dict(n=30, d=12, k=3, gamma=2., seed=123)))
    cfg = dict(scope='paper', starts=2, seeds=2, k=[3], m=[4], q=[1, 2],
               eps=.15, pairs=30, threads=1, bootstrap=30,
               tau_ari=.8, tau_phi=1.1, source_sha256='fixture', methods=module.METHODS)
    directory = tmp_path / 'tiny'
    worker('tiny', cfg, directory)
    rows = read_rows(directory / 'rows.jsonl')
    assert {r['method'] for r in rows} == set(module.METHODS) | {'full'}
    assert len([r for r in rows if r['method'] == 'pca']) == 2
    assert len([r for r in rows if r['method'] == 'sparse_jl']) == 8
    X = np.load(directory / 'artifacts/dataset.npz')['X']
    for row in rows:
        if row['method'] == 'full':
            continue
        artifact = np.load(directory / row['pointwise_artifact'])
        assert artifact['labels'].shape == (2, len(X))
        assert len(artifact['pair_delta']) == 30
        assert artifact['assigned_original_sq'].shape == (2, len(X))
        projected = np.load(directory / row['projection_artifact'])['Z']
        assert projected.shape == (len(X), 4)
        assert row['qualified_violations'] == 0
    output = export(tmp_path)
    assert (output / 'tiny_k3_fidelity.pdf').stat().st_size > 1000
    assert (output / 'fidelity_table.tex').exists()
    table = pd.read_csv(output / 'fidelity_table.csv')
    assert set(table.method) == set(module.METHODS) | {'full'}
    assert table[table.method == 'pisot_surrogate'].status.eq('unverified surrogate').all()
    summary = pd.read_csv(directory / 'summary.csv')
    assert summary.complete.all()
    assert not summary[summary.method == 'pisot_surrogate'].eligible.any()
    # A missing whole method must remain visible in the completion inventory.
    keep = [r for r in rows if r['method'] != 'gaussian']
    (directory / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in keep))
    export(tmp_path)
    inventory = pd.read_csv(output / 'completion_inventory.csv')
    missing = inventory[inventory.method == 'gaussian']
    assert missing.status.eq('not_completed').all()
    assert missing.rows.eq(0).all()
