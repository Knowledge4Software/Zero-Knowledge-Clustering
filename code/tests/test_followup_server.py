import json
from pathlib import Path

from campaigns.followup_server import tasks_for, queue_plan
from experiments.rq2345_complete import digest, DATASETS


def original_fixture(tmp_path):
    for name in DATASETS:
        folder=tmp_path/name
        folder.mkdir()
        (folder/'selected_operating_points.csv').write_text(
            'method,k,m,q\nsparse_jl,5,64,2\n',encoding='utf-8')
    return tmp_path


def test_real_quality_lookup_matches_selected_worker(tmp_path):
    tasks=tasks_for('real',original_fixture(tmp_path))
    for selected in [t for t in tasks if t['stage']=='rq2_selected' and t['row']['variant']=='sparse']:
        lookup=dict(stage='rq2_quality',name=selected['name'],directory=selected['directory'],
                    row=selected['row'],seeds=10,starts=5)
        assert any(t['id']==digest(lookup) for t in tasks)
        assert selected['row']['proof_n_max']>=selected['row']['k']
        assert selected['row']['precision_bits']==selected['row']['normalization_reference_bits']


def test_precision_uses_selected_dimension_and_fixed_scale(tmp_path):
    original=original_fixture(tmp_path)
    (original/'dbscan_margin/selected_operating_points.csv').write_text('method,k,m,q\npca,5,64,0\n')
    tasks=tasks_for('precision',original)
    assert len(tasks)==16
    for task in tasks:
        assert task['row']['normalization_reference_bits']==24
        assert task['row']['precision_bits'] in [16,24]
        if task['name']=='dbscan_margin':
            assert task['scope']=='exploratory_no_eligible_sparse_RQ1_point'
        else:
            assert task['row']['m']==64


def test_ladder_and_queue_are_finite_and_separate(tmp_path):
    tasks=tasks_for('sizing',tmp_path/'original')
    assert len(tasks)==6
    assert sum(t['adaptive'] for t in tasks)==3
    for task in tasks:
        row=task['row']
        assert row['dataset']['kwargs']['n']*row['dataset']['kwargs']['d']<=16384
    plan=queue_plan(tmp_path/'followup',tmp_path/'original')
    assert len(plan)==5
    assert sum(p['timeout_seconds'] for p in plan)==17*3600
    assert all(p['timeout_seconds']>0 for p in plan)
