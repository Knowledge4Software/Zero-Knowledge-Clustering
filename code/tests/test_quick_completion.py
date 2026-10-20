from pathlib import Path
from campaigns.quick_completion import baseline_tasks
from experiments.rq2345_complete import digest


def test_baselines_are_small_and_quality_lookup_matches():
    tasks=baseline_tasks(Path('/original/rq1'))
    assert len(tasks)==6
    assert [t['stage'] for t in tasks[:3]]==['rq2_quality']*3
    assert {t['row']['variant'] for t in tasks}=={'dense','coord','pca-public'}
    for task in tasks:
        row=task['row']
        assert row['proof_n_max']==32 and row['m']==16 and row['k']==5
        assert row['precision_bits']==row['normalization_reference_bits']==24
        if task['stage']=='rq2_selected':
            lookup=dict(stage='rq2_quality',name=task['name'],directory=task['directory'],
                        row=row,seeds=10,starts=5)
            assert any(t['id']==digest(lookup) for t in tasks)
