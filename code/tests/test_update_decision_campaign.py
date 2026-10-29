from experiments.review_decision_campaign import jobs_for
from experiments.review_update_decisions import PROOF_FITS,update_trace
import numpy as np


def test_job_selection_and_controls_do_not_depend_on_native_outcomes():
    entries=[]
    for dataset,k,seed in PROOF_FITS:
        for method in ['learned','pca']:
            for rank in [15,63]:
                entries.append(dict(dataset=dataset,k=k,fit_seed=seed,method=method,rank=rank,
                    batch=1 if dataset=='20ng_tfidf' and k==10 else 2 if dataset=='20ng_tfidf' else 4,
                    path=f'{dataset}_{k}_{seed}_{method}_{rank}.json'))
    jobs=jobs_for(entries)
    assert len(jobs)==32 and len({j['name'] for j in jobs})==32
    for fit in PROOF_FITS:
        selected=[j for j in jobs if j['name'].startswith(f'{fit[0]}_k{fit[1]}_seed{fit[2]}_')]
        assert len(selected)==5
        assert len({j['batch'] for j in selected})==len({j['reps'] for j in selected})==1
        assert sum(j['full'] for j in selected)==1
    assert jobs[0]['reps']==jobs[1]['reps']==1 and jobs[0]['warmup']==0


def test_trace_retains_duplicates_and_marks_reverse_control_separately():
    snapshots={step:np.asarray([[step,0],[-step,0]]) for step in range(1,25)}
    sets,roles=update_trace(snapshots,1)
    assert len(sets)==6 and [r['iteration'] for r in roles[:5]]==[1,2,4,8,16]
    assert roles[-1]['kind']=='reverse_step_diagnostic'
    sets,roles=update_trace({step:np.zeros((2,2),dtype=int) for step in range(1,25)},8)
    assert len(sets)==6 and all(r['unchanged_centers'] for r in roles)
