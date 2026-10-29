"""Explicit follow-up: modern RaBitQ numerical control on already frozen inputs.

Registered after the first gate's native outcomes, before RaBitQ outcomes.
Run only when the serial proof queue is idle; never contend with proof timings.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from commitjl.pilot_io import save,sha
from commitjl.published_audit_baselines import distances,interval_decision,point_decision
from commitjl.rabitq_audit_reference import RaBitQAudit
from experiments.published_baseline_gate import DEFAULT,read,measured_batch

REPO=Path(__file__).resolve().parents[2]


def prepare(root):
    out=root/'rabitq_followup';out.mkdir(exist_ok=False)
    save(out/'plan.json',dict(status='frozen_before_RaBitQ_outcomes',
        registered_after_original_native_outcomes=True,source='https://arxiv.org/abs/2405.12497',
        authors='Gao and Long',venue='SIGMOD2024',doi='10.1145/3654970',
        cases=sorted(p.name for p in (root/'inputs').iterdir()),catalog_sizes=[32,128],requests=8,
        seeds='Pinned learning seed plus929292; rotation independent of incoming proposals.',
        failure_probability=.01,query_family='n*k*9 fixed row/center pairs; no adaptive candidates',
        anchor='Exact old nearest group; original IVF-style residual encoding; all row/center distances scanned, no ANN pruning.',
        implementations='Independent float64 one-bit estimator, padded to multiple64. No SIMD, no query quantization, no production-library speed claim.',
        interval_scope='Finite-query statistical envelope from spherical-coordinate quantiles. No deterministic integer or cryptographic certificate. Floating intervals have no outward-rounding proof.',
        input_pins={p.parent.name:sha(p) for p in sorted((root/'inputs').glob('*/pin.json'))},
        implementation_sha256=sha(REPO/'code/commitjl/rabitq_audit_reference.py')))
    save(out/'plan_pin.json',dict(plan_sha256=sha(out/'plan.json')))
    save(out/'execution_pin.json',dict(experiment_sha256=sha(Path(__file__)),
        implementation_sha256=sha(REPO/'code/commitjl/rabitq_audit_reference.py'),
        note='Frozen before RaBitQ outcomes; exact old cost from native preprocessing is used for point and interval decisions.'))


def run(root):
    out=root/'rabitq_followup';plan=read(out/'plan.json')
    assert sha(out/'plan.json')==read(out/'plan_pin.json')['plan_sha256']
    execution=read(out/'execution_pin.json')
    assert sha(Path(__file__))==execution['experiment_sha256']
    assert sha(REPO/'code/commitjl/rabitq_audit_reference.py')==plan['implementation_sha256']
    assert not (out/'results.json').exists()
    queue=read(root/'proofs/matched_queue.json')
    if queue.get('status')=='running':raise RuntimeError('Keep numerical work serial with real proof timings')
    records=[];timings=[]
    with threadpool_limits(limits=1):
        for name in plan['cases']:
            case=root/'inputs'/name;meta=read(case/'pin.json');data=np.load(case/'arrays.npz')
            assert sha(case/'pin.json')==plan['input_pins'][name]
            for n in plan['catalog_sizes']:
                x=data['audit'][:n];banks=data['centers'];old=banks[0]
                tick=time.perf_counter();audit=RaBitQAudit(x,old,meta['fit_seed']+929292,n*len(old)*len(banks),.01)
                prep=time.perf_counter()-tick
                results,times=measured_batch(audit.query,banks)
                truths=[distances(x,b) for b in banks];costs=[sum(map(int,d.min(1))) for d in truths]
                for j,result in enumerate(results[1:]):
                    delta=costs[j+1]-costs[0];estimate=result['cost']-audit.old_cost
                    decision=interval_decision(result['lower'],result['upper'])
                    records.append(dict(case=name,dataset=meta['dataset'],fit_seed=meta['fit_seed'],n=n,request=j,
                        assignment_agreement=float(np.mean(result['labels']==truths[j+1].argmin(1))),
                        point_sign_correct=point_decision(estimate)==point_decision(delta),
                        estimated_change=estimate,true_change=delta,
                        statistical_decision=decision,statistical_conclusive=decision!='inconclusive',
                        statistical_interval_contains_truth=result['lower']-1e-6<=delta<=result['upper']+1e-6,
                        wrong_statistical_decision=decision not in ['inconclusive',point_decision(delta)],
                        full_distance_envelope_contains_truth=bool(np.all(result['distance_lower']<=truths[j+1]+1e-6) and np.all(result['distance_upper']>=truths[j+1]-1e-6))))
                timings.append(dict(case=name,dataset=meta['dataset'],n=n,seconds=times,
                     preparation_seconds=prep,code_bytes=audit.code_bytes,dimension=audit.dimension,
                     rotation_bytes=audit.rotation.nbytes,component_bound=audit.component_bound))
                print(json.dumps(dict(rabitq_complete=name,n=n)),flush=True)
    save(out/'results.json',dict(complete=True,native_only=True,new_proofs=0,records=records,timings=timings,
         no_deterministic_or_ZK_certificate_claim=True,
         point_estimate_reference='Exact old cost already computed by native old-group preprocessing; this remains unauthenticated in this numerical baseline.',
         experiment_sha256=sha(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,default=DEFAULT);args=parser.parse_args()
    (prepare if args.phase=='prepare' else run)(args.root.resolve())
