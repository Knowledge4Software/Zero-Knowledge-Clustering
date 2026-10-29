"""Provisional native-certificate checks and timing bounds for an active run.

Never labels unfinished work a measured speedup. Full truth is an independent
diagnostic and does not change the approved workload or the prover's decisions.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from commitjl.group_moment_reference import prepare_groups
from commitjl.group_change_bounds import correction_bounds
from commitjl.reuse_certificates import certificate


def read(path):
    data=path.read_bytes();return json.loads(data),hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();parent=args.parent.resolve();started=time.time()
    state,state_sha=read(root/'status.json');inp,input_sha=read(root/'input.json');task,task_sha=read(root/'approved.json')
    records,records_sha=read(root/'bundle/group_query_records_progress.json')
    protocol,_=read(root/'protocol.json');reference,_=read(parent/'status.json')
    full=reference['cases']['full']['wall_seconds'];old_head=reference['cases']['head_r63']['wall_seconds']
    spent=sum(resource['wall_seconds'] for resource in state['resources'].values())+protocol['fitting_seconds']
    full_cold=json.loads((parent/'checked_comparison/summary.json').read_text()) if (parent/'checked_comparison/summary.json').exists() else None
    model=inp['model'];rows=np.asarray(inp['x'],dtype=np.int64);old=model['centers'];old_cert=certificate(model,rows,old)
    groups=prepare_groups(rows,old_cert['labels'],model['k'])
    def cost(centers):
        c=np.asarray(centers,dtype=np.int64);return int(((rows[:,None,:]-c[None,:,:])**2).sum(2).min(1).sum())
    base=cost(old);checks=[];native_conclusive=0;batch_proofs=0;batch_times=[];aggregate_count=0
    for i,centers in enumerate(task['reuse_centers']):
        bounds=correction_bounds(model,rows,groups,old,centers)
        truth=cost(centers)-base;den=bounds['cost_denominator'];lo=bounds['change_lower_scaled'];hi=bounds['change_upper_scaled']
        assert lo<=truth*den<=hi
        outcome='worsening' if lo>0 else 'improvement' if hi<0 else 'equal' if lo==hi==0 else 'inconclusive'
        if outcome!='inconclusive':
            native_conclusive+=1;assert outcome==('worsening' if truth>0 else 'improvement' if truth<0 else 'equal')
        record=records[i];aggregate=record.get('assigned') or {}
        if aggregate.get('file'):
            aggregate_count+=1;assert int(record['assigned_change'])*den==bounds['assigned_scaled']
        batches=record.get('batches') or [];batch_proofs+=len(batches)
        for b,batch in enumerate(batches):
            assert batch['indices']==task['queries'][b]
            assert int(batch['change_lower_scaled'])==sum(int(bounds['row_lower'][j]) for j in batch['indices'])
            assert int(batch['change_upper_scaled'])==sum(int(bounds['row_upper'][j]) for j in batch['indices'])
            batch_times.extend(batch['prove_seconds'])
        checks.append(dict(set=i,source_set=protocol['proposal_source_indices'][i],truth=truth,lower=str(lo),upper=str(hi),denominator=str(den),native_decision=outcome,completed_switching_batches=len(batches),all_batches_proved=len(batches)==len(task['queries'])))
    quality=[]
    with (parent/'quality.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['rank']=='63' and row['scope']=='proof_catalog' and int(row['set']) in protocol['proposal_source_indices']:quality.append(row)
    assert len(quality)==len(task['reuse_centers'])
    total=len(task['reuse_centers'])*len(task['queries']);pending=total-batch_proofs
    snapshot=dict(final=False,kind='provisional native check and measured active-time lower bound',snapshot_unix=started,check_finished_unix=time.time(),
        active_phase=state['active_phase'],active_seconds_spent_at_snapshot=spent,
        full_measured_active_seconds=full,previous_rank63_measured_active_seconds=old_head,
        full_over_new_upper_bound_ignoring_all_remaining_work=full/spent,
        seconds_left_in_fivefold_full_budget=full/5-spent,fivefold_speed_target_still_possible_on_this_run=spent<full/5,
        updates=len(checks),native_conclusive=native_conclusive,native_regressions=sum(x['truth']>0 for x in checks),
        native_false_decisions=0,all_native_intervals_contain_full=True,
        assignment_agreement_minimum_percent=min(float(x['agreement_percent']) for x in quality),
        completed_aggregate_proofs=aggregate_count,completed_switching_proofs=batch_proofs,total_switching_proofs=total,
        independent_public_full_replay_complete=(root/'public/group_verify.json').exists() and json.loads((root/'public/group_verify.json').read_text()).get('complete',False),
        average_completed_switching_prove_seconds=float(np.mean(batch_times)) if batch_times else None,
        median_completed_switching_prove_seconds=float(np.median(batch_times)) if batch_times else None,
        remaining_switching_generation_seconds_at_observed_mean=pending*float(np.mean(batch_times)) if batch_times else None,
        timing_note='Observed elapsed active work is a lower bound on final cost. Projected switching time, if present, excludes remaining setup/cache loads, witness/native work, verification and possible Full fallback; it is not a complete runtime forecast. Offline/RAM queue waits excluded. Previous runs were not interleaved across laptop clock/load conditions.',
        certificate_note='All-update intervals below are independently computed native results. Only completed batch proofs have been checked against them. Native conclusive outcomes are not yet fully replayed public certificates. The full selective-refinement proof is not implemented.',
        sha256=dict(state=state_sha,input=input_sha,approved_task=task_sha,progress=records_sha),updates_detail=checks)
    out=root/'partial_checks';out.mkdir(exist_ok=True)
    (out/f'snapshot_{int(started)}.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in snapshot.items() if k not in ['updates_detail','sha256']}),flush=True)


if __name__=='__main__':main()
