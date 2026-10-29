"""Frozen-fit screen of old-gap movement bounds; no proof-speed claim."""
from pathlib import Path
import argparse,json,os,time
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from experiments.review_update_decisions import array_sha
from experiments.pilot_certificate_refinement import sign
from data.loaders import load_dataset
from commitjl.group_moment_reference import prepare_groups
from commitjl.old_gap_bounds import old_gap_correction_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--all-regressions',action='store_true');p.add_argument('--minimum-only',action='store_true');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2];started=time.time();results=[];datasets={};audits={};pins_used={}
    sources=[('historical',repo/'paper_results/review_controls/local_group_moments_extended_20261023'),('fresh',repo/'paper_results/review_controls/group_change_confirmation_20261024/source')]
    save(out/'protocol.json',dict(scope='Native development screen only. Original prediction basis/outputs unchanged. Exact old distance gaps would require authenticated preprocessing and query proofs; neither implementation nor speed is claimed.',selection='All36 natural head regressions across the frozen80-archive suite.' if a.all_regressions else 'All6 head regressions unresolved by the existing exact-old numerical certificate. Development stress cohort, not independent validation.',minimum_only=a.minimum_only,posthoc_development=True,thread_limit=1))
    with threadpool_limits(limits=1):
        for cohort,source in sources:
            baseline=repo/'code/validation/outputs/group_exact_confirmation_20261024'/cohort/'results.csv'
            previous=pd.read_csv(baseline);selected=previous[(previous.method=='learned')&(previous.variant=='group_exact_old')&(previous.truth=='worsens')&(previous.budget_fraction==0)]
            if not a.all_regressions:selected=selected[selected.decision=='inconclusive']
            pins={r['fit']:r for r in json.loads((source/'pins.json').read_text())}
            for fit,requests in selected.groupby('fit',sort=True):
                if psutil.virtual_memory().available<750*2**20:raise RuntimeError('RAM reserve; preserve partial numerical results')
                item=requests.iloc[0];dataset=item.dataset;seed=int(item.seed);k=int(item.k);pin=pins[fit];archive=source/(fit+'.npz');frozen=np.load(archive)
                if dataset not in datasets:
                    x=np.asarray(load_dataset(dataset,n_max=5000,seed=0)['X'],float);datasets[dataset]=x[np.linalg.norm(x,axis=1)>0]
                if (dataset,seed) not in audits:
                    x=datasets[dataset];indices=np.random.default_rng(741+seed).permutation(len(x));cut=int(.6*len(x));tr=x[indices[:cut]];scale=4095/max(float(np.abs(tr).max()),1e-12)
                    rows=np.clip(np.rint(x[indices[cut:cut+1024]]*scale),-4096,4096).astype(np.int64);assert array_sha(rows)==pin['audit_sha256'];assert array_sha(np.rint(tr*scale).astype(np.int64))==pin['training_sha256'];audits[dataset,seed]=rows
                rows=audits[dataset,seed];old=frozen['centers'][0];basis=frozen['basis'];assert array_sha(basis)==pin['basis_sha256'];groups=((rows[:,None,:]-old[None,:,:])**2).sum(2).argmin(1).tolist();prepared=prepare_groups(rows,groups,k)
                model=dict(basis=basis.tolist(),basis_scale=16384,centers=old.tolist(),input_bits=12);pins_used[fit]=dict(archive_sha256=sha(archive),pins_sha256=sha(source/'pins.json'))
                for _,request in requests.iterrows():
                    s=int(request['set']);new=frozen['centers'][s];bounds=old_gap_correction_bounds(model,rows,prepared,old,new,minimum_only=a.minimum_only)
                    truth=int(request.true_change);lo,hi=bounds['change_lower_scaled'],bounds['change_upper_scaled'];assert lo<=truth*bounds['cost_denominator']<=hi;decision=sign(lo,hi);assert decision in ['inconclusive','worsens']
                    results.append(dict(fit=fit,cohort=cohort,set=s,true_change=truth,previous_decision=request.decision,decision=decision,lower=lo,upper=hi,denominator=bounds['cost_denominator'],additional_private_scalars_per_row=bounds['additional_private_scalars_per_row']))
                    pd.DataFrame(results).to_csv(out/'results.csv',index=False);save(out/'status.json',dict(complete=False,pid=os.getpid(),requests=len(results),active_fit=fit,active_set=s,detected=sum(r['decision']=='worsens' for r in results),elapsed_seconds=time.time()-started));print(json.dumps(dict(fit=fit,set=s,decision=decision)),flush=True)
    assert len(results)==(36 if a.all_regressions else 6)
    summary=dict(complete=True,requests=len(results),detected=sum(r['decision']=='worsens' for r in results),false_conclusive=0,all_intervals_contain_truth=True,minimum_only=a.minimum_only,native_only=True,elapsed_seconds=time.time()-started,source_hashes=pins_used)
    save(out/'summary.json',summary);save(out/'status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
