"""Training-only auxiliary residual directions on all six unresolved cases.

The original63-dimensional prediction basis is frozen. This native development
screen does not implement an adaptive-rank proof or claim its speed.
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np
import pandas as pd
import psutil
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from experiments.review_update_decisions import array_sha
from experiments.pilot_certificate_refinement import sign
from data.loaders import load_dataset
from commitjl.group_moment_reference import prepare_groups
from commitjl.group_change_bounds import exact_old_correction_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--all-stored-updates',action='store_true');p.add_argument('--ranks',type=int,nargs='+',choices=[79,95,127],default=[79,95,127]);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2];sources={'historical':repo/'paper_results/review_controls/local_group_moments_extended_20261023','fresh':repo/'paper_results/review_controls/group_change_confirmation_20261024/source'}
    cohorts=[]
    for cohort,source in sources.items():
        previous=pd.read_csv(repo/'code/validation/outputs/group_exact_confirmation_20261024'/cohort/'results.csv');part=previous[(previous.method=='learned')&(previous.variant=='group_exact_old')&(previous.budget_fraction==0)].copy();part['source_cohort']=cohort
        if not a.all_stored_updates:part=part[(part.truth=='worsens')&(part.decision=='inconclusive')]
        cohorts.append(part)
    selected=pd.concat(cohorts,ignore_index=True);assert len(selected)==(160 if a.all_stored_updates else 6)
    results=[];started=time.time();datasets={};checks={}
    save(out/'protocol.json',dict(scope='Native feasibility only; no adaptive-rank proof or measured speed. Original prediction basis remains63 and unchanged. Rank extension is fitted to original training data only; no audit rows/labels used for fitting. Combined intervals would require authenticating both representations and charging both proof paths.',selection='Every160 stored head update including36 natural regressions, fixed before this complete-cohort screen. Stored historical/confirmation fits, not new independently fitted prediction models.' if a.all_stored_updates else 'All six zero-refinement natural head regressions previously unresolved. This is explicitly a post-hoc development stress cohort, requiring further confirmation before adoption.',ranks=[63]+a.ranks,scale_bits=14,rank_extension='QR span of original rounded basis, followed by uncentered randomized SVD of the training residual, seed20261025, four iterations, always64 components. Preserve original63 rows exactly.',threads=1))
    with threadpool_limits(limits=1):
        for (cohort,fit),requests in selected.groupby(['source_cohort','fit'],sort=True):
            if psutil.virtual_memory().available<850*2**20:raise RuntimeError('RAM reserve; preserve partial results')
            source=sources[cohort];pins={r['fit']:r for r in json.loads((source/'pins.json').read_text())};item=requests.iloc[0];dataset=item.dataset;seed=int(item.seed);k=int(item.k);pin=pins[fit];archive=source/(fit+'.npz');frozen=np.load(archive)
            if dataset not in datasets:
                x=np.asarray(load_dataset(dataset,n_max=5000,seed=0)['X'],float);datasets[dataset]=x[np.linalg.norm(x,axis=1)>0]
            x=datasets[dataset];indices=np.random.default_rng(741+seed).permutation(len(x));cut=int(.6*len(x));tr=x[indices[:cut]];scale=4095/max(float(np.abs(tr).max()),1e-12);training=np.rint(tr*scale).astype(np.int64);rows=np.clip(np.rint(x[indices[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
            assert array_sha(rows)==pin['audit_sha256'] and array_sha(training)==pin['training_sha256'];basis=frozen['basis'];assert basis.shape==(63,784) and array_sha(basis)==pin['basis_sha256'];old=frozen['centers'][0]
            fit_started=time.monotonic();q=np.linalg.qr(basis.T.astype(float),mode='reduced')[0];w=training.astype(float)-old[0];residual=w-(w@q)@q.T
            _,_,directions=randomized_svd(residual,n_components=64,n_iter=4,random_state=20261025);extra=np.rint(directions*16384).astype(np.int64);fitting=time.monotonic()-fit_started
            np.savez_compressed(out/(fit+'.npz'),original_basis=basis,auxiliary_basis=np.concatenate((basis,extra)),training_sha256=np.asarray(pin['training_sha256']));checks[fit]=dict(source_archive_sha256=sha(archive),auxiliary_archive_sha256=sha(out/(fit+'.npz')),auxiliary_fitting_seconds=fitting)
            groups=((rows[:,None,:]-old[None,:,:])**2).sum(2).argmin(1).tolist();prepared=prepare_groups(rows,groups,k)
            for _,request in requests.iterrows():
                s=int(request['set']);new=frozen['centers'][s];truth=int(((rows[:,None,:]-new[None,:,:])**2).sum(2).min(1).sum()-((rows[:,None,:]-old[None,:,:])**2).sum(2).min(1).sum());assert truth==int(request.true_change);true_decision=sign(truth,truth);den=16384**6;previous_lo,previous_hi=int(request.lower),int(request.upper);assert previous_lo<=truth*den<=previous_hi
                results.append(dict(fit=fit,cohort=cohort,set=s,rank=63,decision=request.decision,combined_with_63=request.decision,true_change=truth,truth=true_decision,lower=previous_lo,upper=previous_hi,denominator=den,previously_recorded=True))
                for rank in a.ranks:
                    model=dict(basis=np.concatenate((basis,extra[:rank-63])).tolist(),basis_scale=16384,centers=old.tolist(),input_bits=12);bounds=exact_old_correction_bounds(model,rows,prepared,old,new);lo,hi=bounds['change_lower_scaled'],bounds['change_upper_scaled'];assert lo<=truth*den<=hi;decision=sign(lo,hi);assert decision in ['inconclusive',true_decision];combined_lo,combined_hi=max(lo,previous_lo),min(hi,previous_hi);assert combined_lo<=truth*den<=combined_hi
                    results.append(dict(fit=fit,cohort=cohort,set=s,rank=rank,decision=decision,combined_with_63=sign(combined_lo,combined_hi),true_change=truth,truth=true_decision,lower=lo,upper=hi,denominator=den,previously_recorded=False));pd.DataFrame(results).to_csv(out/'results.csv',index=False);save(out/'status.json',dict(complete=False,pid=os.getpid(),active_fit=fit,active_set=s,rank=rank,records=len(results),elapsed_seconds=time.time()-started));print(json.dumps(dict(fit=fit,set=s,rank=rank,decision=decision)),flush=True)
    frame=pd.DataFrame(results);summary=dict(complete=True,unique_requests=len(selected),results_by_rank={str(rank):dict(requests=len(g),conclusive=int((g.decision!='inconclusive').sum()),regressions=int((g.truth=='worsens').sum()),detected=int(((g.decision=='worsens')&(g.truth=='worsens')).sum()),combined_conclusive=int((g.combined_with_63!='inconclusive').sum()),combined_detected=int(((g.combined_with_63=='worsens')&(g.truth=='worsens')).sum())) for rank,g in frame.groupby('rank')},all_intervals_contain_truth=True,false_conclusive=0,native_only=True,prediction_model_unchanged=True,source_checks=checks,elapsed_seconds=time.time()-started)
    save(out/'summary.json',summary);save(out/'status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
