"""Fixed-budget ensemble JL confirmation and covariance-sketch stress test."""
from pathlib import Path
import argparse,csv,json,os,re,time
from math import ceil
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from commitjl.pilot_io import save,sha,array_sha,sign,frozen_image_sample
from commitjl.jl_followup import ensemble_scores,covariance_jl_basis
from commitjl.jl_certificate_pilots import hint_context,jl_map,hint_order,refinement_work
from commitjl.reuse_certificates import integer_stats
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from experiments.pilot_jl_certificate_ideas import cases,distances,prediction_labels

REPO=Path(__file__).resolve().parents[2]


def write_csv(path,records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)


def wait_ram(out,stage):
    while psutil.virtual_memory().available<900*2**20:
        save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=os.getpid(),stage=stage));time.sleep(15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();records=[];pins={}
    seeds=[20261101,20261102,20261103]
    variants=[('single64',64,1,1.,False),('median3x16',16,3,1.,False),('median3x32',32,3,1.,False),('median3x64',64,3,1.,False),('median3x16_shrink',16,3,.125,False),('median3x16_gate',16,3,.125,True),('median3x32_gate',32,3,.125,True)]
    save(out/'protocol.json',dict(seeds=seeds,variants=variants,ensemble_requests='All160 stored head updates; all variants retained. These are prior development fits, not a new-fit confirmation.',subspace_requests='All6 previously unresolved natural head regressions; post-hoc stress set, not held-out selection.',subspace_powers=[0,1,2],extension=16,oversampling=8,precision=14,refinement_budget=.05,selection='Parameters and seeds fixed before reading outcomes; no best-seed selection.',scope='Native feasibility only. Gating changes untrusted predictions/hints, never certificate endpoints. Refined corrections still need new authenticated exact-replacement proofs. Power sketches reuse existing deterministic bounds; no new JL theorem or novelty claim.',runner_sha256=sha(Path(__file__)),helper_sha256=sha(REPO/'code/commitjl/jl_followup.py')))
    with threadpool_limits(limits=1):
        for fit,model,x,updates,learned,pin in cases('all'):
            wait_ram(out,'ensemble');pins[fit]=pin;h,_=integer_stats(model,x)
            maps={(w,j):jl_map(x.shape[1],w,seeds[j]) for w in [16,32,64] for j in range(3)}
            for s,new in updates:
                context=hint_context(model,x,new,[0]*len(x));truth=distances(x,new).argmin(1);original=prediction_labels(model,h,new);control=context['central'].argmin(1)
                for name,w,count,alpha,gate in variants:
                    tick=time.monotonic();scores=ensemble_scores(context,[maps[w,j] for j in range(count)],alpha,gate);labels=scores.argmin(1)
                    records.append(dict(idea='prediction',fit=fit,set=s,variant=name,n=len(x),original_correct=int((original==truth).sum()),gram_control_correct=int((control==truth).sum()),correct=int((labels==truth).sum()),stored_jl_values_per_row=w*count,native_seconds=time.monotonic()-tick))
            write_csv(out/'prediction.csv',[r for r in records if r['idea']=='prediction']);save(out/'status.json',dict(complete=False,status='running',pid=os.getpid(),stage='ensemble',completed_requests=sum(r['variant']=='single64' for r in records)));print(json.dumps(dict(stage='ensemble',fit=fit)),flush=True)
        for fit,model,x,updates,learned,pin in cases('quick'):
            wait_ram(out,'covariance_basis');match=re.match(r'(mnist|fashion_mnist)_k\d+_seed(\d+)_',fit);dataset,seed=match.group(1),int(match.group(2));sample=frozen_image_sample(REPO,dataset);sample=sample[np.linalg.norm(sample,axis=1)>0];idx=np.random.default_rng(741+seed).permutation(len(sample));tr=sample[idx[:int(.6*len(sample))]];training=np.rint(tr*(4095/float(np.abs(tr).max()))).astype(np.int64)
            models={'base63':model,'trained79':dict(model,basis=learned)};fit_times={}
            for power in [0,1,2]:
                for sketch_seed in seeds:
                    name=f'jl_cov_p{power}_s{sketch_seed}';tick=time.monotonic();models[name]=covariance_jl_basis(model,training,16,sketch_seed,power);fit_times[name]=time.monotonic()-tick
            save(out/(fit+'_fitting.json'),dict(training_sha256=array_sha(training),fit_seconds=fit_times))
            old=np.asarray(model['centers'],dtype=np.int64);old_dist=distances(x,old);groups=old_dist.argmin(1);prepared=prepare_groups(x,groups.tolist(),len(old));maps={(w,j):jl_map(x.shape[1],w,seeds[j]) for w in [16,32,64] for j in range(3)}
            for s,new in updates:
                nd=distances(x,new);truth=int(nd.min(1).sum()-old_dist.min(1).sum());den=int(model['basis_scale'])**6
                for name,m in models.items():
                    b=exact_old_correction_bounds(m,x,prepared,old,new);lo,hi=b['change_lower_scaled'],b['change_upper_scaled'];assert lo<=truth*den<=hi and sign(lo,hi) in ['inconclusive',sign(truth,truth)]
                    records.append(dict(idea='basis',fit=fit,set=s,variant=name,n=len(x),truth=truth,decision=sign(lo,hi),lower=str(lo),upper=str(hi),denominator=str(den)))
                for rank in [63,79]:
                    m=models['base63' if rank==63 else 'trained79'];b=exact_old_correction_bounds(m,x,prepared,old,new);context=hint_context(m,x,new,groups);exact=[int(v)*den for v in nd.min(1)-nd[np.arange(len(x)),groups]];cap=ceil(.05*len(x))
                    scores={'width':None}
                    for name,w,count,alpha,gate in variants:
                        # Refinement estimates use the full residual correction;
                        # predictor-only alpha/gates are not duplicated here.
                        if alpha!=1.:continue
                        scores[name]=ensemble_scores(context,[maps[w,j] for j in range(count)])
                    for name,score in scores.items():
                        estimate=None if score is None else score.min(1)-score[np.arange(len(x)),groups];order=hint_order(b,estimate,'width' if score is None else 'predicted_gain');r=refinement_work(b,exact,order,cap);assert r['lower']<=truth*den<=r['upper'] and r['decision'] in ['inconclusive',sign(truth,truth)]
                        records.append(dict(idea='refinement',fit=fit,set=s,variant=f'r{rank}_{name}',n=len(x),truth=truth,decision=r['decision'],exact_rows=r['exact_rows'],lower=str(r['lower']),upper=str(r['upper']),denominator=str(den)))
            write_csv(out/'certificates.csv',[r for r in records if r['idea']!='prediction']);save(out/'status.json',dict(complete=False,status='running',pid=os.getpid(),stage='covariance_and_refinement',active_fit=fit));print(json.dumps(dict(stage='certificates',fit=fit)),flush=True)
    result=[]
    for idea in ['prediction','basis','refinement']:
        for variant in sorted({r['variant'] for r in records if r['idea']==idea}):
            g=[r for r in records if r['idea']==idea and r['variant']==variant];r=dict(idea=idea,variant=variant,requests=len(g))
            if idea=='prediction':
                n=sum(v['n'] for v in g);r.update(agreement_percent=100*sum(v['correct'] for v in g)/n,original_agreement_percent=100*sum(v['original_correct'] for v in g)/n,minimum_agreement_percent=min(100*v['correct']/v['n'] for v in g),improved=sum(v['correct']>v['original_correct'] for v in g),worsened=sum(v['correct']<v['original_correct'] for v in g),additional_errors=sum(v['original_correct']-v['correct'] for v in g),stored_jl_values_per_row=g[0]['stored_jl_values_per_row'])
            else:
                r.update(conclusive=sum(v['decision']!='inconclusive' for v in g))
                if idea=='refinement':r.update(total_exact_rows=sum(v['exact_rows'] for v in g),max_exact_rows=max(v['exact_rows'] for v in g))
            result.append(r)
    summary=dict(complete=True,native_only=True,new_proofs=0,elapsed_seconds=time.monotonic()-started,false_conclusive=0,all_intervals_contain_truth=True,results=result,source_hashes=pins);save(out/'summary.json',summary);save(out/'status.json',dict(complete=True,status='complete',pid=os.getpid(),elapsed_seconds=summary['elapsed_seconds']));print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'}),flush=True)


if __name__=='__main__':main()
