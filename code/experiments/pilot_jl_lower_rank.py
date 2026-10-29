"""Post-hoc lower-rank follow-up to the training-aware JL feasibility screen."""
from pathlib import Path
import argparse,csv,json,os,re,time
import numpy as np
from threadpoolctl import threadpool_limits
from commitjl.pilot_io import save,sha,array_sha,sign,frozen_image_sample
from commitjl.jl_followup import covariance_jl_basis
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from experiments.pilot_jl_certificate_ideas import cases,distances

REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--proof-queue',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();records=[];times=[]
    save(out/'protocol.json',dict(ranks=[71,75,79],powers=[1,2],seeds=[20261101,20261102,20261103],selection='All6 previously unresolved natural head regressions. Post-hoc development extension of the completed JL screen; every rank/power/seed retained. Trained residual SVD at equal rank is the paired control.',scope='Numerical only; no new proofs or proof-speed claim. No untrained extension is advertised as a JL theorem. All bases use training only and preserve the original63 prefix.',runner_sha256=sha(Path(__file__))))
    with threadpool_limits(limits=1):
        for fit,model,x,updates,learned,pin in cases('quick'):
            # Never deliberately overlap this additional development pilot with
            # the measured proof stages. Native screens may share laptop CPUs.
            while a.proof_queue.exists():
                queue=json.loads(a.proof_queue.read_text())
                if queue.get('phase') in (None,'native_replication') or queue.get('complete'):break
                save(out/'status.json',dict(complete=False,status='waiting_for_proof_queue',pid=os.getpid(),phase=queue.get('phase')));time.sleep(15)
            match=re.match(r'(mnist|fashion_mnist)_k\d+_seed(\d+)_',fit);dataset,seed=match.group(1),int(match.group(2));sample=frozen_image_sample(REPO,dataset);sample=sample[np.linalg.norm(sample,axis=1)>0];idx=np.random.default_rng(741+seed).permutation(len(sample));tr=sample[idx[:int(.6*len(sample))]];training=np.rint(tr*(4095/float(np.abs(tr).max()))).astype(np.int64);models={}
            for rank in [71,75,79]:
                models[f'trained_r{rank}']=dict(model,basis=learned[:rank])
                for power in [1,2]:
                    for sketch_seed in [20261101,20261102,20261103]:
                        name=f'jl_cov_r{rank}_p{power}_s{sketch_seed}';tick=time.monotonic();models[name]=covariance_jl_basis(model,training,rank-63,sketch_seed,power);times.append(dict(fit=fit,variant=name,training_sha256=array_sha(training),seconds=time.monotonic()-tick))
            old=np.asarray(model['centers'],dtype=np.int64);od=distances(x,old);prepared=prepare_groups(x,od.argmin(1).tolist(),len(old));den=int(model['basis_scale'])**6
            for s,new in updates:
                truth=int(distances(x,new).min(1).sum()-od.min(1).sum())
                for name,m in models.items():
                    b=exact_old_correction_bounds(m,x,prepared,old,new);lo,hi=b['change_lower_scaled'],b['change_upper_scaled'];decision=sign(lo,hi);assert lo<=truth*den<=hi and decision in ['inconclusive',sign(truth,truth)]
                    records.append(dict(fit=fit,set=s,variant=name,rank=len(m['basis']),truth=truth,decision=decision,lower=str(lo),upper=str(hi),denominator=str(den)))
            with (out/'results.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
            save(out/'fitting.json',times);save(out/'status.json',dict(complete=False,status='running',pid=os.getpid(),active_fit=fit,rows=len(records),elapsed_seconds=time.monotonic()-started));print(json.dumps(dict(fit=fit,rows=len(records))),flush=True)
    results=[]
    for name in sorted({r['variant'] for r in records}):
        g=[r for r in records if r['variant']==name];assert len(g)==6;results.append(dict(variant=name,rank=g[0]['rank'],requests=len(g),conclusive=sum(r['decision']!='inconclusive' for r in g)))
    summary=dict(complete=True,scope='Post-hoc numerical stress test, not independent validation or proof cost.',new_proofs=0,all_intervals_contain_truth=True,false_conclusive=0,elapsed_seconds=time.monotonic()-started,results=results);save(out/'summary.json',summary);save(out/'status.json',dict(complete=True,status='complete',pid=os.getpid()));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
