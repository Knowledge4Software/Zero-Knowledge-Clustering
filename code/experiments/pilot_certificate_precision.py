"""Screen certificate-only precision; keep prediction basis and all updates frozen."""
from pathlib import Path
import argparse,json,time,copy
import numpy as np
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);v=json.loads(a.input.read_text());original=v['model'];x=np.asarray(v['x'],dtype=np.int64);old=np.asarray(original['centers'],dtype=np.int64)
    groups=((x[:,None,:]-old[None,:,:])**2).sum(2).argmin(1);prepared=prepare_groups(x,groups.tolist(),original['k']);old_cost=int(((x[:,None,:]-old[None,:,:])**2).sum(2).min(1).sum());records=[]
    save(out/'protocol.json',dict(input_sha256=sha(a.input),prediction_basis_unchanged=True,certificate_scale_bits=[6,8,10,12,14],selection='Every nonidentity source update. Quantization changes only the auxiliary certificate basis, not frozen prediction outputs. No refitting.',scope='Native screening; no proof-speed claim.'))
    with threadpool_limits(limits=1):
        for bits in [6,8,10,12,14]:
            model=copy.deepcopy(original);scale=2**bits;model['basis_scale']=scale;model['basis']=np.rint(np.asarray(original['basis'],float)*scale/original['basis_scale']).astype(np.int64).tolist();started=time.monotonic()
            for s,centers in enumerate(v['reuse_centers']):
                if centers==original['centers']:continue
                new=np.asarray(centers,dtype=np.int64);truth=int(((x[:,None,:]-new[None,:,:])**2).sum(2).min(1).sum())-old_cost
                bounds=exact_old_correction_bounds(model,x,prepared,old,new);lo,hi=bounds['change_lower_scaled'],bounds['change_upper_scaled'];den=bounds['cost_denominator'];assert lo<=truth*den<=hi
                decision='improvement' if hi<0 else 'worsening' if lo>0 else 'equal' if lo==hi==0 else 'inconclusive';assert decision=='inconclusive' or decision==('improvement' if truth<0 else 'worsening' if truth>0 else 'equal')
                records.append(dict(bits=bits,set=s,truth=truth,lower=str(lo),upper=str(hi),decision=decision,width_over_old_cost_percent=100*(hi-lo)/den/old_cost))
                save(out/'results.json',records)
            print(json.dumps(dict(bits=bits,conclusive=sum(r['bits']==bits and r['decision']!='inconclusive' for r in records),updates=19,seconds=time.monotonic()-started)),flush=True)
    save(out/'summary.json',dict(complete=True,input_sha256=sha(a.input),all_intervals_contain_full=True,no_wrong_conclusions=True,by_bits={str(bits):dict(conclusive=sum(r['bits']==bits and r['decision']!='inconclusive' for r in records),updates=sum(r['bits']==bits for r in records)) for bits in [6,8,10,12,14]},prediction_basis_unchanged=True))


if __name__=='__main__':main()
